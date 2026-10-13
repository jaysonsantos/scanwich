import base64
import io
import json
import os
import random
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import ANY, patch

from PIL import Image

from scanwich.backends.openai_compatible import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL,
    PROVIDER_DETAIL_MAX_CHARACTERS,
    OpenAICompatibleBackend,
    _conversion_copy_modes,
    _jpeg_mode_image,
    _working_bytes,
)
from scanwich.models import OcrRegion, Point

LOGGER_NAME = "scanwich.backends.openai_compatible"


class FakeCompletions:
    """Answer the requests with the contents in sequence. The last content repeats."""

    def __init__(self, *contents: str, finish_reason: str | None = "stop") -> None:
        self.calls: list[dict[str, Any]] = []
        self._contents = contents
        self._finish_reason = finish_reason

    def create(self, **request: Any) -> SimpleNamespace:
        content = self._contents[min(len(self.calls), len(self._contents) - 1)]
        self.calls.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self._finish_reason,
                    message=SimpleNamespace(content=content),
                )
            ]
        )


def sent_image(request: dict[str, Any]) -> Image.Image:
    """Return the image that one request has."""
    prefix, encoded_image = request["messages"][1]["content"][1]["image_url"]["url"].split(",")
    assert prefix == "data:image/jpeg;base64", prefix
    image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    assert image.format == "JPEG", image.format
    return image


def box(text: str, left: float, top: float, right: float, bottom: float) -> dict[str, Any]:
    """Return a model region with an axis-aligned polygon."""
    return {
        "text": text,
        "polygon": [
            {"x": left, "y": top},
            {"x": right, "y": top},
            {"x": right, "y": bottom},
            {"x": left, "y": bottom},
        ],
    }


def corners(left: float, top: float, right: float, bottom: float) -> tuple[Point, ...]:
    return (Point(left, top), Point(right, top), Point(right, bottom), Point(left, bottom))


class OpenAICompatibleBackendTests(TestCase):
    def recognize_payload(
        self, payload: dict[str, Any], image_size: tuple[int, int]
    ) -> list[OcrRegion]:
        """Return the regions that the backend makes from one model payload."""
        completions = FakeCompletions(json.dumps(payload))
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", image_size, "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                return backend.recognize(image_path)

    def recognize_replies(
        self,
        completions: FakeCompletions,
        *,
        image: Image.Image | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[OcrRegion]:
        """Return the regions that the backend makes from the replies for one page."""
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            (image or Image.new("RGB", (2000, 3000), "white")).save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options=options or {})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                return backend.recognize(image_path)

    def recognition_error(self, reply: SimpleNamespace) -> RuntimeError:
        """Return the error that the backend raises for one provider reply."""
        completions = SimpleNamespace(create=lambda **_: reply)
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
                self.assertRaises(RuntimeError) as raised,
            ):
                backend.recognize(image_path)
        return raised.exception

    def cost_records(self, usages: Sequence[Any]) -> tuple[str, list[str]]:
        """Return the session ID and the cost records of a backend with one request per usage."""
        content = json.dumps({"coordinate_system": "normalized", "regions": []})
        choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))
        replies = iter([SimpleNamespace(choices=[choice], usage=usage) for usage in usages])
        completions = SimpleNamespace(create=lambda **_: next(replies))
        client_options: dict[str, Any] = {}

        def make_client(**options: Any) -> SimpleNamespace:
            client_options.update(options)
            return SimpleNamespace(chat=SimpleNamespace(completions=completions))

        fake_openai = SimpleNamespace(OpenAI=make_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
                self.assertLogs(LOGGER_NAME, level="INFO") as logs,
            ):
                for _ in usages:
                    backend.recognize(image_path)
        session_id = client_options["default_headers"]["X-Session-Id"]
        return session_id, [record for record in logs.output if "usage.cost" in record]

    def test_sends_image_and_converts_normalized_polygons_to_pixels(self) -> None:
        payload = {
            "coordinate_system": "normalized",
            "regions": [
                {
                    "text": "Olá 123",
                    "polygon": [
                        {"x": 100, "y": 200},
                        {"x": 1001, "y": 200},
                        {"x": 1001, "y": 400},
                        {"x": 100, "y": 400},
                    ],
                }
            ],
        }
        completions = FakeCompletions(f"```json\n{json.dumps(payload)}\n```")
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        client_calls: list[dict[str, Any]] = []

        def make_client(**options: Any) -> SimpleNamespace:
            client_calls.append(options)
            return fake_client

        fake_openai = SimpleNamespace(OpenAI=make_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["pt", "en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
                self.assertLogs(
                    "scanwich.backends.openai_compatible", level="INFO"
                ) as captured_logs,
            ):
                regions = backend.recognize(image_path)

        self.assertEqual(
            regions[0].polygon,
            (Point(20, 20), Point(200, 20), Point(200, 40), Point(20, 40)),
        )
        self.assertEqual(regions[0].text, "Olá 123")
        self.assertEqual(
            client_calls,
            [
                {
                    "api_key": "test-secret",
                    "base_url": "https://openrouter.ai/api/v1",
                    "timeout": 180.0,
                    "default_headers": {
                        "HTTP-Referer": "https://github.com/jaysonsantos/scanwich",
                        "X-Title": "Scanwich",
                        "X-Session-Id": ANY,
                    },
                }
            ],
        )
        request = completions.calls[0]
        self.assertEqual(request["model"], DEFAULT_MODEL)
        self.assertEqual(request["extra_body"], {"reasoning_effort": "low"})
        self.assertEqual(request["max_tokens"], 65_536)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        user_content = request["messages"][1]["content"]
        self.assertIn("Expected language codes: pt, en", user_content[0]["text"])
        self.assertIn('{"coordinate_system":"normalized","regions":[', user_content[0]["text"])
        self.assertIn("in this 200x100 pixel document image", user_content[0]["text"])
        image = sent_image(request)
        self.assertEqual((image.size, image.mode), ((200, 100), "RGB"))
        self.assertEqual(image.getpixel((100, 50)), (255, 255, 255))
        progress = "\n".join(captured_logs.output)
        self.assertIn("Initializing OpenAI-compatible client", progress)
        self.assertIn("OpenAI-compatible response received after", progress)

    def test_accepts_an_empty_regions_array(self) -> None:
        completions = FakeCompletions(
            json.dumps({"coordinate_system": "normalized", "regions": []})
        )
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (612, 792), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                self.assertEqual(backend.recognize(image_path), [])

    def test_rejects_truncated_completion(self) -> None:
        completions = FakeCompletions(
            json.dumps({"coordinate_system": "normalized", "regions": []}),
            finish_reason="length",
        )
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
                self.assertRaisesRegex(RuntimeError, "finish_reason='length'"),
            ):
                backend.recognize(image_path)
        # A new request cannot correct a reply that the provider stopped.
        self.assertEqual(len(completions.calls), 1)

    def test_reports_provider_error_for_a_reply_without_choices(self) -> None:
        no_choices = "OpenAI-compatible service returned no completion choices"
        provider_error = {
            "code": 502,
            "message": "Provider returned error",
            "metadata": {"flagged_input": "private text", "reasons": ["private reason"]},
        }
        self.assertEqual(
            str(self.recognition_error(SimpleNamespace(choices=None, error=provider_error))),
            f"{no_choices}: provider error code=502, message='Provider returned error'",
        )
        long_error = {"code": "overloaded", "message": "m" * 2 * PROVIDER_DETAIL_MAX_CHARACTERS}
        message = str(self.recognition_error(SimpleNamespace(choices=[], error=long_error)))
        self.assertTrue(message.startswith(f"{no_choices}: provider error code='overloaded'"))
        self.assertTrue(message.endswith("m..."))
        self.assertEqual(len(message) - len(no_choices), PROVIDER_DETAIL_MAX_CHARACTERS)
        for reply in (SimpleNamespace(choices=[]), SimpleNamespace(choices=[], error={})):
            self.assertEqual(str(self.recognition_error(reply)), no_choices)

    def test_reports_named_metadata_of_a_provider_error(self) -> None:
        no_choices = "OpenAI-compatible service returned no completion choices"
        metadata: dict[str, Any] = {
            "raw": {"error": {"type": "invalid_request_error"}},
            "flagged_input": "private text",
            "provider_code": "invalid_request",
            "provider_name": "Example",
            "error_type": "invalid_request",
            "model_slug": "provider/model",
        }
        provider_error = {"code": 400, "message": "Provider returned error", "metadata": metadata}
        self.assertEqual(
            str(self.recognition_error(SimpleNamespace(choices=None, error=provider_error))),
            f"{no_choices}: provider error code=400, message='Provider returned error', "
            "error_type='invalid_request', provider_name='Example', "
            "provider_code='invalid_request', raw={'error': {'type': 'invalid_request_error'}}",
        )
        metadata["raw"] = "r" * 2 * PROVIDER_DETAIL_MAX_CHARACTERS
        message = str(self.recognition_error(SimpleNamespace(choices=[], error=provider_error)))
        self.assertIn("provider_code='invalid_request', raw='rrr", message)
        self.assertTrue(message.endswith("r..."))
        self.assertEqual(len(message) - len(no_choices), PROVIDER_DETAIL_MAX_CHARACTERS)
        provider_error["metadata"] = {"error_type": "server", "provider_code": None, "raw": None}
        self.assertEqual(
            str(self.recognition_error(SimpleNamespace(choices=[], error=provider_error))),
            f"{no_choices}: provider error code=400, message='Provider returned error', "
            "error_type='server'",
        )

    def test_reports_native_finish_reason_without_model_output(self) -> None:
        incomplete = "OpenAI-compatible completion ended before OCR was complete: "
        message = SimpleNamespace(content="private letter", refusal="I cannot read this letter")
        choice = SimpleNamespace(
            finish_reason="content_filter", native_finish_reason="refusal", message=message
        )
        self.assertEqual(
            str(self.recognition_error(SimpleNamespace(choices=[choice]))),
            f"{incomplete}finish_reason='content_filter', native_finish_reason='refusal'",
        )
        choice = SimpleNamespace(finish_reason="length", native_finish_reason=None, message=message)
        self.assertEqual(
            str(self.recognition_error(SimpleNamespace(choices=[choice]))),
            f"{incomplete}finish_reason='length'",
        )

    def test_reports_provider_error_of_a_choice(self) -> None:
        stopped = (
            "OpenAI-compatible completion ended before OCR was complete: "
            "finish_reason='error', native_finish_reason='error'"
        )
        provider_error = {
            "code": 502,
            "message": "Upstream timeout",
            "metadata": {"error_type": "provider_unavailable"},
        }
        choice = SimpleNamespace(
            finish_reason="error",
            native_finish_reason="error",
            error=provider_error,
            message=SimpleNamespace(content="private letter"),
        )
        self.assertEqual(
            str(self.recognition_error(SimpleNamespace(choices=[choice]))),
            f"{stopped}: provider error code=502, message='Upstream timeout', "
            "error_type='provider_unavailable'",
        )
        choice.error = {"code": 502, "message": "m" * 2 * PROVIDER_DETAIL_MAX_CHARACTERS}
        message = str(self.recognition_error(SimpleNamespace(choices=[choice])))
        self.assertTrue(message.startswith(f"{stopped}: provider error code=502"))
        self.assertTrue(message.endswith("m..."))
        self.assertEqual(len(message) - len(stopped), PROVIDER_DETAIL_MAX_CHARACTERS)

    def test_sends_one_session_id_for_all_requests_of_a_backend(self) -> None:
        completions = FakeCompletions(
            json.dumps({"coordinate_system": "normalized", "regions": []})
        )
        client_headers: list[dict[str, str] | None] = []

        def make_client(**options: Any) -> SimpleNamespace:
            client_headers.append(options.get("default_headers"))
            return SimpleNamespace(chat=SimpleNamespace(completions=completions))

        fake_openai = SimpleNamespace(OpenAI=make_client)
        with (
            TemporaryDirectory() as temporary_directory,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            for options in ({}, {}, {"base_url": "https://example.invalid/v1"}):
                backend = OpenAICompatibleBackend(languages=["en"], options=options)
                backend.recognize(image_path)
                backend.recognize(image_path)

        # One client, and thus one header value, serves the two requests of a backend.
        self.assertEqual(len(completions.calls), 6)
        first, second, other_service = client_headers
        session_id = first["X-Session-Id"]
        self.assertEqual(str(uuid.UUID(session_id)), session_id)
        self.assertNotEqual(second["X-Session-Id"], session_id)
        self.assertIsNone(other_service)

    def test_logs_the_cost_of_each_request_and_the_session_total(self) -> None:
        session_id, records = self.cost_records(
            [SimpleNamespace(cost=0.0012), None, SimpleNamespace(cost=0.003)]
        )

        prefix = (
            f"INFO:{LOGGER_NAME}:OpenAI-compatible session {session_id}, model {DEFAULT_MODEL}: "
        )
        self.assertEqual(
            records,
            [
                prefix + "usage.cost=0.001200 for this request, total 0.001200 for 1 request(s)",
                prefix + "usage.cost=0.003000 for this request, total 0.004200 for 2 request(s)",
            ],
        )

    def test_logs_no_cost_when_the_service_reports_none(self) -> None:
        usages = [None, SimpleNamespace()]
        usages += [SimpleNamespace(cost=cost) for cost in (None, "0.1", True, float("nan"), -1.0)]

        self.assertEqual(self.cost_records(usages)[1], [])

    def test_requires_api_key_environment_variable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"),
            ):
                backend.recognize(image_path)

    def test_clamps_out_of_bounds_coordinates_to_the_declared_bounds(self) -> None:
        cases = (
            (
                "normalized",
                (10, 10),
                box("private text", 0, 0, 1051, 1000),
                corners(0, 0, 10, 10),
                (
                    "declared coordinate_system=normalized, used coordinate_system=normalized, "
                    "image=10x10 pixels, 2 of 4 points outside, largest distance 51.0"
                ),
            ),
            (
                "pixels",
                (612, 792),
                box("private text", 100, 700, 700, 900),
                corners(100, 700, 612, 792),
                (
                    "declared coordinate_system=pixels, used coordinate_system=pixels, "
                    "image=612x792 pixels, 3 of 4 points outside, largest distance 108.0"
                ),
            ),
        )
        for coordinate_system, image_size, region, polygon, summary in cases:
            with self.subTest(coordinate_system=coordinate_system):
                payload = {"coordinate_system": coordinate_system, "regions": [region]}
                with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
                    regions = self.recognize_payload(payload, image_size)
                self.assertEqual([region.polygon for region in regions], [polygon])
                self.assertEqual([region.text for region in regions], ["private text"])
                clamped = [line for line in logs.output if "were clamped" in line]
                self.assertEqual(len(clamped), 1)
                self.assertIn(summary, clamped[0])
                self.assertNotIn("private", "\n".join(logs.output))

    def test_uses_pixels_when_most_regions_are_far_outside_the_grid(self) -> None:
        cases = (
            (
                "portrait page",
                (2000, 3000),
                [(300, 300, 1800, 360), (300, 1700, 1800, 1760), (300, 500, 800, 560)],
                "2 of 3 regions",
            ),
            (
                "landscape page",
                (2000, 1000),
                [(100, 100, 1800, 160), (100, 400, 1800, 460), (100, 700, 1800, 760)],
                "3 of 3 regions",
            ),
        )
        for name, image_size, boxes, share in cases:
            with self.subTest(name):
                payload = {
                    "coordinate_system": "normalized",
                    "regions": [box("private text", *edges) for edges in boxes],
                }
                with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
                    regions = self.recognize_payload(payload, image_size)
                self.assertEqual(
                    [region.polygon for region in regions], [corners(*edges) for edges in boxes]
                )
                (record,) = logs.output
                self.assertIn(
                    f"declared normalized coordinates, but {share} are more than 250 units "
                    f"outside the 0-to-1000 grid and all points fit the "
                    f"{image_size[0]}x{image_size[1]} pixel image; using the coordinates as pixels",
                    record,
                )
                self.assertNotIn("private", record)

    def test_keeps_normalized_scaling_without_clear_evidence_of_pixels(self) -> None:
        # All these points also fit the image, but no point is far outside the grid.
        rows = range(1, 9)
        cases = (
            ("one region partly outside", [box("private text", 100, 100, 1051, 130)], 2),
            (
                "all regions partly outside",
                [box("private text", 100, 100 * row, 1060, 100 * row + 30) for row in rows],
                16,
            ),
        )
        for name, page, points_outside in cases:
            with self.subTest(name):
                payload = {"coordinate_system": "normalized", "regions": page}
                with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
                    regions = self.recognize_payload(payload, (2000, 3000))
                self.assertEqual(
                    [region.polygon for region in regions],
                    [corners(200, 300 * row, 2000, 300 * row + 90) for row in rows][: len(page)],
                )
                (record,) = logs.output
                self.assertIn("used coordinate_system=normalized", record)
                self.assertIn(f"{points_outside} of {4 * len(page)} points outside", record)

    def test_keeps_normalized_scaling_for_a_page_with_one_region(self) -> None:
        for bottom in (1200, 1400):
            with self.subTest(bottom=bottom):
                payload = {
                    "coordinate_system": "normalized",
                    "regions": [box("private text", 100, 950, 900, bottom)],
                }
                with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
                    regions = self.recognize_payload(payload, (2000, 3000))
                self.assertEqual(
                    [region.polygon for region in regions], [corners(200, 2850, 1800, 3000)]
                )
                (record,) = logs.output
                self.assertIn("used coordinate_system=normalized", record)
                self.assertIn(
                    f"2 of 4 points outside, largest distance {bottom - 1000:.1f}", record
                )

    def test_keeps_good_regions_of_a_page_with_out_of_bounds_regions(self) -> None:
        payload = {
            "coordinate_system": "normalized",
            "regions": [
                box("private good line one", 100, 100, 900, 130),
                box("private good line two", 100, 200, 900, 230),
                box("private good line three", 100, 300, 900, 330),
                box("private bad bottom line", 100, 960, 900, 1500),
            ],
        }
        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            regions = self.recognize_payload(payload, (2000, 3000))
        self.assertEqual(
            [region.polygon for region in regions],
            [
                corners(200, 300, 1800, 390),
                corners(200, 600, 1800, 690),
                corners(200, 900, 1800, 990),
                corners(200, 2880, 1800, 3000),
            ],
        )
        self.assertEqual(
            [region.text for region in regions],
            [
                "private good line one",
                "private good line two",
                "private good line three",
                "private bad bottom line",
            ],
        )
        (record,) = logs.output
        self.assertIn(
            "declared coordinate_system=normalized, used coordinate_system=normalized, "
            "image=2000x3000 pixels, 2 of 16 points outside, largest distance 500.0",
            record,
        )
        self.assertNotIn("private", record)

    def test_accepts_native_pixel_coordinates_for_a_page(self) -> None:
        payload = {
            "coordinate_system": "pixels",
            "regions": [
                {
                    "text": "pixel text",
                    "polygon": [
                        {"x": 100, "y": 120},
                        {"x": 500, "y": 120},
                        {"x": 500, "y": 180},
                        {"x": 100, "y": 180},
                    ],
                }
            ],
        }
        completions = FakeCompletions(json.dumps(payload))
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (612, 792), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                regions = backend.recognize(image_path)

        self.assertEqual(
            regions[0].polygon,
            (Point(100, 120), Point(500, 120), Point(500, 180), Point(100, 180)),
        )

    def test_sends_other_image_types_as_jpeg_with_a_white_background(self) -> None:
        empty = json.dumps({"coordinate_system": "normalized", "regions": []})
        white_palette = Image.new("P", (40, 20), 0)
        white_palette.putpalette([255, 255, 255])
        transparent_palette = Image.new("P", (40, 20), 0)
        transparent_palette.putpalette([0, 0, 0])
        transparent_palette.info["transparency"] = 0
        images = (
            transparent_palette,
            Image.new("RGBA", (40, 20), (0, 0, 0, 0)),
            Image.new("LA", (40, 20), (0, 0)),
            Image.new("L", (40, 20), 255),
            Image.new("1", (40, 20), 1),
            white_palette,
            # 16-bit gray, as some scanners write it.
            Image.new("I;16", (40, 20), 65_000),
        )
        for image in images:
            with self.subTest(mode=image.mode):
                completions = FakeCompletions(empty)
                self.recognize_replies(completions, image=image)
                sent = sent_image(completions.calls[0])
                self.assertEqual(sent.size, (40, 20))
                darkest, _ = sent.convert("L").getextrema()
                self.assertGreaterEqual(darkest, 250)

    def test_mixes_partly_transparent_pixels_with_white(self) -> None:
        completions = FakeCompletions(json.dumps({"coordinate_system": "pixels", "regions": []}))
        self.recognize_replies(completions, image=Image.new("RGBA", (40, 20), (0, 0, 0, 128)))
        sent = sent_image(completions.calls[0])
        self.assertEqual(sent.mode, "RGB")
        darkest, lightest = sent.convert("L").getextrema()
        self.assertAlmostEqual(darkest, 127, delta=3)
        self.assertAlmostEqual(lightest, 127, delta=3)

    def test_rejects_an_image_above_the_memory_limit_before_it_decodes_the_image(self) -> None:
        completions = FakeCompletions("{}")
        with (
            # An RGBA image of 1000x500 pixels needs 16 bytes for each pixel.
            patch("scanwich.backends.openai_compatible.MAX_IMAGE_WORKING_BYTES", 7 * 1024 * 1024),
            patch("scanwich.backends.openai_compatible._jpeg_mode_image") as convert,
            self.assertRaisesRegex(
                RuntimeError,
                "image of 1000x500 pixels needs 8 MiB of memory, and the limit is 7 MiB; "
                "use a lower DPI or a smaller image",
            ),
        ):
            self.recognize_replies(completions, image=Image.new("RGBA", (1000, 500)))
        convert.assert_not_called()
        self.assertEqual(completions.calls, [])

    def test_memory_estimate_has_each_copy_of_the_conversion(self) -> None:
        transparent_palette = Image.new("P", (10, 10), 0)
        transparent_palette.info["transparency"] = 0
        # The source, the copies of the conversion, and two copies for a resize.
        cases = (
            (Image.new("L", (10, 10)), 1 + 0 + 2 * 1, "L"),
            (Image.new("RGB", (10, 10)), 4 + 0 + 2 * 4, "RGB"),
            (Image.new("1", (10, 10)), 1 + 1 + 2 * 1, "L"),
            (Image.new("I;16", (10, 10)), 2 + (2 + 1) + 2 * 1, "L"),
            (Image.new("I", (10, 10)), 4 + (4 + 1) + 2 * 1, "L"),
            (Image.new("P", (10, 10)), 1 + 4 + 2 * 4, "RGB"),
            (Image.new("CMYK", (10, 10)), 4 + 4 + 2 * 4, "RGB"),
            (Image.new("RGBA", (10, 10)), 4 + 4 + 2 * 4, "RGB"),
            (Image.new("LA", (10, 10)), 4 + (4 + 4) + 2 * 4, "RGB"),
            (transparent_palette, 1 + (4 + 4) + 2 * 4, "RGB"),
        )
        for image, bytes_per_pixel, jpeg_mode in cases:
            with self.subTest(mode=image.mode, transparency="transparency" in image.info):
                self.assertEqual(_working_bytes(image), 100 * bytes_per_pixel)
                self.assertEqual(_jpeg_mode_image(image).mode, jpeg_mode)
                self.assertEqual((_conversion_copy_modes(image) or (image.mode,))[-1], jpeg_mode)

    def test_correction_attempt_reports_no_provider_text_of_a_completion_error(self) -> None:
        unusable = '{"coordinate_system": "normalized", "regions": [{"text": "private" letter"}]}'
        provider_error = {
            "code": 502,
            "message": "private letter",
            "metadata": {"raw": "private letter", "provider_name": "private provider"},
        }
        stopped = "OpenAI-compatible completion ended before OCR was complete: "
        message = SimpleNamespace(content="private letter")
        cases = (
            (
                SimpleNamespace(choices=None, error=provider_error),
                "OpenAI-compatible service returned no completion choices",
            ),
            (
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="error",
                            native_finish_reason="private letter",
                            error=provider_error,
                            message=message,
                        )
                    ]
                ),
                f"{stopped}finish_reason='error'",
            ),
            (
                SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="private letter", message=message)]
                ),
                f"{stopped}finish_reason=unknown",
            ),
        )
        for reply, expected in cases:
            with self.subTest(expected=expected):
                replies = iter([FakeCompletions(unusable).create(), reply])
                completions = SimpleNamespace(create=lambda replies=replies, **_: next(replies))
                with (
                    self.assertLogs(LOGGER_NAME, level="WARNING") as logs,
                    self.assertRaises(RuntimeError) as raised,
                ):
                    self.recognize_replies(completions)
                # The provider can repeat the text of the correction request in its error.
                self.assertEqual(str(raised.exception), expected)
                self.assertNotIn("private", "\n".join(logs.output))

    def test_scales_16_bit_gray_values_to_8_bits(self) -> None:
        completions = FakeCompletions(json.dumps({"coordinate_system": "pixels", "regions": []}))
        # Half of the 16-bit range must become half of the 8-bit range, not white.
        self.recognize_replies(completions, image=Image.new("I;16", (40, 20), 32_768))
        darkest, lightest = sent_image(completions.calls[0]).convert("L").getextrema()
        self.assertAlmostEqual(darkest, 128, delta=2)
        self.assertAlmostEqual(lightest, 128, delta=2)
        # Older Pillow versions open a 16-bit gray PNG in this mode.
        self.assertEqual(_jpeg_mode_image(Image.new("I", (4, 4), 32_768)).getextrema(), (128, 128))

    def test_keeps_the_exif_orientation_and_no_other_metadata(self) -> None:
        orientation_tag, make_tag = 0x0112, 0x010F
        empty = json.dumps({"coordinate_system": "normalized", "regions": []})
        noise = random.Random(0).randbytes(400 * 200 * 3)
        # A normal orientation needs no tag. The limit makes the backend scale the image down.
        cases = ((6, None, 6), (6, 20_000, 6), (1, None, None), (None, None, None))
        for orientation, byte_limit, expected in cases:
            with self.subTest(orientation=orientation, byte_limit=byte_limit):
                exif = Image.Exif()
                exif[make_tag] = "private camera"
                if orientation is not None:
                    exif[orientation_tag] = orientation
                completions = FakeCompletions(empty)
                backend = OpenAICompatibleBackend(languages=["en"], options={})
                backend._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
                limit = byte_limit or 10**9
                with (
                    TemporaryDirectory() as temporary_directory,
                    patch("scanwich.backends.openai_compatible.MAX_IMAGE_BYTES", limit),
                    self.assertLogs(LOGGER_NAME, level="INFO"),
                ):
                    image_path = Path(temporary_directory) / "page"
                    page = Image.frombytes("RGB", (400, 200), noise)
                    page.save(image_path, format="JPEG", quality=95, exif=exif)
                    backend.recognize(image_path)

                sent = sent_image(completions.calls[0])
                self.assertEqual(sent.getexif().get(orientation_tag), expected)
                self.assertNotIn(make_tag, sent.getexif())
                # The backend does not turn the pixels, so the regions fit the page image file.
                self.assertGreater(sent.width, sent.height)
                self.assertEqual(sent.size == (400, 200), byte_limit is None)

    def test_rejects_an_image_type_that_is_not_in_the_list(self) -> None:
        completions = FakeCompletions("{}")
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page"
            # A TIFF can have 32-bit values, and their range is not known.
            Image.new("I", (10, 10), 200).save(image_path, format="TIFF")
            backend = OpenAICompatibleBackend(languages=["en"], options={})
            backend._client = fake_client

            with self.assertRaisesRegex(RuntimeError, "does not support image type image/tiff"):
                backend.recognize(image_path)
        self.assertEqual(completions.calls, [])

    def test_failed_correction_request_reports_no_provider_text(self) -> None:
        class ProviderError(Exception):
            status_code = 403

        unusable = '{"coordinate_system": "normalized", "regions": [{"text": "private" letter"}]}'
        replies = iter([FakeCompletions(unusable).create(), ProviderError("private letter")])

        def create(**_: Any) -> SimpleNamespace:
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        completions = SimpleNamespace(create=create)
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING") as logs,
            self.assertRaises(RuntimeError) as raised,
        ):
            self.recognize_replies(completions)

        # The provider can repeat the text of the correction request in its error.
        self.assertEqual(
            str(raised.exception),
            f"OpenAI-compatible request failed for model {DEFAULT_MODEL} in attempt 2 of 3: "
            "ProviderError, HTTP status 403",
        )
        self.assertIsNone(raised.exception.__cause__)
        self.assertTrue(raised.exception.__suppress_context__)
        self.assertNotIn("private", "\n".join(logs.output))

    def test_scales_down_an_image_above_the_size_limit(self) -> None:
        noise = random.Random(0).randbytes(400 * 200 * 3)
        page = Image.frombytes("RGB", (400, 200), noise)
        limit = 20_000

        def recognize(coordinate_system: str, region: dict[str, Any]) -> tuple[Any, ...]:
            payload = {"coordinate_system": coordinate_system, "regions": [region]}
            completions = FakeCompletions(json.dumps(payload))
            with (
                patch("scanwich.backends.openai_compatible.MAX_IMAGE_BYTES", limit),
                self.assertLogs(LOGGER_NAME, level="WARNING") as logs,
            ):
                (region,) = self.recognize_replies(completions, image=page)
            too_large = [line for line in logs.output if "too large" in line]
            return region.polygon, completions.calls[0], too_large

        polygon, request, (record,) = recognize("normalized", box("text", 100, 100, 900, 900))
        user_content = request["messages"][1]["content"]
        encoded_image = user_content[1]["image_url"]["url"].split(",")[1]
        self.assertLessEqual(len(base64.b64decode(encoded_image)), limit)
        sent = sent_image(request)
        self.assertLess(sent.width, 400)
        self.assertAlmostEqual(sent.width / sent.height, 2, delta=0.05)
        size = f"{sent.width}x{sent.height}"
        self.assertIn(f"in this {size} pixel document image", user_content[0]["text"])
        self.assertIn(
            f"page image of 400x200 pixels is too large for one request; sending it as {size} "
            f"pixels ({len(base64.b64decode(encoded_image))} bytes)",
            record,
        )
        # The regions use the pixels of the page image, not of the smaller image.
        self.assert_polygons_almost_equal(polygon, corners(40, 20, 360, 180))
        polygon, _, _ = recognize("pixels", box("text", 0, 0, *sent.size))
        self.assert_polygons_almost_equal(polygon, corners(0, 0, 400, 200))

    def assert_polygons_almost_equal(
        self, polygon: Sequence[Point], expected: Sequence[Point]
    ) -> None:
        for point, expected_point in zip(polygon, expected, strict=True):
            self.assertAlmostEqual(point.x, expected_point.x)
            self.assertAlmostEqual(point.y, expected_point.y)

    def test_asks_the_model_to_correct_an_unusable_reply(self) -> None:
        good_region = box('private „letter" end', 100, 100, 900, 130)
        good = json.dumps(
            {"coordinate_system": "normalized", "regions": [good_region]}, ensure_ascii=False
        )
        cases = (
            (
                "quote without an escape",
                good.replace('\\"', '"'),
                (
                    "RuntimeError: OpenAI-compatible service returned invalid JSON: "
                    "Expecting ',' delimiter: line 1 column 76 (char 75)"
                ),
                (
                    "Your last reply cannot be used: it is not valid JSON (Expecting ',' "
                    "delimiter). The JSON parser stopped directly after this text: "
                    'stem": "normalized", "regions": [{"text": "private „letter" \n'
                ),
            ),
            (
                "point in the list of regions",
                json.dumps(
                    {"coordinate_system": "normalized", "regions": [good_region, {"x": 1, "y": 2}]}
                ),
                "TypeError: OpenAI-compatible OCR region 2 text must be a string",
                (
                    "Your last reply cannot be used: "
                    "OpenAI-compatible OCR region 2 text must be a string\n"
                ),
            ),
        )
        for name, unusable, defect, correction_start in cases:
            with self.subTest(name):
                completions = FakeCompletions(unusable, good)
                with self.assertLogs(LOGGER_NAME, level="INFO") as logs:
                    regions = self.recognize_replies(completions)

                self.assertEqual([region.text for region in regions], ['private „letter" end'])
                first, second = completions.calls
                self.assertEqual(len(first["messages"]), 2)
                self.assertEqual(second["messages"][:2], first["messages"])
                self.assertEqual(second["messages"][2], {"role": "assistant", "content": unusable})
                correction = second["messages"][3]
                self.assertEqual(correction["role"], "user")
                self.assertEqual(
                    correction["content"],
                    correction_start
                    + "Return the complete JSON object again, in the shape that the first "
                    'message gives. In a JSON string, write a double quote as \\" and a '
                    "backslash as \\\\. Return only the JSON object.",
                )
                # Only the messages change.
                del first["messages"], second["messages"]
                self.assertEqual(second, first)
                self.assertEqual(
                    [line for line in logs.output if "attempt" in line],
                    [
                        (
                            f"WARNING:{LOGGER_NAME}:OpenAI-compatible model {DEFAULT_MODEL} "
                            f"returned an unusable reply in attempt 1 of 3 ({defect})"
                        ),
                        (
                            f"INFO:{LOGGER_NAME}:OpenAI-compatible model {DEFAULT_MODEL} "
                            "corrected its reply in attempt 2 of 3"
                        ),
                    ],
                )
                self.assertNotIn("private", "\n".join(logs.output))

    def test_stops_after_the_maximum_number_of_attempts(self) -> None:
        unusable = '{"coordinate_system": "normalized", "regions": [{"text": "private" letter"}]}'
        self.assertEqual(DEFAULT_MAX_ATTEMPTS, 3)
        for options, attempts in (({}, 3), ({"max_attempts": 1}, 1), ({"max_attempts": 2}, 2)):
            with self.subTest(options=options):
                completions = FakeCompletions(unusable)
                with (
                    self.assertLogs(LOGGER_NAME, level="INFO") as logs,
                    self.assertRaisesRegex(RuntimeError, "returned invalid JSON$") as raised,
                ):
                    self.recognize_replies(completions, options=options)

                self.assertEqual(len(completions.calls), attempts)
                self.assertEqual(
                    [len(call["messages"]) for call in completions.calls],
                    [2 * attempt for attempt in range(1, attempts + 1)],
                )
                # Each unusable reply has one record, and the last reply too.
                warnings = [line for line in logs.output if line.startswith("WARNING")]
                self.assertEqual(
                    [line.split(" in attempt ")[1].split(" (")[0] for line in warnings],
                    [f"{attempt} of {attempts}" for attempt in range(1, attempts + 1)],
                )
                self.assertNotIn("private", str(raised.exception) + "\n".join(logs.output))

    def test_rejects_unknown_and_invalid_options(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported.*unknown"):
            OpenAICompatibleBackend(languages=["en"], options={"unknown": True})
        with self.assertRaisesRegex(TypeError, "max_tokens"):
            OpenAICompatibleBackend(languages=["en"], options={"max_tokens": 10.5})
        with self.assertRaisesRegex(ValueError, "timeout"):
            OpenAICompatibleBackend(languages=["en"], options={"timeout": 0})
        with self.assertRaisesRegex(ValueError, "max_attempts"):
            OpenAICompatibleBackend(languages=["en"], options={"max_attempts": 0})
