import asyncio
import base64
import io
import json
import os
import random
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import ANY, patch

from PIL import Image

from scanwich.backends import openai_compatible
from scanwich.backends.openai_compatible import (
    DEFAULT_LAYOUT_MODEL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MODEL,
    PROVIDER_DETAIL_MAX_CHARACTERS,
    OpenAICompatibleBackend,
    _conversion_copy_modes,
    _jpeg_mode_image,
    _working_bytes,
)
from scanwich.layout import (
    BLOCK_TYPES,
    Box,
    Figure,
    FigureLocation,
    Footnote,
    Heading,
    LayoutFormatError,
    ListBlock,
    ListItem,
    PageFurniture,
    PageLayout,
    Paragraph,
    Table,
    TableCell,
)
from scanwich.models import OcrRegion, Point
from scanwich.ocr import (
    AsyncFigureLocator,
    AsyncLayoutBackend,
    FigureLocator,
    LayoutBackend,
    LayoutSettings,
    NoReplyError,
    OcrBackend,
    RateLimitError,
)

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
            backend = OpenAICompatibleBackend(languages=["en"], options={"rotations": []})

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
            backend = OpenAICompatibleBackend(
                languages=["en"], options={"rotations": [], **(options or {})}
            )

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
            backend = OpenAICompatibleBackend(languages=["en"], options={"rotations": []})

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
            backend = OpenAICompatibleBackend(languages=["en"], options={"rotations": []})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
                self.assertLogs(LOGGER_NAME, level="INFO") as logs,
            ):
                for _ in usages:
                    backend.recognize(image_path)
        session_id = client_options["default_headers"]["X-Session-Id"]
        return session_id, [record for record in logs.output if "usage." in record]

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
            for options in (
                {"rotations": []},
                {"rotations": []},
                {"base_url": "https://example.invalid/v1", "rotations": []},
            ):
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

    def test_logs_the_tokens_of_each_request(self) -> None:
        session_id, records = self.cost_records(
            [
                SimpleNamespace(prompt_tokens=1800, completion_tokens=950, cost=0.002),
                SimpleNamespace(
                    prompt_tokens=1500,
                    completion_tokens=6200,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=5400),
                ),
                SimpleNamespace(
                    prompt_tokens=1500,
                    completion_tokens=800,
                    completion_tokens_details={"reasoning_tokens": 0},
                ),
                # A usage without the two counts as integers gives no record of tokens.
                SimpleNamespace(prompt_tokens=1800, cost=0.001),
                SimpleNamespace(prompt_tokens="1800", completion_tokens=True),
                SimpleNamespace(prompt_tokens=0, completion_tokens=0),
            ]
        )

        prefix = (
            f"INFO:{LOGGER_NAME}:OpenAI-compatible session {session_id}, model {DEFAULT_MODEL}: "
        )
        self.assertEqual(
            records,
            [
                prefix + "usage.prompt_tokens=1800, usage.completion_tokens=950 for this request",
                prefix + "usage.cost=0.002000 for this request, total 0.002000 for 1 request(s)",
                prefix
                + "usage.prompt_tokens=1500, usage.completion_tokens=6200, reasoning_tokens=5400 "
                "for this request",
                prefix
                + "usage.prompt_tokens=1500, usage.completion_tokens=800, reasoning_tokens=0 "
                "for this request",
                prefix + "usage.cost=0.001000 for this request, total 0.003000 for 2 request(s)",
                prefix + "usage.prompt_tokens=0, usage.completion_tokens=0 for this request",
            ],
        )

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


class LayoutRecognitionTests(TestCase):
    def recognize_layout(
        self,
        completions: FakeCompletions,
        *,
        image_size: tuple[int, int] = (1200, 800),
        languages: Sequence[str] = ("fr", "de"),
        options: dict[str, Any] | None = None,
        upright: bool = False,
    ) -> PageLayout:
        """Return the layout that the backend makes from the replies for one page."""
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", image_size, "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=languages, options=options or {})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                return backend.recognize_layout(image_path, upright=upright)

    def test_backend_recognizes_layouts(self) -> None:
        backend = OpenAICompatibleBackend(languages=["en"], options={})
        self.assertIsInstance(backend, LayoutBackend)
        self.assertIsInstance(backend, OcrBackend)

    def test_returns_the_layout_of_the_reply(self) -> None:
        payload = {
            "page_top": "top",
            "content_box": [100, 50, 900, 950],
            "blocks": [
                {"type": "page_number", "text": "406"},
                {"type": "heading", "level": 1, "text": "A <sc>heading</sc>"},
                {"type": "paragraph", "text": "A paragraph.<fn>1</fn>"},
                {
                    "type": "table",
                    "header_rows": 1,
                    "rules": "columns",
                    "rows": [["a", "b"], ["1", {"text": "2", "colspan": 1}]],
                },
                {"type": "figure", "box": [200, 300, 700, 600], "caption": "Fig. 1"},
                {"type": "footnote", "mark": "1", "text": "A footnote."},
            ],
        }

        layout = self.recognize_layout(FakeCompletions(json.dumps(payload)))

        self.assertEqual(
            layout,
            PageLayout(
                content_box=Box(100, 50, 900, 950),
                blocks=(
                    PageFurniture("page_number", "406"),
                    Heading("A <sc>heading</sc>"),
                    Paragraph("A paragraph.<fn>1</fn>"),
                    Table(
                        rows=(
                            (TableCell("a"), TableCell("b")),
                            (TableCell("1"), TableCell("2")),
                        ),
                        header_rows=1,
                        rules="columns",
                    ),
                    Figure(box=Box(200, 300, 700, 600), caption="Fig. 1"),
                    Footnote("A footnote.", mark="1"),
                ),
            ),
        )

    def test_asks_for_the_layout_as_json(self) -> None:
        completions = FakeCompletions('{"page_top":"top","blocks":[]}')

        self.assertEqual(self.recognize_layout(completions), PageLayout())

        (request,) = completions.calls
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(sent_image(request).size, (1200, 800))
        system_message, user_message = request["messages"]
        self.assertIn(
            "never translate, summarize, or infer missing text", system_message["content"]
        )
        prompt = user_message["content"][0]["text"]
        self.assertIn("This 1200x800 pixel image shows one page of a printed document.", prompt)
        self.assertIn("Expected language codes: fr, de.", prompt)
        box = '{"left":L,"top":T,"right":R,"bottom":B}'
        # The prompt names each field of the layout that the backend reads. A box has
        # names for its sides, so that their sequence is not in doubt.
        for field in (
            f'{{"page_top":"top","content_box":{box},"blocks":[...]}}',
            '"continued":true',
            '"header_rows"',
            '"colspan"',
            '"rowspan"',
            '"rules"',
            f'"box":{box}',
            '"caption"',
            '"mark"',
            "0-to-1000 grid",
            "<fn>mark</fn>",
            # The place of a running head and of a page number. The shape names no
            # place, so that the reply has the place of the page and not that of an example.
            '{"type":"page_number","text":"...","position":"..."}',
            "A left page and a right page of a book can have different places.",
            (
                '"top-left", "top-center", "top-right", "bottom-left", "bottom-center", or '
                '"bottom-right"'
            ),
            '"double_rule":true',
            # A lost tag must not change a capital form to a lowercase letter.
            "Capital forms. Write each letter in the case that the page prints.",
            "write its letters as capital letters, also in a caption and in a running head.",
            "Do not change such a line to lowercase letters or to sentence case.",
            "Put only them in <sc>, and write a large capital letter before the tag.",
            # The examples are two different forms, and not the form of one page.
            'write "P<sc>ART</sc> O<sc>NE</sc>" and not "Part One".',
            'write "<sc>THE END OF THE FIRST BOOK</sc>" and not "The end of the first book".',
            "Do not add an accent that the page does not print.",
            "the lines of a centered caption",
            # The writer needs the hyphen to know that the page boundary divides a word.
            (
                "When the last line of the page ends with such a hyphen, keep that hyphen at "
                "the end of the text: the rest of the word is on the next page."
            ),
            # The model does not read a page that is not upright.
            'Return only {"page_top":"<side>","blocks":[]}.',
            'Return {"page_top":"top","blocks":[]} for a page that has no content.',
        ):
            with self.subTest(field=field):
                self.assertIn(field, prompt)
        # The prompt names each block type that the layout parser knows.
        for block_type in BLOCK_TYPES:
            with self.subTest(block_type=block_type):
                self.assertIn(f'{{"type":"{block_type}"', prompt)

    def test_does_not_ask_for_the_direction_of_a_page_that_is_upright(self) -> None:
        # The caller knows the direction. A request for the direction lets the model
        # return no blocks, and the page is then not in the document.
        completions = FakeCompletions('{"blocks":[{"type":"paragraph","text":"Text."}]}')

        layout = self.recognize_layout(completions, upright=True)

        self.assertEqual(layout, PageLayout(blocks=(Paragraph("Text."),)))
        (request,) = completions.calls
        prompt = request["messages"][1]["content"][0]["text"]
        self.assertIn("The page is upright in the image.", prompt)
        self.assertIn(
            'Return one JSON object with this shape: {"content_box":'
            '{"left":L,"top":T,"right":R,"bottom":B},"blocks":[...]}',
            prompt,
        )
        self.assertIn('Return {"blocks":[]} for a page that has no content.', prompt)
        self.assertNotIn("page_top", prompt)
        self.assertNotIn("do not read the page", prompt)
        # The rest of the request is the same as for a page with an unknown direction.
        other_completions = FakeCompletions('{"blocks":[]}')
        self.recognize_layout(other_completions)
        (other_request,) = other_completions.calls
        other_prompt = other_request["messages"][1]["content"][0]["text"]

        def shared_part(text: str) -> str:
            return text[text.index("Coordinates.") : text.rindex("Return {")]

        self.assertEqual(shared_part(prompt), shared_part(other_prompt))
        self.assertGreater(len(shared_part(prompt)), 2000)

    def test_asks_the_model_to_correct_a_layout_that_loses_content(self) -> None:
        # The reply has a nested list. The layout has no place for it, so its text must
        # not disappear.
        secret = "private text of the page"
        nested = {"marker": "1.", "text": "a", "sublist": [{"marker": "a)", "text": secret}]}
        unusable = json.dumps({"blocks": [{"type": "list", "items": [nested]}]})
        flat = [{"marker": "1.", "text": "a"}, {"marker": "a)", "text": secret}]
        usable = json.dumps({"blocks": [{"type": "list", "items": flat}]})
        completions = FakeCompletions(unusable, usable)

        with self.assertLogs("scanwich", level="DEBUG") as captured_logs:
            layout = self.recognize_layout(completions)

        self.assertEqual(
            layout,
            PageLayout(
                blocks=(
                    ListBlock(items=(ListItem("a", marker="1."), ListItem(secret, marker="a)"))),
                )
            ),
        )
        self.assertEqual(len(completions.calls), 2)
        # The model gets the name of the field and the place of the object.
        self.assertEqual(
            completions.calls[1]["messages"][3]["content"].split("\n")[0],
            "Your last reply cannot be used: block 1 (list) item 1 has text in a field that "
            "the layout does not read; its fields are: marker, text. The layout does not read "
            'these fields: "sublist" in 1 object (block 1 (list) item 1). This rule is for '
            "each object of the reply. Remove these fields from all objects, or put their "
            "text in a field that the layout reads.",
        )
        # A log record has no text from the reply: no page text and no field name.
        log_output = "\n".join(captured_logs.output)
        self.assertNotIn(secret, log_output)
        self.assertNotIn("sublist", log_output)
        self.assertIn(
            "returned an unusable reply in attempt 1 of 3 (LayoutFormatError: block 1 (list) "
            "item 1 has text in a field that the layout does not read; its fields are: "
            "marker, text)",
            log_output,
        )

    def test_correction_request_names_a_field_that_each_block_has(self) -> None:
        # A model that adds one field mostly adds it to each block. A message that names
        # only the first block lets the model correct one block for each request.
        blocks = [
            {"type": "paragraph", "shelfmark": f"b{number}", "text": f"Paragraph {number}."}
            for number in range(1, 5)
        ]
        unusable = json.dumps({"page_top": "top", "blocks": blocks})
        completions = FakeCompletions(unusable)

        with (
            self.assertLogs("scanwich", level="DEBUG") as captured_logs,
            self.assertRaises(LayoutFormatError) as raised,
        ):
            self.recognize_layout(completions)

        self.assertEqual(len(completions.calls), DEFAULT_MAX_ATTEMPTS)
        for request in completions.calls[1:]:
            self.assertIn(
                'The layout does not read these fields: "shelfmark" in 4 objects (the first '
                "one is block 1 (paragraph)). This rule is for each object of the reply.",
                request["messages"][-1]["content"],
            )
        # The error of the conversion and the log records do not name the field.
        self.assertEqual(
            str(raised.exception),
            "block 1 (paragraph) has text in a field that the layout does not read; "
            "its fields are: type, text, align, continued",
        )
        self.assertNotIn("shelfmark", "\n".join(captured_logs.output))

    def test_reply_cannot_name_the_image_file_of_a_figure(self) -> None:
        # The pipeline makes the image file from the box. Text in an image field of a
        # reply would disappear.
        box = {"left": 100, "top": 200, "right": 800, "bottom": 900}
        unusable = json.dumps({"blocks": [{"type": "figure", "box": box, "image": "A wheel"}]})
        usable = json.dumps({"blocks": [{"type": "figure", "box": box, "caption": "A wheel"}]})
        completions = FakeCompletions(unusable, usable)

        with self.assertLogs(LOGGER_NAME, level="WARNING"):
            layout = self.recognize_layout(completions)

        self.assertEqual(
            layout,
            PageLayout(blocks=(Figure(box=Box(100, 200, 800, 900), caption="A wheel"),)),
        )
        self.assertIn(
            "block 1 (figure) has text in a field that the layout does not read; its fields "
            'are: type, box, caption. The layout does not read these fields: "image" in 1 '
            "object (block 1 (figure)).",
            completions.calls[1]["messages"][3]["content"],
        )

    def test_returns_the_side_of_a_page_that_is_not_upright(self) -> None:
        layout = self.recognize_layout(FakeCompletions('{"page_top":"right","blocks":[]}'))
        self.assertEqual(layout, PageLayout(page_top="right"))

    def test_asks_the_model_to_correct_a_layout_that_it_cannot_use(self) -> None:
        # The box uses the pixels of the image and not the grid.
        unusable = json.dumps({"blocks": [{"type": "figure", "box": [100, 200, 2400, 3100]}]})
        usable = json.dumps({"blocks": [{"type": "figure", "box": [100, 200, 800, 900]}]})
        completions = FakeCompletions(unusable, usable)

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            layout = self.recognize_layout(completions)

        self.assertEqual(layout, PageLayout(blocks=(Figure(box=Box(100, 200, 800, 900)),)))
        first_request, second_request = completions.calls
        self.assertEqual(second_request["messages"][:2], first_request["messages"])
        self.assertEqual(
            second_request["messages"][2:],
            [
                {"role": "assistant", "content": unusable},
                {
                    "role": "user",
                    "content": (
                        "Your last reply cannot be used: block 1 (figure) box must use the "
                        "0-to-1000 grid of the image, not its pixels\n"
                        "Return the complete JSON object again, in the shape that the first "
                        'message gives. In a JSON string, write a double quote as \\" and a '
                        "backslash as \\\\. Return only the JSON object."
                    ),
                },
            ],
        )
        defect = (
            "LayoutFormatError: block 1 (figure) box must use the 0-to-1000 grid of the image, "
            "not its pixels"
        )
        self.assertIn(
            f"WARNING:{LOGGER_NAME}:OpenAI-compatible model {DEFAULT_LAYOUT_MODEL} "
            f"returned an unusable reply in attempt 1 of 3 ({defect})",
            captured_logs.output,
        )
        self.assertIn(
            f"INFO:{LOGGER_NAME}:OpenAI-compatible model {DEFAULT_LAYOUT_MODEL} "
            "corrected its reply in attempt 2 of 3",
            captured_logs.output,
        )

    def test_fails_when_each_attempt_gives_an_unusable_layout(self) -> None:
        completions = FakeCompletions('{"blocks":[{"type":"table"}]}')
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(LayoutFormatError, r"block 1 \(table\) must have a rows array"),
        ):
            self.recognize_layout(completions)
        self.assertEqual(len(completions.calls), DEFAULT_MAX_ATTEMPTS)

        completions = FakeCompletions("not JSON")
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(RuntimeError, "returned invalid JSON"),
        ):
            self.recognize_layout(completions, options={"max_attempts": 1})
        self.assertEqual(len(completions.calls), 1)

    def test_log_records_have_no_text_from_the_reply(self) -> None:
        secret = "private text of the page"
        unusable = json.dumps(
            {"blocks": [{"type": secret, "text": secret}, {"type": secret, "rows": secret}]}
        )
        completions = FakeCompletions(unusable)
        with (
            self.assertLogs("scanwich", level="DEBUG") as captured_logs,
            self.assertRaises(LayoutFormatError) as raised,
        ):
            self.recognize_layout(completions)

        self.assertEqual(
            str(raised.exception),
            "block 2 (unknown type) has a type that the layout does not have; the types are: "
            "heading, paragraph, quote, list, table, figure, footnote, page_header, "
            "page_footer, page_number",
        )
        self.assertNotIn(secret, "\n".join(captured_logs.output))


class FigureLocationTests(TestCase):
    def locate_figure(
        self,
        completions: FakeCompletions,
        *,
        image_size: tuple[int, int] = (900, 600),
        estimate: Box | None = None,
        caption: str | None = None,
    ) -> FigureLocation | None:
        """Return the place that the backend makes from the replies for one part of a page."""
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "region.png"
            Image.new("RGB", image_size, "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                return backend.locate_figure(image_path, estimate=estimate, caption=caption)

    def test_backend_finds_the_box_of_an_illustration(self) -> None:
        backend = OpenAICompatibleBackend(languages=["en"], options={})
        self.assertIsInstance(backend, FigureLocator)

    def test_returns_the_box_and_the_cut_sides_of_the_reply(self) -> None:
        reply = '{"box":{"left":250,"top":150,"right":750,"bottom":640},"cut":["top"]}'
        completions = FakeCompletions(reply)

        location = self.locate_figure(completions)

        self.assertEqual(location, FigureLocation(Box(250, 150, 750, 640), frozenset({"top"})))
        (request,) = completions.calls
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(sent_image(request).size, (900, 600))
        prompt = request["messages"][1]["content"][0]["text"]
        for part in (
            "This 900x600 pixel image shows a part of one page of a printed document.",
            (
                'Return one JSON object with this shape: {"box":{"left":L,"top":T,"right":R,'
                '"bottom":B},"cut":[]}'
            ),
            "0-to-1000 grid",
            # A crop that is too large loses nothing. A crop that cuts the drawing does.
            "all its lines, and all the labels and numbers that are a part of it",
            "It does not contain the caption",
            "When you are not sure that a mark is a part of the illustration, include it.",
            '"cut" has the sides of the image where the illustration continues outside',
            'Return {"box":null,"cut":[]} when the image shows no illustration.',
        ):
            with self.subTest(part=part):
                self.assertIn(part, prompt)
        # Without data about the illustration, the request does not tell which one to find.
        self.assertNotIn("Which illustration.", prompt)

    def test_tells_the_model_which_illustration_to_find(self) -> None:
        # An image can show two illustrations. Without these data, a model returns one
        # box around the two of them.
        completions = FakeCompletions('{"box":[250,150,750,640],"cut":[]}')
        caption = 'Fig. 2. <i>A wheel</i> with "spokes"<br>and a hub<fn>1</fn>'

        self.locate_figure(completions, estimate=Box(213, 317, 617, 723), caption=caption)

        prompt = completions.calls[0]["messages"][1]["content"][0]["text"]
        self.assertIn(
            "Which illustration. The page gives this caption to the illustration: "
            '"Fig. 2. A wheel with \\"spokes\\" and a hub". A first estimate puts the center '
            "of the illustration near x=415, y=520 on the grid. The estimate is not exact. "
            "When the image shows more than one illustration, find only the one that these "
            "data describe.\n\n",
            prompt,
        )
        # The sides of the estimate are the values that the request must correct, so the
        # model does not get them.
        for side in ("213", "317", "617", "723"):
            self.assertNotIn(side, prompt)

        completions = FakeCompletions('{"box":[250,150,750,640],"cut":[]}')
        self.locate_figure(completions, caption="Plate " + "long caption " * 60)
        prompt = completions.calls[0]["messages"][1]["content"][0]["text"]
        quoted = prompt.split("caption to the illustration: ")[1].split(". When the image")[0]
        self.assertEqual(len(json.loads(quoted)), 300)
        self.assertNotIn("A first estimate", prompt)

    def test_returns_none_when_the_image_shows_no_illustration(self) -> None:
        self.assertIsNone(self.locate_figure(FakeCompletions('{"box":null,"cut":[]}')))

    def test_one_call_is_one_request(self) -> None:
        # The caller has a limit for its calls for one illustration. A correction request
        # for each call would multiply that limit by the attempts of the backend.
        replies = {
            # The box uses the pixels of the image and not the grid.
            '{"box":{"left":250,"top":150,"right":2400,"bottom":1700},"cut":[]}': (
                LayoutFormatError,
                "box must use the 0-to-1000 grid of the image, not its pixels",
            ),
            '{"cut":[]}': (LayoutFormatError, "the reply must be a JSON object with a box"),
            "not JSON": (RuntimeError, "returned invalid JSON"),
        }
        for reply, (error_type, message) in replies.items():
            completions = FakeCompletions(reply, '{"box":[250,150,750,640],"cut":[]}')
            with (
                self.subTest(reply=reply),
                self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
                self.assertRaisesRegex(error_type, message),
            ):
                self.locate_figure(completions)
            self.assertEqual(len(completions.calls), 1)
            self.assertIn("returned an unusable reply in attempt 1 of 1 (", captured_logs.output[0])


class RequestDefaultsTests(TestCase):
    """LaTeX output and the searchable PDF have their own default model and effort."""

    def requests(self, options: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Return the request of each request type, for a backend with these options."""
        replies = {
            "regions": '{"coordinate_system":"normalized","regions":[]}',
            "layout": '{"page_top":"top","blocks":[]}',
            "figure": '{"box":null,"cut":[]}',
        }
        result = {}
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (400, 300), "white").save(image_path)
            for name, reply in replies.items():
                completions = FakeCompletions(reply)
                fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
                fake_openai = SimpleNamespace(OpenAI=lambda _client=fake_client, **_: _client)
                backend = OpenAICompatibleBackend(
                    languages=["en"], options={"rotations": [], **options}
                )
                call = {
                    "regions": backend.recognize,
                    "layout": backend.recognize_layout,
                    "figure": backend.locate_figure,
                }[name]
                with (
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                    patch.dict(sys.modules, {"openai": fake_openai}),
                ):
                    call(image_path)
                (result[name],) = completions.calls
        return result

    def test_latex_requests_have_their_own_model_and_effort(self) -> None:
        requests = self.requests({})
        # The requests for text regions did not get a test with the model for layouts.
        self.assertEqual(DEFAULT_MODEL, "deepseek/deepseek-v4-flash-vision-exp")
        self.assertEqual(requests["regions"]["model"], DEFAULT_MODEL)
        self.assertEqual(requests["regions"]["extra_body"], {"reasoning_effort": "low"})
        for name in ("layout", "figure"):
            with self.subTest(request=name):
                self.assertEqual(requests[name]["model"], "~openai/gpt-luna-latest")
                self.assertEqual(requests[name]["model"], DEFAULT_LAYOUT_MODEL)
                self.assertEqual(requests[name]["extra_body"], {"reasoning_effort": "medium"})

    def test_option_of_the_caller_is_for_all_requests(self) -> None:
        requests = self.requests({"model": "provider/vision-model", "reasoning_effort": "high"})
        for name, request in requests.items():
            with self.subTest(request=name):
                self.assertEqual(request["model"], "provider/vision-model")
                self.assertEqual(request["extra_body"], {"reasoning_effort": "high"})
        # One option does not change the default of the other option.
        requests = self.requests({"model": "provider/vision-model"})
        self.assertEqual(requests["regions"]["extra_body"], {"reasoning_effort": "low"})
        self.assertEqual(requests["layout"]["extra_body"], {"reasoning_effort": "medium"})
        requests = self.requests({"reasoning_effort": "low"})
        self.assertEqual(requests["layout"]["model"], DEFAULT_LAYOUT_MODEL)
        self.assertEqual(requests["figure"]["extra_body"], {"reasoning_effort": "low"})

    def test_no_effort_and_an_alias_are_for_all_requests(self) -> None:
        # A value of null sends no effort. That is a value of the caller, not "no value".
        for name, request in self.requests({"reasoning_effort": None}).items():
            with self.subTest(request=name):
                self.assertNotIn("extra_body", request)
        # The built-in alias is not the default model for layouts: it names a model of
        # a different endpoint.
        for name, request in self.requests({"model": "luna"}).items():
            with self.subTest(request=name):
                self.assertEqual(request["model"], "gpt-5.6-luna")

    def test_log_records_name_the_model_of_the_request_and_of_the_response(self) -> None:
        usage = SimpleNamespace(prompt_tokens=1500, completion_tokens=800, cost=0.004)

        class Completions(FakeCompletions):
            def __init__(self, response_model: object) -> None:
                super().__init__('{"blocks":[]}')
                self.response_model = response_model

            def create(self, **request: Any) -> SimpleNamespace:
                completion = super().create(**request)
                completion.usage = usage
                if self.response_model is not None:
                    completion.model = self.response_model
                return completion

        cases = {
            # The model ID of the request is an alias that the provider moves. The
            # response names the model that made the reply.
            "openai/gpt-luna-2026-09-30": (
                "; the response names the model openai/gpt-luna-2026-09-30"
            ),
            None: "",
            # A log record has only the characters of a model ID from the response.
            "a model\nwith a line break": "",
            17: "",
        }
        for response_model, suffix in cases.items():
            completions = Completions(response_model)
            fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
            fake_openai = SimpleNamespace(OpenAI=lambda _client=fake_client, **_: _client)
            with TemporaryDirectory() as temporary_directory:
                image_path = Path(temporary_directory) / "page.png"
                Image.new("RGB", (400, 300), "white").save(image_path)
                with (
                    self.subTest(response_model=response_model),
                    patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                    patch.dict(sys.modules, {"openai": fake_openai}),
                    patch("uuid.uuid4", return_value=uuid.UUID(int=7)),
                    self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs,
                ):
                    backend = OpenAICompatibleBackend(languages=["en"], options={})
                    backend.recognize_layout(image_path)
            prefix = f"INFO:{LOGGER_NAME}:OpenAI-compatible session {uuid.UUID(int=7)}, model "
            records = [
                record
                for record in captured_logs.output
                if "usage." in record or "client" in record
            ]
            self.assertEqual(
                records,
                [
                    (
                        f"INFO:{LOGGER_NAME}:Initializing OpenAI-compatible client for model "
                        f"{DEFAULT_LAYOUT_MODEL}"
                    ),
                    (
                        f"{prefix}{DEFAULT_LAYOUT_MODEL}: usage.prompt_tokens=1500, "
                        f"usage.completion_tokens=800 for this request{suffix}"
                    ),
                    (
                        f"{prefix}{DEFAULT_LAYOUT_MODEL}: usage.cost=0.004000 for this request, "
                        "total 0.004000 for 1 request(s)"
                    ),
                ],
            )


class FakeAsyncCompletions(FakeCompletions):
    """Answer the requests of the async client. An error in the replies is raised."""

    def __init__(self, *contents: str | Exception) -> None:
        super().__init__(*(content for content in contents if isinstance(content, str)))
        self._replies = contents

    async def create(self, **request: Any) -> SimpleNamespace:  # type: ignore[override]
        reply = self._replies[min(len(self.calls), len(self._replies) - 1)]
        if isinstance(reply, Exception):
            self.calls.append(request)
            raise reply
        return FakeCompletions.create(self, **request)


class AsyncRequestTests(TestCase):
    """The async methods for LaTeX output, which a caller awaits for some pages at one time."""

    LAYOUT_REPLY = '{"page_top":"top","blocks":[{"type":"paragraph","text":"Text of the page."}]}'
    FIGURE_REPLY = '{"box":{"left":250,"top":150,"right":750,"bottom":640},"cut":["top"]}'

    def setUp(self) -> None:
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.image_path = Path(temporary_directory.name) / "page.png"
        Image.new("RGB", (400, 300), "white").save(self.image_path)

    def run_async(
        self, completions: FakeAsyncCompletions, method: str, **hints: Any
    ) -> tuple[Any, OpenAICompatibleBackend]:
        """Return the result of one async method, and the backend."""
        clients = []

        def make_client(**options: Any) -> SimpleNamespace:
            clients.append(options)
            return SimpleNamespace(chat=SimpleNamespace(completions=completions))

        fake_openai = SimpleNamespace(AsyncOpenAI=make_client)
        backend = OpenAICompatibleBackend(languages=["en"], options={})
        self.backend = backend
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            result = asyncio.run(getattr(backend, method)(self.image_path, **hints))
        self.assertEqual(len(clients), 1)
        return result, backend

    def run_sync(self, reply: str, method: str, **hints: Any) -> tuple[Any, dict[str, Any]]:
        completions = FakeCompletions(reply)
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        backend = OpenAICompatibleBackend(languages=["en"], options={})
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            result = getattr(backend, method)(self.image_path, **hints)
        (request,) = completions.calls
        return result, request

    def test_backend_has_the_contracts_for_parallel_pages(self) -> None:
        backend = OpenAICompatibleBackend(languages=["en"], options={})
        self.assertIsInstance(backend, AsyncLayoutBackend)
        self.assertIsInstance(backend, AsyncFigureLocator)
        self.assertIsInstance(backend, LayoutSettings)

    def test_async_request_is_the_request_of_the_synchronous_method(self) -> None:
        cases = {
            "recognize_layout": (self.LAYOUT_REPLY, {"upright": True}),
            "locate_figure": (
                self.FIGURE_REPLY,
                {"estimate": Box(100, 200, 800, 900), "caption": "Fig. 2"},
            ),
        }
        for method, (reply, hints) in cases.items():
            with self.subTest(method=method):
                expected, request = self.run_sync(reply, method, **hints)
                completions = FakeAsyncCompletions(reply)

                result, _ = self.run_async(completions, f"{method}_async", **hints)

                self.assertEqual(result, expected)
                self.assertEqual(completions.calls, [request])
                self.assertEqual(request["model"], DEFAULT_LAYOUT_MODEL)

    def test_async_layout_request_gets_a_correction_request(self) -> None:
        completions = FakeAsyncCompletions("not JSON", self.LAYOUT_REPLY)
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            layout, _ = self.run_async(completions, "recognize_layout_async")
        self.assertEqual(layout, PageLayout(blocks=(Paragraph("Text of the page."),)))
        self.assertEqual(len(completions.calls), 2)
        self.assertIn("returned an unusable reply in attempt 1 of 3", captured_logs.output[0])

    def test_one_async_call_for_a_box_is_one_request(self) -> None:
        completions = FakeAsyncCompletions("not JSON", self.FIGURE_REPLY)
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(RuntimeError, "returned invalid JSON"),
        ):
            self.run_async(completions, "locate_figure_async")
        self.assertEqual(len(completions.calls), 1)

    def service_error(self, status_code: int, headers: object) -> Exception:
        error = RuntimeError("the text of the provider")
        error.status_code = status_code  # type: ignore[attr-defined]
        error.response = SimpleNamespace(headers=headers)  # type: ignore[attr-defined]
        return error

    def test_service_with_too_many_requests_gives_its_own_error(self) -> None:
        # The caller makes all its requests wait, so it must know this cause.
        cases = [
            ({"retry-after": "7"}, 7.0),
            ({"retry-after": "1.5"}, 1.5),
            ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
            ({"retry-after": "-1"}, None),
            ({"retry-after": "nan"}, None),
            ({}, None),
            (None, None),
        ]
        for headers, retry_after in cases:
            for method in ("recognize_layout_async", "locate_figure_async"):
                error = self.service_error(429, headers)
                with (
                    self.subTest(headers=headers, method=method),
                    self.assertRaises(RateLimitError) as raised,
                ):
                    self.run_async(FakeAsyncCompletions(error, self.LAYOUT_REPLY), method)
                self.assertEqual(raised.exception.retry_after, retry_after)
                self.assertEqual(
                    str(raised.exception),
                    f"OpenAI-compatible request failed for model {DEFAULT_LAYOUT_MODEL}: the "
                    "service gets too many requests (HTTP status 429)",
                )
                self.assertIs(raised.exception.__cause__, error)

    def test_other_error_of_the_service_is_the_error_of_the_synchronous_method(self) -> None:
        completions = FakeAsyncCompletions(self.service_error(503, {"retry-after": "7"}))
        with self.assertRaises(RuntimeError) as raised:
            self.run_async(completions, "recognize_layout_async")
        self.assertNotIsInstance(raised.exception, RateLimitError)
        # The service gave no reply. The caller counts such pages.
        self.assertIsInstance(raised.exception, NoReplyError)
        self.assertEqual(
            str(raised.exception),
            f"OpenAI-compatible request failed for model {DEFAULT_LAYOUT_MODEL}: the text of "
            "the provider",
        )
        self.assertEqual(len(completions.calls), 1)

    def test_error_tells_if_the_service_gave_a_reply(self) -> None:
        # A page that got a reply does not count for the rule that stops a conversion
        # when the service gives no reply.
        no_choices = SimpleNamespace(choices=None, error={"message": "the provider is down"})
        too_long = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    native_finish_reason=None,
                    message=SimpleNamespace(content="{"),
                )
            ]
        )

        class Replies(FakeAsyncCompletions):
            def __init__(self, reply: object) -> None:
                super().__init__("")
                self.reply = reply

            async def create(self, **request: Any) -> Any:
                self.calls.append(request)
                if isinstance(self.reply, Exception):
                    raise self.reply
                return self.reply

        cases = {
            "no connection": (ConnectionError("no route"), True),
            "an HTTP error": (self.service_error(500, {}), True),
            "no choices in the completion": (no_choices, True),
            "a completion that stops at the token limit": (too_long, False),
        }
        for name, (reply, no_reply) in cases.items():
            with self.subTest(case=name), self.assertRaises(RuntimeError) as raised:
                self.run_async(Replies(reply), "recognize_layout_async")
            self.assertEqual(isinstance(raised.exception, NoReplyError), no_reply)
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaises(RuntimeError) as raised,
        ):
            self.run_async(FakeAsyncCompletions("not JSON"), "recognize_layout_async")
        self.assertNotIsInstance(raised.exception, NoReplyError)

    def test_status_of_an_answer_about_a_page_is_not_no_reply(self) -> None:
        # An answer about one page: the service works, and another page can get a reply.
        # The other statuses are the same for each page, or the service does not work.
        statuses = {
            **dict.fromkeys((400, 403, 413, 422), False),
            **dict.fromkeys((401, 402, 404, 408, 409, 500, 502, 503, 504, 529), True),
        }
        for status_code, no_reply in statuses.items():
            for first_reply in (None, "not JSON"):
                # The status at the first request, and at a correction request.
                replies = [self.service_error(status_code, {})]
                if first_reply is not None:
                    replies.insert(0, first_reply)
                with (
                    self.subTest(status=status_code, correction=first_reply is not None),
                    self.assertLogs(LOGGER_NAME, level="INFO"),
                    self.assertRaises(RuntimeError) as raised,
                ):
                    self.run_async(FakeAsyncCompletions(*replies), "recognize_layout_async")
                self.assertEqual(isinstance(raised.exception, NoReplyError), no_reply)
                if first_reply is not None:
                    # A correction request has text from a reply, so its error has no
                    # text of the provider.
                    self.assertIn(
                        f"in attempt 2 of 3: RuntimeError, HTTP status {status_code}",
                        str(raised.exception),
                    )

    def test_request_with_no_key_gives_no_reply(self) -> None:
        backend = OpenAICompatibleBackend(languages=["en"], options={})
        with (
            patch.dict(os.environ, clear=True),
            self.assertRaisesRegex(NoReplyError, "requires OPENROUTER_API_KEY to be set"),
        ):
            asyncio.run(backend.recognize_layout_async(self.image_path))

    def test_refused_request_goes_again_with_the_replies_before_it(self) -> None:
        # The replies before a refusal have a cost. They count for the limit of 3
        # requests, so the request after the refusal is the same correction request.
        refusal = self.service_error(429, {})
        completions = FakeAsyncCompletions("not JSON", "not JSON", refusal, "not JSON")
        backend = OpenAICompatibleBackend(languages=["en"], options={})
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        async def read() -> None:
            with self.assertRaises(RateLimitError) as raised:
                await backend.recognize_layout_async(self.image_path, upright=True)
            self.assertEqual(len(completions.calls), 3)
            assert raised.exception.resume is not None
            # The third request fails again, with a reply that has a cost.
            await raised.exception.resume()

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
            patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=lambda **_: client)}),
            self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
            self.assertRaisesRegex(RuntimeError, "returned invalid JSON"),
        ):
            asyncio.run(read())

        # 4 requests: 3 replies with a cost, and the refusal.
        self.assertEqual(len(completions.calls), 4)
        self.assertEqual(completions.calls[3], completions.calls[2])
        # A correction request has the reply before it and the defect.
        self.assertEqual([len(call["messages"]) for call in completions.calls], [2, 4, 6, 6])
        self.assertIn("returned an unusable reply in attempt 3 of 3", captured_logs.output[-1])

    def test_refused_request_with_a_good_reply_after_it(self) -> None:
        completions = FakeAsyncCompletions(
            "not JSON", self.service_error(429, {}), self.LAYOUT_REPLY
        )
        backend = OpenAICompatibleBackend(languages=["en"], options={})
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        async def read() -> PageLayout:
            try:
                return await backend.recognize_layout_async(self.image_path)
            except RateLimitError as error:
                assert error.resume is not None
                return await error.resume()

        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
            patch.dict(sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=lambda **_: client)}),
            self.assertLogs(LOGGER_NAME, level="WARNING"),
        ):
            layout = asyncio.run(read())
        self.assertEqual(layout, PageLayout(blocks=(Paragraph("Text of the page."),)))
        self.assertEqual(len(completions.calls), 3)

    def test_error_of_a_correction_request_has_no_text_of_the_provider(self) -> None:
        completions = FakeAsyncCompletions("not JSON", self.service_error(503, {}))
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaises(RuntimeError) as raised,
        ):
            self.run_async(completions, "recognize_layout_async")
        self.assertEqual(
            str(raised.exception),
            f"OpenAI-compatible request failed for model {DEFAULT_LAYOUT_MODEL} in attempt 2 "
            "of 3: RuntimeError, HTTP status 503",
        )

    def test_caller_can_close_the_async_client(self) -> None:
        closed = []

        class Completions(FakeAsyncCompletions):
            pass

        completions = Completions(self.LAYOUT_REPLY)

        async def close() -> None:
            closed.append(True)

        async def read_and_close() -> None:
            backend = OpenAICompatibleBackend(languages=["en"], options={})
            client = SimpleNamespace(chat=SimpleNamespace(completions=completions), close=close)
            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(
                    sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=lambda **_: client)}
                ),
            ):
                await backend.recognize_layout_async(self.image_path)
                await backend.aclose()
                # The next request makes a new client, for the event loop of that request.
                self.assertIsNone(backend._async_client)

        asyncio.run(read_and_close())
        self.assertEqual(closed, [True])

    def test_image_encode_does_not_stop_the_event_loop(self) -> None:
        # The encode of a page image uses the processor, so it is in a thread. A task
        # that needs the event loop at short intervals gets it during the encode.
        work_seconds = 0.3
        encode_image = openai_compatible._encode_image
        threads: list[str] = []

        def slow_encode(image_path: Path) -> object:
            threads.append(threading.current_thread().name)
            time.sleep(work_seconds)
            return encode_image(image_path)

        for method, reply in (
            ("recognize_layout_async", self.LAYOUT_REPLY),
            ("locate_figure_async", self.FIGURE_REPLY),
        ):
            gaps: list[float] = []
            backend = OpenAICompatibleBackend(languages=["en"], options={})
            completions = FakeAsyncCompletions(reply)
            client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

            async def beat(gaps: list[float] = gaps) -> None:
                last = time.monotonic()
                while True:
                    await asyncio.sleep(0.005)
                    gaps.append(time.monotonic() - last)
                    last = time.monotonic()

            async def read(backend: Any = backend, method: str = method) -> None:
                heartbeat = asyncio.create_task(beat())
                try:
                    await getattr(backend, method)(self.image_path)
                finally:
                    heartbeat.cancel()

            with (
                self.subTest(method=method),
                patch.object(openai_compatible, "_encode_image", slow_encode),
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(
                    sys.modules, {"openai": SimpleNamespace(AsyncOpenAI=lambda _c=client, **_: _c)}
                ),
            ):
                asyncio.run(read())
                self.assertEqual(len(completions.calls), 1)
                self.assertGreater(len(gaps), 20)
                self.assertLess(max(gaps), work_seconds / 2)
        self.assertEqual(len(threads), 2)
        self.assertNotIn(threading.main_thread().name, threads)

    def test_settings_that_change_a_layout(self) -> None:
        backend = OpenAICompatibleBackend(languages=["fr", "la"], options={})
        self.assertEqual(
            backend.layout_settings(),
            {
                "model": "~openai/gpt-luna-latest",
                "reasoning_effort": "medium",
                "languages": ["fr", "la"],
            },
        )
        backend = OpenAICompatibleBackend(
            languages=[],
            options={
                "model": "provider/vision-model",
                "reasoning_effort": None,
                "base_url": "https://user:password@example.test/v1",
                "api_key_env": "A_SECRET_NAME",
            },
        )
        settings = backend.layout_settings()
        self.assertEqual(
            settings, {"model": "provider/vision-model", "reasoning_effort": None, "languages": []}
        )
        # The settings go to a file, so they have no address and no name of a key.
        self.assertEqual(json.loads(json.dumps(settings)), settings)


def _pixel_page(text: str, left: float, top: float, right: float, bottom: float) -> str:
    return json.dumps(
        {"coordinate_system": "pixels", "regions": [box(text, left, top, right, bottom)]}
    )


class _TurningCompletions:
    """Return an upright word or a turned word, chosen from the image the request shows."""

    def __init__(self, turned_text: str, turned_box: tuple[float, float, float, float]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._turned_text = turned_text
        self._turned_box = turned_box

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls.append(request)
        image = sent_image(request)
        if image.size == (80, 30):
            content = _pixel_page("ALPHA", 10, 4, 50, 14)
        else:
            left, top, right, bottom = self._turned_box
            content = _pixel_page(self._turned_text, left, top, right, bottom)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))
            ]
        )

    async def create_async(self, **request: Any) -> SimpleNamespace:
        return self.create(**request)


class RotatedOpenAITextTests(TestCase):
    def recognize(
        self, completions: _TurningCompletions | FakeCompletions, **options: Any
    ) -> list[OcrRegion]:
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (80, 30), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options=options)
            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                return backend.recognize(image_path)

    def test_maps_a_word_from_a_quarter_turn_back_onto_the_page(self) -> None:
        for degrees, text, turned_box, polygon in (
            (
                90,
                "BETA",
                (4, 10, 16, 14),
                (Point(10, 26), Point(10, 14), Point(14, 14), Point(14, 26)),
            ),
            (
                270,
                "GAMMA",
                (4, 10, 16, 14),
                (Point(70, 4), Point(70, 16), Point(66, 16), Point(66, 4)),
            ),
        ):
            with self.subTest(degrees=degrees):
                completions = _TurningCompletions(text, turned_box)
                regions = self.recognize(completions, rotations=[degrees])
                by_text = {region.text: region for region in regions}
                self.assertEqual(set(by_text), {"ALPHA", text})
                self.assertEqual(
                    by_text["ALPHA"].polygon,
                    (Point(10, 4), Point(50, 4), Point(50, 14), Point(10, 14)),
                )
                self.assertEqual(by_text[text].polygon, polygon)
                self.assertEqual(sent_image(completions.calls[0]).size, (80, 30))
                self.assertEqual(sent_image(completions.calls[1]).size, (30, 80))
                self.assertIn("30x80", completions.calls[1]["messages"][1]["content"][0]["text"])

    def test_drops_a_sideways_reread_of_a_horizontal_word(self) -> None:
        # A tall box in the turned image is horizontal page text stood on end. Mapping
        # it makes a wide flat copy that may not overlap the upright word.
        completions = _TurningCompletions("MOTORS", (10, 20, 14, 60))
        regions = self.recognize(
            completions,
            rotations=[90],
        )
        # The upright fixture still returns ALPHA. MOTORS is the tall turned box.
        self.assertEqual([region.text for region in regions], ["ALPHA"])

    def test_keeps_an_upright_word_when_a_turned_pass_reads_the_same_spot(self) -> None:
        # The turned box maps onto ALPHA. Neither reading has a confidence.
        completions = _TurningCompletions("NOPE", (16, 10, 26, 50))
        regions = self.recognize(completions, rotations=[90])

        self.assertEqual([region.text for region in regions], ["ALPHA"])

    def test_default_rotations_send_three_requests(self) -> None:
        completions = FakeCompletions(
            json.dumps({"coordinate_system": "normalized", "regions": []})
        )
        self.recognize(completions)
        self.assertEqual(
            [sent_image(request).size for request in completions.calls],
            [(80, 30), (30, 80), (30, 80)],
        )

    def test_empty_rotations_send_one_request(self) -> None:
        completions = FakeCompletions(
            json.dumps({"coordinate_system": "normalized", "regions": []})
        )
        self.recognize(completions, rotations=[])
        self.assertEqual(len(completions.calls), 1)

    def test_async_recognition_maps_a_turned_word(self) -> None:
        completions = _TurningCompletions("BETA", (4, 10, 16, 14))

        async def create(**request: Any) -> SimpleNamespace:
            return completions.create(**request)

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        fake_openai = SimpleNamespace(AsyncOpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (80, 30), "white").save(image_path)
            backend = OpenAICompatibleBackend(languages=["en"], options={"rotations": [90]})
            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                regions = asyncio.run(backend.recognize_async(image_path))

        by_text = {region.text: region for region in regions}
        self.assertEqual(
            by_text["BETA"].polygon,
            (Point(10, 26), Point(10, 14), Point(14, 14), Point(14, 26)),
        )
        self.assertIn("ALPHA", by_text)

    def test_rejects_a_rotation_that_is_not_a_quarter_turn(self) -> None:
        with self.assertRaisesRegex(ValueError, "90, 180, or 270"):
            OpenAICompatibleBackend(languages=["en"], options={"rotations": [45]})
        with self.assertRaisesRegex(TypeError, "list of 90, 180, and 270"):
            OpenAICompatibleBackend(languages=["en"], options={"rotations": "90"})
