import base64
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from scanwich.backends.openai_compatible import (
    DEFAULT_MODEL,
    PROVIDER_DETAIL_MAX_CHARACTERS,
    OpenAICompatibleBackend,
)
from scanwich.models import OcrRegion, Point

LOGGER_NAME = "scanwich.backends.openai_compatible"


class FakeCompletions:
    def __init__(self, content: str, *, finish_reason: str | None = "stop") -> None:
        self.calls: list[dict[str, Any]] = []
        self._content = content
        self._finish_reason = finish_reason

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls.append(request)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self._finish_reason,
                    message=SimpleNamespace(content=self._content),
                )
            ]
        )


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
            expected_image_bytes = image_path.read_bytes()
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
        image_url = user_content[1]["image_url"]["url"]
        prefix, encoded_image = image_url.split(",", maxsplit=1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(base64.b64decode(encoded_image), expected_image_bytes)
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

    def test_reports_provider_error_for_a_reply_without_choices(self) -> None:
        no_choices = "OpenAI-compatible service returned no completion choices"
        provider_error = {
            "code": 502,
            "message": "Provider returned error",
            "metadata": {"raw": "upstream reply"},
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
            f"{stopped}: provider error code=502, message='Upstream timeout'",
        )
        choice.error = {"code": 502, "message": "m" * 2 * PROVIDER_DETAIL_MAX_CHARACTERS}
        message = str(self.recognition_error(SimpleNamespace(choices=[choice])))
        self.assertTrue(message.startswith(f"{stopped}: provider error code=502"))
        self.assertTrue(message.endswith("m..."))
        self.assertEqual(len(message) - len(stopped), PROVIDER_DETAIL_MAX_CHARACTERS)

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

    def test_rejects_unknown_and_invalid_options(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported.*unknown"):
            OpenAICompatibleBackend(languages=["en"], options={"unknown": True})
        with self.assertRaisesRegex(TypeError, "max_tokens"):
            OpenAICompatibleBackend(languages=["en"], options={"max_tokens": 10.5})
        with self.assertRaisesRegex(ValueError, "timeout"):
            OpenAICompatibleBackend(languages=["en"], options={"timeout": 0})
