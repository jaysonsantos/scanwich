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

from scanwich.backends.openrouter import DEFAULT_MODEL, OpenRouterBackend
from scanwich.models import Point


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.calls: list[dict[str, Any]] = []
        self._content = content

    def create(self, **request: Any) -> SimpleNamespace:
        self.calls.append(request)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class OpenRouterBackendTests(TestCase):
    def test_sends_image_and_converts_normalized_polygons_to_pixels(self) -> None:
        payload = {
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
            ]
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
            backend = OpenRouterBackend(languages=["pt", "en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
                self.assertLogs("scanwich.backends.openrouter", level="INFO") as captured_logs,
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
                        "X-OpenRouter-Title": "Scanwich",
                    },
                }
            ],
        )
        request = completions.calls[0]
        self.assertEqual(request["model"], DEFAULT_MODEL)
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(request["max_tokens"], 65_536)
        self.assertNotIn("extra_body", request)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        user_content = request["messages"][1]["content"]
        self.assertIn("Expected language codes: pt, en", user_content[0]["text"])
        self.assertIn('{"regions":[', user_content[0]["text"])
        image_url = user_content[1]["image_url"]["url"]
        prefix, encoded_image = image_url.split(",", maxsplit=1)
        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(base64.b64decode(encoded_image), expected_image_bytes)
        progress = "\n".join(captured_logs.output)
        self.assertIn("Initializing OpenRouter client", progress)
        self.assertIn("OpenRouter response received after", progress)

    def test_requires_api_key_environment_variable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            backend = OpenRouterBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"),
            ):
                backend.recognize(image_path)

    def test_rejects_out_of_bounds_model_coordinates(self) -> None:
        payload = {
            "regions": [
                {
                    "text": "text",
                    "polygon": [
                        {"x": 0, "y": 0},
                        {"x": 1051, "y": 0},
                        {"x": 1000, "y": 1000},
                        {"x": 0, "y": 1000},
                    ],
                }
            ]
        }
        completions = FakeCompletions(json.dumps(payload))
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (10, 10), "white").save(image_path)
            backend = OpenRouterBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
                self.assertRaisesRegex(RuntimeError, "exceed both normalized and image bounds"),
            ):
                backend.recognize(image_path)

    def test_accepts_native_pixel_coordinates_for_a_page(self) -> None:
        payload = {
            "regions": [
                {
                    "text": "pixel text",
                    "polygon": [
                        {"x": 100, "y": 1200},
                        {"x": 1500, "y": 1200},
                        {"x": 1500, "y": 1300},
                        {"x": 100, "y": 1300},
                    ],
                }
            ]
        }
        completions = FakeCompletions(json.dumps(payload))
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        fake_openai = SimpleNamespace(OpenAI=lambda **_: fake_client)
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            Image.new("RGB", (2000, 2000), "white").save(image_path)
            backend = OpenRouterBackend(languages=["en"], options={})

            with (
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                regions = backend.recognize(image_path)

        self.assertEqual(
            regions[0].polygon,
            (Point(100, 1200), Point(1500, 1200), Point(1500, 1300), Point(100, 1300)),
        )

    def test_rejects_unknown_and_invalid_options(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported.*unknown"):
            OpenRouterBackend(languages=["en"], options={"unknown": True})
        with self.assertRaisesRegex(TypeError, "max_tokens"):
            OpenRouterBackend(languages=["en"], options={"max_tokens": 10.5})
        with self.assertRaisesRegex(ValueError, "timeout"):
            OpenRouterBackend(languages=["en"], options={"timeout": 0})
