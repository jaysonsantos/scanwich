import asyncio
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from scanwich.api import ApiConfig, BackendConfig, create_app
from scanwich.backends.openai_compatible import OpenAICompatibleBackend
from scanwich.models import OcrRegion, Point


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (200, 100)).save(output, format="PNG")
    return output.getvalue()


class TestApi(TestCase):
    def test_settings_priority_and_nested_environment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "max_image_bytes": 100,
                        "port": 9000,
                        "backends": {
                            "openai-compatible": {
                                "options": {
                                    "model": "deepseek",
                                    "model_aliases": {"deepseek": "~deepseek/model"},
                                }
                            }
                        },
                    }
                )
            )
            with patch.dict(
                os.environ,
                {
                    "SCANWICH_API_CONFIG": str(path),
                    "SCANWICH_API_MAX_IMAGE_BYTES": "200",
                    "SCANWICH_API_BACKENDS__OPENAI-COMPATIBLE__OPTIONS__MODEL": "glm-ocr",
                },
                clear=True,
            ):
                settings = ApiConfig()
                self.assertEqual(settings.max_image_bytes, 200)
                self.assertEqual(settings.port, 9000)
                options = settings.backends["openai-compatible"].options
                self.assertEqual(options["model"], "glm-ocr")
                self.assertEqual(options["model_aliases"], {"deepseek": "~deepseek/model"})
                self.assertEqual(ApiConfig(max_image_bytes=300).max_image_bytes, 300)

    def test_invalid_settings_fail_at_startup(self):
        with (
            patch.dict(os.environ, {"SCANWICH_API_MAX_IMAGE_BYTES": "0"}, clear=True),
            self.assertRaises(ValidationError),
        ):
            create_app()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with self.assertRaises(FileNotFoundError):
                ApiConfig(config=path)
            path.write_text('{"unknown_option":true}')
            with self.assertRaises(ValidationError):
                ApiConfig(config=path)

    def test_response_schema_defines_regions(self):
        with TestClient(create_app()) as api:
            schema = api.get("/openapi.json").json()
        region = schema["components"]["schemas"]["RegionResponse"]
        self.assertEqual(region["properties"]["polygon"]["minItems"], 4)
        self.assertEqual(region["properties"]["polygon"]["maxItems"], 4)
        self.assertEqual(region["required"], ["text", "polygon"])

    def test_loads_server_configuration_from_environment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"backends":{},"max_image_bytes":100}')
            with (
                patch.dict(os.environ, {"SCANWICH_API_CONFIG": str(path)}),
                TestClient(create_app()) as api,
            ):
                self.assertEqual(api.get("/backends").json(), {"backends": []})
                self.assertEqual(
                    api.post(
                        "/ocr/openai-compatible", files={"image": ("page.png", image_bytes())}
                    ).status_code,
                    404,
                )

    def test_async_backend_alias_and_cleanup(self):
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "coordinate_system": "normalized",
                                "regions": [
                                    {
                                        "text": "hello",
                                        "polygon": [
                                            {"x": 0, "y": 0},
                                            {"x": 1000, "y": 0},
                                            {"x": 1000, "y": 1000},
                                            {"x": 0, "y": 1000},
                                        ],
                                    }
                                ],
                            }
                        )
                    ),
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=completion))
            ),
            close=AsyncMock(),
        )
        config = ApiConfig(
            backends={
                "openai-compatible": BackendConfig(
                    options={
                        "model_aliases": {"glm-ocr": "provider/real-glm-model"},
                    }
                )
            }
        )
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "secret"}),
            patch("openai.AsyncOpenAI", return_value=client),
            TestClient(create_app(config)) as api,
        ):
            response = api.post(
                "/ocr/openai-compatible",
                data={"model": "glm-ocr"},
                files={"image": ("page.png", image_bytes())},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["regions"][0]["polygon"][2], [200, 100])
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"], "provider/real-glm-model"
        )
        client.close.assert_awaited_once()

    def test_sync_plugin_and_temporary_file_cleanup(self):
        paths = []

        def recognize(path):
            self.assertTrue(path.exists())
            paths.append(path)
            return [OcrRegion(text="hello", polygon=(Point(0, 0),) * 4)]

        config = ApiConfig(backends={"plugin": BackendConfig()})
        with (
            patch("scanwich.api.available_backends", return_value=["plugin"]),
            patch("scanwich.api.load_backend", return_value=SimpleNamespace(recognize=recognize)),
            TestClient(create_app(config)) as api,
        ):
            self.assertEqual(api.get("/backends").json(), {"backends": ["plugin"]})
            response = api.post("/ocr/plugin", files={"image": ("../../page", image_bytes())})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(paths[0].exists())

    def test_invalid_uploads_and_backend_errors(self):
        backend = SimpleNamespace(
            recognize_async=AsyncMock(side_effect=RuntimeError("secret")), aclose=AsyncMock()
        )
        with (
            patch("scanwich.api.load_backend", return_value=backend),
            TestClient(create_app()) as api,
        ):
            self.assertEqual(
                api.post("/ocr/missing", files={"image": ("x", b"x")}).status_code, 404
            )
            self.assertEqual(
                api.post("/ocr/openai-compatible", files={"image": ("x", b"x")}).status_code, 422
            )
            response = api.post("/ocr/openai-compatible", files={"image": ("x", image_bytes())})
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("secret", response.text)
        backend.aclose.assert_awaited_once()
        with TestClient(create_app(ApiConfig(max_image_bytes=1))) as api:
            self.assertEqual(
                api.post("/ocr/openai-compatible", files={"image": ("x", b"xx")}).status_code, 413
            )


class TestAsyncBackend(IsolatedAsyncioTestCase):
    def test_alias_validation_and_defaults(self):
        for alias, expected in (
            ("glm-ocr", "zai-org/GLM-OCR"),
            ("provider/custom", "provider/custom"),
        ):
            backend = OpenAICompatibleBackend(languages=["en"], options={"model": alias})
            self.assertEqual(backend._model, expected)
        for aliases in ([], {"glm-ocr": ""}, {"glm-ocr": 42}):
            with self.assertRaisesRegex(TypeError, "model_aliases"):
                OpenAICompatibleBackend(languages=["en"], options={"model_aliases": aliases})

    async def test_requests_overlap_and_reuse_client(self):
        entered = 0
        both_entered = asyncio.Event()

        async def create(**request):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), 2)
            self.assertEqual(request["model"], "~deepseek/custom")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content='{"coordinate_system":"normalized","regions":[]}'
                        ),
                    )
                ]
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=AsyncMock()
        )
        constructor = Mock(return_value=client)
        backend = OpenAICompatibleBackend(
            languages=["en"],
            options={
                "model": "deepseek",
                "model_aliases": {"deepseek": "~deepseek/custom"},
            },
        )
        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "secret"}),
            patch("openai.AsyncOpenAI", constructor),
        ):
            constructor.assert_not_called()
            path = Path(directory) / "page.png"
            path.write_bytes(image_bytes())
            self.assertEqual(
                await asyncio.gather(backend.recognize_async(path), backend.recognize_async(path)),
                [[], []],
            )
            await backend.aclose()
        constructor.assert_called_once()
        client.close.assert_awaited_once()

    async def test_sync_and_async_requests_match(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="{}"))]
        )
        sync_create = Mock(return_value=response)
        async_create = AsyncMock(return_value=response)
        backend = OpenAICompatibleBackend(languages=["en"], options={"model": "provider/model"})
        backend._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=sync_create))
        )
        backend._async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "page.png"
            path.write_bytes(image_bytes())
            with self.assertRaisesRegex(RuntimeError, "finish_reason"):
                backend.recognize(path)
            with self.assertRaisesRegex(RuntimeError, "finish_reason"):
                await backend.recognize_async(path)
        self.assertEqual(sync_create.call_args.kwargs, async_create.call_args.kwargs)
