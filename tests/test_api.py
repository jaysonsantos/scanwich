import asyncio
import base64
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
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from scanwich.api import (
    MULTIPART_OVERHEAD_BYTES,
    PAGE_SEPARATOR,
    TOO_LARGE_DETAIL,
    ApiConfig,
    BackendConfig,
    create_app,
)
from scanwich.backends.openai_compatible import OpenAICompatibleBackend
from scanwich.models import OcrRegion, Point


def image_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (200, 100)).save(output, format="PNG")
    return output.getvalue()


def pdf_bytes(page_count: int) -> bytes:
    output = io.BytesIO()
    document = canvas.Canvas(output, pagesize=(144, 72))
    for number in range(page_count):
        document.drawString(10, 30, f"source page {number}")
        document.showPage()
    document.save()
    return output.getvalue()


def stem_regions(image_path: Path) -> list[OcrRegion]:
    """Recognize the page image file name so tests can map pages to results."""
    return [
        OcrRegion(
            text=image_path.stem,
            polygon=(Point(10, 10), Point(100, 10), Point(100, 30), Point(10, 30)),
        )
    ]


class TestApi(TestCase):
    def test_settings_priority_and_nested_environment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "max_upload_bytes": 100,
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
                    "SCANWICH_API_MAX_UPLOAD_BYTES": "200",
                    "SCANWICH_API_BACKENDS__OPENAI-COMPATIBLE__OPTIONS__MODEL": "glm-ocr",
                },
                clear=True,
            ):
                settings = ApiConfig()
                self.assertEqual(settings.max_upload_bytes, 200)
                self.assertEqual(settings.port, 9000)
                options = settings.backends["openai-compatible"].options
                self.assertEqual(options["model"], "glm-ocr")
                self.assertEqual(options["model_aliases"], {"deepseek": "~deepseek/model"})
                self.assertEqual(ApiConfig(max_upload_bytes=300).max_upload_bytes, 300)

    def test_legacy_upload_limit_name_still_applies(self):
        self.assertEqual(ApiConfig(max_image_bytes=300).max_upload_bytes, 300)
        self.assertEqual(ApiConfig(max_image_bytes=300, max_upload_bytes=400).max_upload_bytes, 400)
        with patch.dict(os.environ, {"SCANWICH_API_MAX_IMAGE_BYTES": "500"}, clear=True):
            self.assertEqual(ApiConfig().max_upload_bytes, 500)

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

    def test_response_schema_defines_output_choices(self):
        with TestClient(create_app()) as api:
            schema = api.get("/openapi.json").json()
        self.assertEqual(
            schema["components"]["schemas"]["OutputFormat"]["enum"], ["pdf", "text", "pdf+text"]
        )
        response = schema["paths"]["/ocr/{backend_name}"]["post"]["responses"]["200"]
        self.assertEqual(
            response["content"]["application/json"]["schema"]["discriminator"]["propertyName"],
            "output",
        )

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
                        "/ocr/openai-compatible", files={"file": ("page.png", image_bytes())}
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
        pdf = PdfReader(io.BytesIO(base64.b64decode(response.json()["pdf_base64"])))
        self.assertIn("hello", pdf.pages[0].extract_text())
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

    def test_model_option_follows_backend_capability(self):
        regions = [OcrRegion(text="hello", polygon=(Point(0, 0),) * 4)]
        config = ApiConfig(backends={"plugin": BackendConfig()})
        for supported, status in ((frozenset(), 422), (frozenset({"model"}), 200)):
            with self.subTest(supported=sorted(supported)):
                load = Mock(return_value=SimpleNamespace(recognize=Mock(return_value=regions)))
                with (
                    patch("scanwich.api.available_backends", return_value=["plugin"]),
                    patch("scanwich.api.backend_request_options", return_value=supported),
                    patch("scanwich.api.load_backend", load),
                    TestClient(create_app(config)) as api,
                ):
                    response = api.post(
                        "/ocr/plugin",
                        data={"model": "provider/plugin-model"},
                        files={"image": ("page.png", image_bytes())},
                    )
                self.assertEqual(response.status_code, status, response.text)
                if status == 422:
                    load.assert_not_called()
                else:
                    self.assertEqual(
                        load.call_args.kwargs["options"], {"model": "provider/plugin-model"}
                    )

    def test_multi_page_pdf_keeps_every_page(self):
        config = ApiConfig(backends={"plugin": BackendConfig()})
        with (
            patch("scanwich.api.available_backends", return_value=["plugin"]),
            patch(
                "scanwich.api.load_backend", return_value=SimpleNamespace(recognize=stem_regions)
            ),
            TestClient(create_app(config)) as api,
        ):
            response = api.post(
                "/ocr/plugin",
                data={"output": "pdf+text"},
                files={"file": ("scan.pdf", pdf_bytes(3))},
            )
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        reader = PdfReader(io.BytesIO(base64.b64decode(data["pdf_base64"])))
        self.assertEqual(len(reader.pages), 3)
        for number, page in enumerate(reader.pages):
            self.assertIn(f"page-{number:06d}", page.extract_text())
            self.assertAlmostEqual(float(page.mediabox.width), 144.0)
        self.assertEqual(
            data["text"].split(PAGE_SEPARATOR), [f"page-{number:06d}" for number in range(3)]
        )

    def test_async_backend_recognizes_every_pdf_page(self):
        backend = SimpleNamespace(
            recognize_async=AsyncMock(
                return_value=[OcrRegion(text="hello", polygon=(Point(0, 0),) * 4)]
            ),
            recognize_text_async=AsyncMock(return_value="hello"),
            aclose=AsyncMock(),
        )
        with (
            patch("scanwich.api.load_backend", return_value=backend),
            TestClient(create_app()) as api,
        ):
            response = api.post(
                "/ocr/openai-compatible",
                data={"output": "text"},
                files={"file": ("scan.pdf", pdf_bytes(4))},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["text"], PAGE_SEPARATOR.join(["hello"] * 4))
        self.assertEqual(backend.recognize_text_async.await_count, 4)
        backend.recognize_async.assert_not_awaited()
        backend.aclose.assert_awaited_once()

    def test_page_limit_and_invalid_pdf(self):
        load = Mock(return_value=SimpleNamespace(recognize=stem_regions))
        with (
            patch("scanwich.api.load_backend", load),
            TestClient(create_app(ApiConfig(max_pages=2))) as api,
        ):
            limited = api.post("/ocr/openai-compatible", files={"file": ("scan.pdf", pdf_bytes(3))})
            broken = api.post("/ocr/openai-compatible", files={"file": ("scan.pdf", b"%PDF-1.7\n")})
            accepted = api.post(
                "/ocr/openai-compatible", files={"file": ("scan.pdf", pdf_bytes(2))}
            )
        self.assertEqual(limited.status_code, 413, limited.text)
        self.assertIn("3 pages", limited.json()["detail"])
        self.assertEqual(broken.status_code, 422, broken.text)
        self.assertEqual(broken.json()["detail"], "Invalid PDF")
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(
            len(PdfReader(io.BytesIO(base64.b64decode(accepted.json()["pdf_base64"]))).pages), 2
        )
        load.assert_called_once()

    def test_dpi_field_stays_within_the_configured_ceiling(self):
        load = Mock(return_value=SimpleNamespace(recognize=stem_regions))
        with (
            patch("scanwich.api.load_backend", load),
            TestClient(create_app(ApiConfig(max_dpi=150))) as api,
        ):
            rejected = api.post(
                "/ocr/openai-compatible",
                data={"dpi": "10000"},
                files={"file": ("scan.pdf", pdf_bytes(1))},
            )
            accepted = api.post(
                "/ocr/openai-compatible",
                data={"dpi": "150"},
                files={"file": ("scan.pdf", pdf_bytes(1))},
            )
        self.assertEqual(rejected.status_code, 422, rejected.text)
        self.assertEqual(rejected.json()["detail"], "The dpi field must not exceed 150")
        load.assert_called_once()
        self.assertEqual(accepted.status_code, 200, accepted.text)

    def test_upload_field_choice_and_dpi(self):
        config = ApiConfig(backends={"plugin": BackendConfig()})
        with (
            patch("scanwich.api.available_backends", return_value=["plugin"]),
            patch(
                "scanwich.api.load_backend", return_value=SimpleNamespace(recognize=stem_regions)
            ),
            TestClient(create_app(config)) as api,
        ):
            missing = api.post("/ocr/plugin", data={"output": "text"})
            both = api.post(
                "/ocr/plugin",
                files=[
                    ("file", ("page.png", image_bytes())),
                    ("image", ("page.png", image_bytes())),
                ],
            )
            scaled = api.post(
                "/ocr/plugin", data={"dpi": "72"}, files={"image": ("page.png", image_bytes())}
            )
        for response in (missing, both):
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(
                response.json()["detail"], "Provide one PDF or image in the file field"
            )
        self.assertEqual(scaled.status_code, 200, scaled.text)
        page = PdfReader(io.BytesIO(base64.b64decode(scaled.json()["pdf_base64"]))).pages[0]
        self.assertAlmostEqual(float(page.mediabox.width), 200.0)
        self.assertAlmostEqual(float(page.mediabox.height), 100.0)

    def test_oversized_body_is_rejected_before_parsing(self):
        payload = b"x" * (2 * MULTIPART_OVERHEAD_BYTES)
        load = Mock()
        with (
            patch("scanwich.api.load_backend", load),
            TestClient(create_app(ApiConfig(max_image_bytes=1))) as api,
        ):
            declared = api.post("/ocr/openai-compatible", files={"image": ("page.png", payload)})
            streamed = api.post(
                "/ocr/openai-compatible",
                headers={"content-type": "multipart/form-data; boundary=test"},
                content=iter([b"--test\r\n", payload, b"\r\n--test--\r\n"]),
            )
        for response in (declared, streamed):
            self.assertEqual(response.status_code, 413, response.text)
            self.assertEqual(response.json(), {"detail": TOO_LARGE_DETAIL})
        load.assert_not_called()

    def test_cleanup_failure_keeps_the_request_outcome(self):
        backend = SimpleNamespace(
            recognize_async=AsyncMock(
                return_value=[OcrRegion(text="hello", polygon=(Point(0, 0),) * 4)]
            ),
            aclose=AsyncMock(side_effect=RuntimeError("close failed")),
        )
        with (
            patch("scanwich.api.load_backend", return_value=backend),
            TestClient(create_app()) as api,
            self.assertLogs("scanwich.api", level="WARNING") as logs,
        ):
            response = api.post(
                "/ocr/openai-compatible", files={"image": ("page.png", image_bytes())}
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["output"], "pdf")
        backend.aclose.assert_awaited_once()
        self.assertIn("Closing OCR backend openai-compatible failed", "\n".join(logs.output))

    def test_all_output_modes(self):
        regions = [
            OcrRegion(
                text="hello", polygon=(Point(10, 10), Point(100, 10), Point(100, 30), Point(10, 30))
            )
        ]
        for mode in ("pdf", "text", "pdf+text"):
            with self.subTest(mode=mode):
                backend = SimpleNamespace(
                    recognize_async=AsyncMock(return_value=regions),
                    recognize_text_async=AsyncMock(return_value="hello\n\n  world\n"),
                    aclose=AsyncMock(),
                )
                with (
                    patch("scanwich.api.load_backend", return_value=backend),
                    TestClient(create_app()) as api,
                ):
                    response = api.post(
                        "/ocr/openai-compatible",
                        data={"output": mode},
                        files={"image": ("page.png", image_bytes())},
                    )
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                self.assertEqual(data["output"], mode)
                if mode != "pdf":
                    self.assertEqual(data["text"], "hello\n\n  world\n")
                    backend.recognize_text_async.assert_awaited_once()
                else:
                    backend.recognize_text_async.assert_not_awaited()
                    self.assertNotIn("text", data)
                if mode != "text":
                    reader = PdfReader(io.BytesIO(base64.b64decode(data["pdf_base64"])))
                    self.assertIn("hello", reader.pages[0].extract_text())
                    backend.recognize_async.assert_awaited_once()
                else:
                    backend.recognize_async.assert_not_awaited()
                    self.assertNotIn("pdf_base64", data)
                backend.aclose.assert_awaited_once()

    def test_invalid_output_and_sync_text(self):
        backend = SimpleNamespace(
            recognize=Mock(return_value=[OcrRegion(text="hello", polygon=(Point(0, 0),) * 4)])
        )
        with (
            patch("scanwich.api.load_backend", return_value=backend),
            TestClient(create_app()) as api,
        ):
            response = api.post(
                "/ocr/openai-compatible",
                data={"output": "xml"},
                files={"image": ("page.png", image_bytes())},
            )
            self.assertEqual(response.status_code, 422)
            backend.recognize.assert_not_called()
            response = api.post(
                "/ocr/openai-compatible",
                data={"output": "text"},
                files={"image": ("page.png", image_bytes())},
            )
            self.assertEqual(response.json(), {"output": "text", "text": "hello"})

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
    async def test_plain_text_prompt_and_response(self):
        for content in ("hello\n\n  world\n", ""):
            create = AsyncMock(
                return_value=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop", message=SimpleNamespace(content=content)
                        )
                    ]
                )
            )
            backend = OpenAICompatibleBackend(languages=["pt"], options={})
            backend._async_client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            with TemporaryDirectory() as directory:
                path = Path(directory) / "page.png"
                path.write_bytes(image_bytes())
                self.assertEqual(await backend.recognize_text_async(path), content)
            request = create.call_args.kwargs
            self.assertNotIn("response_format", request)
            prompt = request["messages"][1]["content"][0]["text"]
            self.assertIn("plain text", prompt)
            self.assertIn("line breaks", prompt)
            self.assertIn("pt", prompt)

    def test_alias_validation_and_defaults(self):
        for alias, expected in (
            ("glm-ocr", "zai-org/GLM-OCR"),
            ("luna", "gpt-5.6-luna"),
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
