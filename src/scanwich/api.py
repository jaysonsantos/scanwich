"""HTTP access to configured OCR backends."""

from __future__ import annotations

import argparse
import base64
import logging
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from scanwich.models import OcrRegion
from scanwich.ocr import available_backends, backend_request_options, load_backend
from scanwich.pdf import DEFAULT_DPI, RasterizedPage, assemble_searchable_pdf

logger = logging.getLogger(__name__)

# Multipart boundaries, headers, and form fields travel with the image.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
TOO_LARGE_DETAIL = "Request exceeds the configured size limit"


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    options: dict[str, Any] = Field(default_factory=dict)


class ApiJsonSettingsSource(JsonConfigSettingsSource):
    def __call__(self) -> dict[str, Any]:
        path = self.current_state.get("config")
        if path is None:
            return {}
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return JsonConfigSettingsSource(self.settings_cls, json_file=path)()


class ApiConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid", env_prefix="SCANWICH_API_", env_nested_delimiter="__"
    )
    config: Path | None = Field(default=None, exclude=True)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    backends: dict[str, BackendConfig] = Field(
        default_factory=lambda: {"openai-compatible": BackendConfig()}
    )
    max_image_bytes: int = Field(default=20 * 1024 * 1024, gt=0)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            ApiJsonSettingsSource(settings_cls),
            file_secret_settings,
        )


class BackendsResponse(BaseModel):
    backends: list[str]


class OutputFormat(StrEnum):
    PDF = "pdf"
    TEXT = "text"
    PDF_TEXT = "pdf+text"


class PdfResponse(BaseModel):
    output: Literal["pdf"] = "pdf"
    pdf_base64: str


class TextResponse(BaseModel):
    output: Literal["text"] = "text"
    text: str


class PdfTextResponse(BaseModel):
    output: Literal["pdf+text"] = "pdf+text"
    pdf_base64: str
    text: str


OcrResponse = Annotated[PdfResponse | TextResponse | PdfTextResponse, Field(discriminator="output")]


def _make_pdf(path: Path, regions: Sequence[OcrRegion]) -> str:
    with Image.open(path) as image:
        width, height = image.size
    page = RasterizedPage(
        image_path=path,
        width_points=width * 72 / DEFAULT_DPI,
        height_points=height * 72 / DEFAULT_DPI,
        dpi=DEFAULT_DPI,
    )
    output_path = path.parent / "result.pdf"
    assemble_searchable_pdf([page], [regions], output_path)
    return base64.b64encode(output_path.read_bytes()).decode("ascii")


def _check_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


class MaxBodySizeMiddleware:
    """Reject oversized request bodies before the multipart parser reads them."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            response = JSONResponse({"detail": TOO_LARGE_DETAIL}, status_code=413)
            await response(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise HTTPException(413, TOO_LARGE_DETAIL)
            return message

        await self.app(scope, limited_receive, send)


def create_app(config: ApiConfig | None = None) -> FastAPI:
    if config is None:
        config = ApiConfig()
    app = FastAPI(title="Scanwich OCR API")
    app.add_middleware(
        MaxBodySizeMiddleware,
        max_body_bytes=config.max_image_bytes + MULTIPART_OVERHEAD_BYTES,
    )

    @app.get("/backends")
    async def backends() -> BackendsResponse:
        installed = available_backends()
        return BackendsResponse(backends=sorted(set(config.backends).intersection(installed)))

    @app.post("/ocr/{backend_name}", response_model_exclude_none=True)
    async def recognize(
        backend_name: str,
        image: UploadFile,
        output: Annotated[OutputFormat, Form()] = OutputFormat.PDF,
        model: Annotated[str | None, Form()] = None,
        languages: Annotated[list[str] | None, Form()] = None,
    ) -> OcrResponse:
        settings = config.backends.get(backend_name)
        if settings is None or backend_name not in available_backends():
            raise HTTPException(404, "Unknown or disabled OCR backend")
        options = dict(settings.options)
        if model is not None:
            supported = await run_in_threadpool(backend_request_options, backend_name)
            if "model" not in supported:
                raise HTTPException(422, "This OCR backend does not accept a model option")
            options["model"] = model
        with TemporaryDirectory(prefix="scanwich-api-") as directory:
            path = Path(directory) / "page"
            size = 0
            try:
                with path.open("wb") as image_file:
                    while chunk := await image.read(1024 * 1024):
                        size += len(chunk)
                        if size > config.max_image_bytes:
                            raise HTTPException(413, "Image exceeds the configured size limit")
                        await run_in_threadpool(image_file.write, chunk)
                await run_in_threadpool(_check_image, path)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                raise HTTPException(422, "Invalid image") from error
            finally:
                await image.close()
            backend = None
            try:
                backend = await run_in_threadpool(
                    load_backend, backend_name, languages=languages or ["en"], options=options
                )
                async_recognize = getattr(backend, "recognize_async", None)
                text_recognize = getattr(backend, "recognize_text_async", None)
                regions = None
                if output != OutputFormat.TEXT or text_recognize is None:
                    regions = list(
                        await async_recognize(path)
                        if async_recognize is not None
                        else await run_in_threadpool(backend.recognize, path)
                    )
                text = ""
                if output != OutputFormat.PDF:
                    text = (
                        await text_recognize(path)
                        if text_recognize is not None
                        else "\n".join(region.text for region in regions)
                    )
                if output == OutputFormat.TEXT:
                    return TextResponse(text=text)
                pdf = await run_in_threadpool(_make_pdf, path, regions)
                if output == OutputFormat.PDF:
                    return PdfResponse(pdf_base64=pdf)
                return PdfTextResponse(pdf_base64=pdf, text=text)
            except Exception as error:
                logger.warning("OCR backend %s failed: %s", backend_name, type(error).__name__)
                raise HTTPException(502, "OCR backend failed") from error
            finally:
                close = getattr(backend, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        logger.warning("Closing OCR backend %s failed", backend_name, exc_info=True)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve configured OCR backends")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = ApiConfig(**{key: value for key, value in vars(args).items() if value is not None})
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
