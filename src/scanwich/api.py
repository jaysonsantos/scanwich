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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
from scanwich.pdf import (
    DEFAULT_DPI,
    PdfPipelineError,
    RasterizedPage,
    assemble_searchable_pdf,
    count_pdf_pages,
    rasterize_pdf,
)

logger = logging.getLogger(__name__)

# Multipart boundaries, headers, and form fields travel with the upload.
MULTIPART_OVERHEAD_BYTES = 64 * 1024
TOO_LARGE_DETAIL = "Request exceeds the configured size limit"
PDF_MAGIC = b"%PDF-"
# Plain text output separates pages with a form feed, as PDF text extractors do.
PAGE_SEPARATOR = "\f"


class TooManyPagesError(Exception):
    """The upload has more pages than the configuration allows."""


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
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_pages: int | None = Field(default=None, gt=0)
    # A page rendered above this DPI can exhaust the worker's memory.
    max_dpi: int = Field(default=600, gt=0)
    # The name used before the API accepted PDF uploads.
    max_image_bytes: int | None = Field(default=None, gt=0, exclude=True)

    @model_validator(mode="after")
    def _apply_legacy_names(self) -> ApiConfig:
        if self.max_image_bytes is not None and "max_upload_bytes" not in self.model_fields_set:
            self.max_upload_bytes = self.max_image_bytes
        return self

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


def _make_pdf(
    pages: Sequence[RasterizedPage],
    page_regions: Sequence[Sequence[OcrRegion]],
    output_path: Path,
) -> str:
    assemble_searchable_pdf(pages, page_regions, output_path)
    return base64.b64encode(output_path.read_bytes()).decode("ascii")


def _check_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def _prepare_pages(
    upload_path: Path,
    pages_directory: Path,
    *,
    dpi: int | None,
    max_pages: int | None,
) -> list[RasterizedPage]:
    """Return one page per PDF page, or a single page for an image upload."""
    with upload_path.open("rb") as upload_file:
        header = upload_file.read(len(PDF_MAGIC))
    if header == PDF_MAGIC:
        if max_pages is not None:
            page_count = count_pdf_pages(upload_path)
            if page_count > max_pages:
                raise TooManyPagesError(f"PDF has {page_count} pages; the limit is {max_pages}")
        return rasterize_pdf(upload_path, pages_directory, dpi=dpi)

    _check_image(upload_path)
    with Image.open(upload_path) as image:
        width, height = image.size
    page_dpi = float(dpi) if dpi is not None else DEFAULT_DPI
    return [
        RasterizedPage(
            image_path=upload_path,
            width_points=width * 72 / page_dpi,
            height_points=height * 72 / page_dpi,
            dpi=page_dpi,
        )
    ]


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
        max_body_bytes=config.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )

    @app.get("/backends")
    async def backends() -> BackendsResponse:
        installed = available_backends()
        return BackendsResponse(backends=sorted(set(config.backends).intersection(installed)))

    @app.post("/ocr/{backend_name}", response_model_exclude_none=True)
    async def recognize(
        backend_name: str,
        file: Annotated[UploadFile | None, File()] = None,
        image: Annotated[UploadFile | None, File()] = None,
        output: Annotated[OutputFormat, Form()] = OutputFormat.PDF,
        model: Annotated[str | None, Form()] = None,
        languages: Annotated[list[str] | None, Form()] = None,
        dpi: Annotated[int | None, Form(gt=0)] = None,
    ) -> OcrResponse:
        if (file is None) == (image is None):
            raise HTTPException(422, "Provide one PDF or image in the file field")
        upload = file if file is not None else image
        if dpi is not None and dpi > config.max_dpi:
            raise HTTPException(422, f"The dpi field must not exceed {config.max_dpi}")
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
            work_directory = Path(directory)
            upload_path = work_directory / "upload"
            size = 0
            try:
                with upload_path.open("wb") as upload_file:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > config.max_upload_bytes:
                            raise HTTPException(413, "Upload exceeds the configured size limit")
                        await run_in_threadpool(upload_file.write, chunk)
            finally:
                await upload.close()
            try:
                pages = await run_in_threadpool(
                    _prepare_pages,
                    upload_path,
                    work_directory / "pages",
                    dpi=dpi,
                    max_pages=config.max_pages,
                )
            except TooManyPagesError as error:
                raise HTTPException(413, str(error)) from error
            except PdfPipelineError as error:
                raise HTTPException(422, "Invalid PDF") from error
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                raise HTTPException(422, "Invalid image") from error
            backend = None
            try:
                backend = await run_in_threadpool(
                    load_backend, backend_name, languages=languages or ["en"], options=options
                )
                async_recognize = getattr(backend, "recognize_async", None)
                text_recognize = getattr(backend, "recognize_text_async", None)
                page_regions: list[Sequence[OcrRegion]] = []
                texts: list[str] = []
                for page_number, page in enumerate(pages, start=1):
                    logger.info(
                        "Recognizing page %d/%d with %s", page_number, len(pages), backend_name
                    )
                    regions: list[OcrRegion] | None = None
                    if output != OutputFormat.TEXT or text_recognize is None:
                        regions = list(
                            await async_recognize(page.image_path)
                            if async_recognize is not None
                            else await run_in_threadpool(backend.recognize, page.image_path)
                        )
                        page_regions.append(regions)
                    if output != OutputFormat.PDF:
                        texts.append(
                            await text_recognize(page.image_path)
                            if text_recognize is not None
                            else "\n".join(region.text for region in regions)
                        )
                text = PAGE_SEPARATOR.join(texts)
                if output == OutputFormat.TEXT:
                    return TextResponse(text=text)
                pdf = await run_in_threadpool(
                    _make_pdf, pages, page_regions, work_directory / "result.pdf"
                )
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
