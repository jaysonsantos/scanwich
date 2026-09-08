"""HTTP access to configured OCR backends."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

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

from scanwich.ocr import available_backends, load_backend

logger = logging.getLogger(__name__)


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


class RegionResponse(BaseModel):
    text: str = Field(min_length=1)
    polygon: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    confidence: float | None = None


class OcrResponse(BaseModel):
    regions: list[RegionResponse]


def _check_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def create_app(config: ApiConfig | None = None) -> FastAPI:
    if config is None:
        config = ApiConfig()
    app = FastAPI(title="Scanwich OCR API")

    @app.get("/backends")
    async def backends() -> BackendsResponse:
        installed = available_backends()
        return BackendsResponse(backends=sorted(set(config.backends).intersection(installed)))

    @app.post("/ocr/{backend_name}", response_model_exclude_none=True)
    async def recognize(
        backend_name: str,
        image: UploadFile,
        model: Annotated[str | None, Form()] = None,
        languages: Annotated[list[str] | None, Form()] = None,
    ) -> OcrResponse:
        settings = config.backends.get(backend_name)
        if settings is None or backend_name not in available_backends():
            raise HTTPException(404, "Unknown or disabled OCR backend")
        options = dict(settings.options)
        if model is not None:
            if backend_name != "openai-compatible":
                raise HTTPException(422, "Model selection requires openai-compatible")
            options["model"] = model
        with TemporaryDirectory(prefix="scanwich-api-") as directory:
            path = Path(directory) / "page"
            size = 0
            try:
                with path.open("wb") as output:
                    while chunk := await image.read(1024 * 1024):
                        size += len(chunk)
                        if size > config.max_image_bytes:
                            raise HTTPException(413, "Image exceeds the configured size limit")
                        await run_in_threadpool(output.write, chunk)
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
                regions = (
                    await async_recognize(path)
                    if async_recognize is not None
                    else await run_in_threadpool(backend.recognize, path)
                )
                return OcrResponse(
                    regions=[RegionResponse.model_validate(region.to_json()) for region in regions]
                )
            except Exception as error:
                logger.warning("OCR backend %s failed: %s", backend_name, type(error).__name__)
                raise HTTPException(502, "OCR backend failed") from error
            finally:
                close = getattr(backend, "aclose", None)
                if close is not None:
                    await close()

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
