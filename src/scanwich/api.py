"""HTTP access to configured OCR backends."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from scanwich.ocr import available_backends, load_backend

logger = logging.getLogger(__name__)


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    options: dict[str, Any] = Field(default_factory=dict)


class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backends: dict[str, BackendConfig] = Field(
        default_factory=lambda: {"openai-compatible": BackendConfig()}
    )
    max_image_bytes: int = Field(default=20 * 1024 * 1024, gt=0)


def _check_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def create_app(config: ApiConfig | None = None) -> FastAPI:
    if config is None:
        config_path = os.environ.get("SCANWICH_API_CONFIG")
        config = (
            ApiConfig.model_validate_json(Path(config_path).read_text())
            if config_path
            else ApiConfig()
        )
    app = FastAPI(title="Scanwich OCR API")

    @app.get("/backends")
    async def backends() -> dict[str, Any]:
        installed = available_backends()
        return {"backends": sorted(set(config.backends).intersection(installed))}

    @app.post("/ocr/{backend_name}")
    async def recognize(
        backend_name: str,
        image: UploadFile,
        model: Annotated[str | None, Form()] = None,
        languages: Annotated[list[str] | None, Form()] = None,
    ) -> dict[str, Any]:
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
                return {"regions": [region.to_json() for region in regions]}
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = ApiConfig.model_validate(json.loads(args.config.read_text())) if args.config else None
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
