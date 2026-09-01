from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scanwich.ocr import OcrBackend
from scanwich.pdf import assemble_searchable_pdf, rasterize_pdf

logger = logging.getLogger(__name__)


def convert_pdf(
    input_pdf: Path,
    output_pdf: Path,
    *,
    backend: OcrBackend,
    dpi: int | None = None,
    ocr_output_directory: Path | None = None,
) -> None:
    input_pdf = input_pdf.expanduser().resolve()
    output_pdf = output_pdf.expanduser().resolve()
    if not input_pdf.is_file():
        raise FileNotFoundError(f"input PDF does not exist: {input_pdf}")
    if input_pdf == output_pdf:
        raise ValueError("input and output PDF paths must be different")
    if dpi is not None and dpi <= 0:
        raise ValueError("DPI must be greater than zero")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="scanwich-") as temporary_directory:
        work_directory = Path(temporary_directory)
        logger.info("Rasterizing %s", input_pdf)
        pages = rasterize_pdf(
            input_pdf,
            work_directory / "pages",
            dpi=dpi,
        )
        logger.info("Rasterized %d page(s)", len(pages))
        page_regions = []
        for page_number, page in enumerate(pages, start=1):
            logger.info(
                "Recognizing page %d/%d with %s",
                page_number,
                len(pages),
                type(backend).__name__,
            )
            regions = list(backend.recognize(page.image_path))
            page_regions.append(regions)
            logger.info("Recognized %d text region(s) on page %d", len(regions), page_number)
            if ocr_output_directory is not None:
                _write_ocr_json(ocr_output_directory, page_number, regions)
                logger.info("Wrote OCR JSON for page %d", page_number)

        temporary_pdf = work_directory / "result.pdf"
        logger.info("Assembling searchable PDF")
        assemble_searchable_pdf(pages, page_regions, temporary_pdf)
        os.replace(temporary_pdf, output_pdf)
        logger.info("Wrote searchable PDF to %s", output_pdf)


def _write_ocr_json(output_directory: Path, page_number: int, regions: Sequence[Any]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"page-{page_number:06d}.json"
    payload = [region.to_json() for region in regions]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_backend_options(values: Sequence[str]) -> Mapping[str, Any]:
    options: dict[str, Any] = {}
    for value in values:
        key, separator, raw_value = value.partition("=")
        if not separator or not key:
            raise ValueError(f"backend option must use KEY=VALUE syntax: {value!r}")
        try:
            parsed_value = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed_value = raw_value
        options[key] = parsed_value
    return options
