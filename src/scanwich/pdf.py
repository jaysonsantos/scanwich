from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from scanwich.models import OcrRegion, Point

POINTS_PER_INCH = 72.0
PDF_FONT = "Helvetica"
DEFAULT_DPI = 300.0
FULL_PAGE_IMAGE_COVERAGE = 0.9
PNG_COMPRESSION_LEVEL = 1

logger = logging.getLogger(__name__)


class PdfPipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class RasterizedPage:
    image_path: Path
    width_points: float
    height_points: float
    dpi: float


def rasterize_pdf(
    input_pdf: Path,
    output_directory: Path,
    *,
    dpi: int | None = None,
) -> list[RasterizedPage]:
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        document = pdfium.PdfDocument(input_pdf)
    except (OSError, pdfium.PdfiumError) as error:
        raise PdfPipelineError(f"PDFium could not open {input_pdf}: {error}") from error

    pages: list[RasterizedPage] = []
    try:
        page_count = len(document)
        for page_index in range(page_count):
            page = document[page_index]
            try:
                width_points, height_points = (float(value) for value in page.get_size())
                page_dpi, dpi_source = _select_page_dpi(
                    page,
                    width_points=width_points,
                    height_points=height_points,
                    requested_dpi=dpi,
                )
                logger.info(
                    "Rasterizing page %d/%d at %.1f DPI (%s)",
                    page_index + 1,
                    page_count,
                    page_dpi,
                    dpi_source,
                )
                bitmap = page.render(scale=page_dpi / POINTS_PER_INCH, rev_byteorder=True)
                try:
                    image_path = output_directory / f"page-{page_index:06d}.png"
                    bitmap.to_pil().save(
                        image_path,
                        format="PNG",
                        compress_level=PNG_COMPRESSION_LEVEL,
                    )
                finally:
                    bitmap.close()
                pages.append(
                    RasterizedPage(
                        image_path=image_path,
                        width_points=width_points,
                        height_points=height_points,
                        dpi=page_dpi,
                    )
                )
            finally:
                page.close()
    except (OSError, ValueError, pdfium.PdfiumError) as error:
        raise PdfPipelineError(f"PDFium could not rasterize {input_pdf}: {error}") from error
    finally:
        document.close()

    if not pages:
        raise PdfPipelineError(f"PDFium found no pages in {input_pdf}")
    return pages


def _select_page_dpi(
    page: pdfium.PdfPage,
    *,
    width_points: float,
    height_points: float,
    requested_dpi: int | None,
) -> tuple[float, str]:
    if requested_dpi is not None:
        return float(requested_dpi), "requested"

    inferred_dpi = _infer_full_page_image_dpi(
        page,
        width_points=width_points,
        height_points=height_points,
    )
    if inferred_dpi is None:
        return DEFAULT_DPI, "default; no full-page image DPI found"

    inferred_dpi = _snap_near_integer(inferred_dpi)
    if inferred_dpi > DEFAULT_DPI:
        return DEFAULT_DPI, f"inferred {inferred_dpi:.1f}, capped"
    return inferred_dpi, "inferred from full-page image"


def _infer_full_page_image_dpi(
    page: pdfium.PdfPage,
    *,
    width_points: float,
    height_points: float,
) -> float | None:
    rotation = page.get_rotation()
    if rotation in (90, 270):
        object_width_points = height_points
        object_height_points = width_points
    else:
        object_width_points = width_points
        object_height_points = height_points

    page_area = object_width_points * object_height_points
    if page_area <= 0:
        return None

    candidates: list[tuple[float, float]] = []
    for image in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)):
        try:
            left, bottom, right, top = image.get_bounds()
            metadata = image.get_metadata()
        except pdfium.PdfiumError:
            logger.debug("Could not inspect a PDF image while inferring DPI", exc_info=True)
            continue

        covered_width = max(0.0, min(right, object_width_points) - max(left, 0.0))
        covered_height = max(0.0, min(top, object_height_points) - max(bottom, 0.0))
        coverage = covered_width * covered_height / page_area
        if coverage < FULL_PAGE_IMAGE_COVERAGE:
            continue

        image_dpi = max(float(metadata.horizontal_dpi), float(metadata.vertical_dpi))
        if math.isfinite(image_dpi) and image_dpi > 0:
            candidates.append((coverage, image_dpi))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[1]


def _snap_near_integer(value: float) -> float:
    nearest_integer = round(value)
    if math.isclose(value, nearest_integer, abs_tol=0.05):
        return float(nearest_integer)
    return value


def assemble_searchable_pdf(
    pages: Sequence[RasterizedPage],
    page_regions: Sequence[Sequence[OcrRegion]],
    output_pdf: Path,
) -> None:
    if len(pages) != len(page_regions):
        raise ValueError("each page image must have a corresponding OCR result")
    if not pages:
        raise ValueError("at least one page image is required")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_pdf), pageCompression=1)
    pdf.setTitle(output_pdf.stem)

    for page, regions in zip(pages, page_regions, strict=True):
        with Image.open(page.image_path) as image:
            image_width, image_height = image.size
        scale_x = page.width_points / image_width
        scale_y = page.height_points / image_height
        pdf.setPageSize((page.width_points, page.height_points))
        pdf.drawImage(
            str(page.image_path),
            0,
            0,
            width=page.width_points,
            height=page.height_points,
            preserveAspectRatio=False,
            mask="auto",
        )
        for region in regions:
            _draw_invisible_region(
                pdf,
                region,
                image_height=image_height,
                scale_x=scale_x,
                scale_y=scale_y,
            )
        pdf.showPage()

    pdf.save()


def _to_pdf_point(
    point: Point,
    *,
    image_height: int,
    scale_x: float,
    scale_y: float,
) -> Point:
    return Point(point.x * scale_x, (image_height - point.y) * scale_y)


def _draw_invisible_region(
    pdf: canvas.Canvas,
    region: OcrRegion,
    *,
    image_height: int,
    scale_x: float,
    scale_y: float,
) -> None:
    top_left, top_right, bottom_right, bottom_left = (
        _to_pdf_point(
            point,
            image_height=image_height,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        for point in region.polygon
    )
    width = (
        math.dist((top_left.x, top_left.y), (top_right.x, top_right.y))
        + math.dist((bottom_left.x, bottom_left.y), (bottom_right.x, bottom_right.y))
    ) / 2
    height = (
        math.dist((top_left.x, top_left.y), (bottom_left.x, bottom_left.y))
        + math.dist((top_right.x, top_right.y), (bottom_right.x, bottom_right.y))
    ) / 2
    if width <= 0 or height <= 0:
        return

    angle = math.degrees(math.atan2(bottom_right.y - bottom_left.y, bottom_right.x - bottom_left.x))
    font_size = max(height, 1.0)
    natural_width = pdfmetrics.stringWidth(region.text, PDF_FONT, font_size)
    horizontal_scale = 100.0 if natural_width <= 0 else 100.0 * width / natural_width

    pdf.saveState()
    pdf.translate(bottom_left.x, bottom_left.y)
    pdf.rotate(angle)
    text = pdf.beginText()
    text.setTextOrigin(0, 0)
    text.setFont(PDF_FONT, font_size)
    text.setHorizScale(horizontal_scale)
    text.setTextRenderMode(3)
    text.textOut(region.text)
    pdf.drawText(text)
    pdf.restoreState()
