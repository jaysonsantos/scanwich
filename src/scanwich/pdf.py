from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from scanwich.models import OcrRegion, Point

POINTS_PER_INCH = 72.0
PDF_FONT = "Helvetica"


class PdfPipelineError(RuntimeError):
    pass


def rasterize_pdf(
    input_pdf: Path,
    output_directory: Path,
    *,
    dpi: int,
    magick_command: str = "magick",
) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    output_pattern = output_directory / "page-%06d.png"
    command = [
        magick_command,
        "-density",
        str(dpi),
        str(input_pdf),
        "-background",
        "white",
        "-alpha",
        "remove",
        "-alpha",
        "off",
        str(output_pattern),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise PdfPipelineError(f"could not find ImageMagick command: {magick_command}") from error
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        raise PdfPipelineError(f"ImageMagick could not rasterize {input_pdf}: {details}") from error

    pages = sorted(output_directory.glob("page-*.png"))
    if not pages:
        raise PdfPipelineError(f"ImageMagick produced no pages for {input_pdf}")
    return pages


def assemble_searchable_pdf(
    page_images: Sequence[Path],
    page_regions: Sequence[Sequence[OcrRegion]],
    output_pdf: Path,
    *,
    dpi: int,
) -> None:
    if len(page_images) != len(page_regions):
        raise ValueError("each page image must have a corresponding OCR result")
    if not page_images:
        raise ValueError("at least one page image is required")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_pdf), pageCompression=1)
    pdf.setTitle(output_pdf.stem)
    scale = POINTS_PER_INCH / dpi

    for image_path, regions in zip(page_images, page_regions, strict=True):
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        page_width = image_width * scale
        page_height = image_height * scale
        pdf.setPageSize((page_width, page_height))
        pdf.drawImage(
            str(image_path),
            0,
            0,
            width=page_width,
            height=page_height,
            preserveAspectRatio=False,
            mask="auto",
        )
        for region in regions:
            _draw_invisible_region(pdf, region, image_height=image_height, scale=scale)
        pdf.showPage()

    pdf.save()


def _to_pdf_point(point: Point, *, image_height: int, scale: float) -> Point:
    return Point(point.x * scale, (image_height - point.y) * scale)


def _draw_invisible_region(
    pdf: canvas.Canvas,
    region: OcrRegion,
    *,
    image_height: int,
    scale: float,
) -> None:
    top_left, top_right, bottom_right, bottom_left = (
        _to_pdf_point(point, image_height=image_height, scale=scale) for point in region.polygon
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
