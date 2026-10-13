from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from scanwich.images import UPRIGHT_TURNS, crop_box, is_pdf, read_photo, turn_clockwise
from scanwich.latex import render_latex
from scanwich.layout import PAGE_TOP_UPRIGHT, SURROGATE, Figure, PageLayout
from scanwich.ocr import LayoutBackend, OcrBackend
from scanwich.pdf import PNG_COMPRESSION_LEVEL, assemble_searchable_pdf, rasterize_pdf

# The illustrations of a LaTeX document are in a directory next to it. The name of the
# directory is the name of the document and this suffix.
FIGURE_DIRECTORY_SUFFIX = "-figures"
# The names that Scanwich gives to the illustrations. A conversion replaces only files
# with such a name.
FIGURE_FILE_NAME = re.compile(r"page-\d{4,}-figure-\d{2,}\.jpg")
FIGURE_JPEG_QUALITY = 95
# Full color resolution keeps the lines of a drawing sharp.
FIGURE_JPEG_SUBSAMPLING = "4:4:4"
# A backend does not give the edges of an illustration exactly. Thus the crop is this much
# larger at each side, on the 0-to-1000 grid of the page image.
FIGURE_MARGIN = 5.0
# LaTeX reads the name of the figure directory in a command, so the name has only
# characters that are safe there.
UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_DOCUMENT_NAME = "document"
# The name of a LaTeX document ends with this suffix.
LATEX_SUFFIX = ".tex"
# A figure directory with a changed name also has this many characters of a hash of the
# document name.
NAME_HASH_CHARACTERS = 8
PAGE_ROTATIONS = (0, 90, 180, 270)

logger = logging.getLogger(__name__)


class PageLayoutError(RuntimeError):
    """A page, or the full document, has no usable layout."""


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
        temporary_pdf.replace(output_pdf)
        logger.info("Wrote searchable PDF to %s", output_pdf)


def convert_to_latex(
    input_path: Path,
    output_tex: Path,
    *,
    backend: LayoutBackend,
    dpi: int | None = None,
    rotate: int | None = None,
    ocr_output_directory: Path | None = None,
) -> None:
    """Write a PDF or a photo of a page as one LaTeX document.

    The illustrations go to a directory next to the document. `rotate` is the direction
    of the pages: the function turns each page image clockwise by that angle before
    recognition. Without it, the backend finds the direction of each page.

    The function raises `PageLayoutError` when a page has no usable layout, and when no
    page has content. It then writes no document.
    """
    input_path = input_path.expanduser().resolve()
    output_tex = output_tex.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if input_path == output_tex:
        raise ValueError("input and output paths must be different")
    if dpi is not None and dpi <= 0:
        raise ValueError("DPI must be greater than zero")
    if rotate is not None and rotate not in PAGE_ROTATIONS:
        raise ValueError("a page image turns by 0, 90, 180, or 270 degrees")
    if not isinstance(backend, LayoutBackend):
        raise TypeError(
            f"{type(backend).__name__} does not implement recognize_layout(image_path); "
            "LaTeX output needs an OCR backend that recognizes the layout of a page"
        )

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    figure_directory = _figure_directory(output_tex)
    with TemporaryDirectory(prefix="scanwich-") as temporary_directory:
        work_directory = Path(temporary_directory)
        page_images = _page_images(input_path, work_directory / "pages", dpi=dpi)
        layouts: list[PageLayout] = []
        for page_number, page_image in enumerate(page_images, start=1):
            logger.info(
                "Recognizing the layout of page %d/%d with %s",
                page_number,
                len(page_images),
                type(backend).__name__,
            )
            layout = _recognize_upright_layout(backend, page_image, page_number, rotate)
            layout = _crop_figures(
                layout,
                page_image,
                work_directory / "figures",
                page_number=page_number,
                figure_directory_name=figure_directory.name,
            )
            layouts.append(layout)
            logger.info("Recognized %d block(s) on page %d", len(layout.blocks), page_number)
            if not layout.blocks:
                logger.warning(
                    "Page %d has no content in its layout; compare it with the source",
                    page_number,
                )
            if ocr_output_directory is not None:
                _write_json(ocr_output_directory, page_number, layout.to_json())
                logger.info("Wrote layout JSON for page %d", page_number)

        if not any(layout.blocks for layout in layouts):
            raise PageLayoutError(
                f"no page of {input_path} has content in its layout, so there is no document"
            )
        logger.info("Writing LaTeX document")
        # The file is in the directory of the document, so the last step only renames it.
        temporary_tex = output_tex.with_name(f".{output_tex.name}.scanwich-tmp")
        try:
            temporary_tex.write_text(render_latex(layouts), encoding="utf-8")
            figure_count = _replace_figures(work_directory / "figures", figure_directory)
            temporary_tex.replace(output_tex)
        finally:
            temporary_tex.unlink(missing_ok=True)
    logger.info("Wrote LaTeX document to %s", output_tex)
    if figure_count:
        logger.info("Wrote %d illustration(s) to %s", figure_count, figure_directory)


def _page_images(input_path: Path, output_directory: Path, *, dpi: int | None) -> list[Path]:
    """Return one image file for each page of a PDF, or one for a photo of a page."""
    if is_pdf(input_path):
        logger.info("Rasterizing %s", input_path)
        pages = rasterize_pdf(input_path, output_directory, dpi=dpi)
        logger.info("Rasterized %d page(s)", len(pages))
        return [page.image_path for page in pages]
    logger.info("Reading %s as the image of one page", input_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    image_path = output_directory / "page-000000.png"
    image = read_photo(input_path)
    logger.info("The page image has %dx%d pixels", image.width, image.height)
    image.save(image_path, format="PNG", compress_level=PNG_COMPRESSION_LEVEL)
    return [image_path]


def _recognize_upright_layout(
    backend: LayoutBackend, page_image: Path, page_number: int, rotate: int | None
) -> PageLayout:
    """Return the layout of a page, with boxes that fit the page image file.

    With `rotate`, the caller gives the direction of the page. The function turns the page
    image file and tells the backend that the page is upright. Without `rotate`, the
    backend finds the direction, and the function turns the page image file one time when
    the backend tells it to.

    The function raises `PageLayoutError` for a page that has no usable layout, so that a
    page does not disappear from the document without a failure.
    """
    if rotate is not None:
        if rotate:
            _turn_page_image(page_image, rotate)
        layout = backend.recognize_layout(page_image, upright=True)
        if layout.page_top == PAGE_TOP_UPRIGHT:
            return layout
        if not layout.blocks:
            raise PageLayoutError(
                f"page {page_number} has no usable layout: the backend returned no blocks and "
                f"reports the top of the page at the {layout.page_top} of the image, but "
                f"--rotate {rotate} gives a different direction; check the value of --rotate"
            )
        logger.warning(
            "Page %d: --rotate %d gives the direction of the page, but the backend reports "
            "the top of the page at the %s of the image; using the blocks of the backend",
            page_number,
            rotate,
            layout.page_top,
        )
        return replace(layout, page_top=PAGE_TOP_UPRIGHT)

    layout = backend.recognize_layout(page_image)
    if layout.page_top == PAGE_TOP_UPRIGHT:
        return layout
    first_side = layout.page_top
    degrees = UPRIGHT_TURNS[first_side]
    logger.info(
        "The top of page %d is at the %s of its image; turning the image %d degrees "
        "clockwise and recognizing the page again",
        page_number,
        first_side,
        degrees,
    )
    _turn_page_image(page_image, degrees)
    layout = backend.recognize_layout(page_image)
    if layout.page_top != PAGE_TOP_UPRIGHT:
        raise PageLayoutError(
            f"page {page_number} has no usable layout: the backend reported the top of the "
            f"page at the {first_side} of the image and, after a turn of {degrees} degrees "
            f"clockwise, at the {layout.page_top}; set --rotate to give the direction of the "
            "pages"
        )
    return layout


def _turn_page_image(page_image: Path, degrees: int) -> None:
    with Image.open(page_image) as image:
        turned = turn_clockwise(image, degrees)
    turned.save(page_image, format="PNG", compress_level=PNG_COMPRESSION_LEVEL)


def _crop_figures(
    layout: PageLayout,
    page_image: Path,
    output_directory: Path,
    *,
    page_number: int,
    figure_directory_name: str,
) -> PageLayout:
    """Save each illustration of a page as an image file, and return the layout with the paths."""
    if not any(isinstance(block, Figure) for block in layout.blocks):
        return layout
    output_directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    figure_number = 0
    with Image.open(page_image) as image:
        for block in layout.blocks:
            if isinstance(block, Figure):
                figure_number += 1
                name = f"page-{page_number:04d}-figure-{figure_number:02d}.jpg"
                figure = crop_box(image, block.box, margin=FIGURE_MARGIN)
                if figure.mode not in {"L", "RGB"}:
                    figure = figure.convert("RGB")
                figure.save(
                    output_directory / name,
                    format="JPEG",
                    quality=FIGURE_JPEG_QUALITY,
                    subsampling=FIGURE_JPEG_SUBSAMPLING,
                )
                block = replace(block, image=f"{figure_directory_name}/{name}")
            blocks.append(block)
    return replace(layout, blocks=tuple(blocks))


def _figure_directory(output_tex: Path) -> Path:
    """Return the directory for the illustrations of a document.

    LaTeX reads the name of the directory in a command, so the name has only characters
    that are safe there. A document name with other characters also gives a part of its
    hash to the name of the directory. Thus two documents in one directory do not share
    a figure directory.
    """
    stem = output_tex.stem
    name = UNSAFE_NAME_CHARACTERS.sub("-", stem).strip("-.") or DEFAULT_DOCUMENT_NAME
    if name != stem or output_tex.suffix != LATEX_SUFFIX:
        name_hash = hashlib.sha256(os.fsencode(output_tex.name)).hexdigest()
        name = f"{name}-{name_hash[:NAME_HASH_CHARACTERS]}"
    return output_tex.with_name(name + FIGURE_DIRECTORY_SUFFIX)


def _figure_files(directory: Path) -> list[Path]:
    """Return the files of a directory that have the name of an illustration of Scanwich."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if FIGURE_FILE_NAME.fullmatch(path.name) and path.is_file()
    )


def _replace_figures(new_directory: Path, figure_directory: Path) -> int:
    """Move the new illustrations to the figure directory, and return their number.

    The function first removes the illustrations of an earlier conversion of the document.
    It does not remove a file with a different name.
    """
    for old_figure in _figure_files(figure_directory):
        old_figure.unlink()
    new_figures = _figure_files(new_directory)
    if new_figures:
        figure_directory.mkdir(parents=True, exist_ok=True)
        for new_figure in new_figures:
            # The temporary directory can be on a different file system.
            shutil.move(new_figure, figure_directory / new_figure.name)
    elif figure_directory.is_dir() and not any(figure_directory.iterdir()):
        figure_directory.rmdir()
    return len(new_figures)


def _write_ocr_json(output_directory: Path, page_number: int, regions: Sequence[Any]) -> None:
    _write_json(output_directory, page_number, [region.to_json() for region in regions])


def _write_json(output_directory: Path, page_number: int, payload: Any) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"page-{page_number:06d}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    # A file in UTF-8 cannot have one half of a character. JSON has an escape for it.
    text = SURROGATE.sub(lambda match: f"\\u{ord(match.group()):04x}", text)
    output_path.write_text(text + "\n", encoding="utf-8")


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
