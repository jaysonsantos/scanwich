from __future__ import annotations

import contextlib
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
from scanwich.layout import (
    BOX_SIDES,
    GRID_MAX,
    PAGE_TOP_UPRIGHT,
    SURROGATE,
    Box,
    Figure,
    PageLayout,
)
from scanwich.ocr import FigureLocator, LayoutBackend, OcrBackend
from scanwich.pdf import PNG_COMPRESSION_LEVEL, assemble_searchable_pdf, rasterize_pdf

# The illustrations of a LaTeX document are in a directory next to it. The name of the
# directory is the name of the document and this suffix.
FIGURE_DIRECTORY_SUFFIX = "-figures"
# The names that Scanwich gives to the illustrations. A conversion replaces only files
# with such a name.
FIGURE_FILE_NAME = re.compile(r"page-\d{4,}-figure-\d{2,}\.jpg")
# While a conversion puts its result in place, the figure directory has temporary files:
# a new illustration that waits for its name, and an earlier illustration that the new
# result replaces. Their names are the name of the illustration with a dot before it and
# one of these suffixes.
NEW_FIGURE_SUFFIX = ".scanwich-new"
OLD_FIGURE_SUFFIX = ".scanwich-old"
TEMPORARY_FIGURE_FILE_NAME = re.compile(
    rf"\.{FIGURE_FILE_NAME.pattern}({re.escape(NEW_FIGURE_SUFFIX)}|{re.escape(OLD_FIGURE_SUFFIX)})"
)
FIGURE_JPEG_QUALITY = 95
# Full color resolution keeps the lines of a drawing sharp.
FIGURE_JPEG_SUBSAMPLING = "4:4:4"
# A backend does not give the edges of an illustration exactly. Thus the crop is this much
# larger at each side, on the 0-to-1000 grid of the page image.
FIGURE_MARGIN = 5.0
# The box of an illustration in the layout of a full page can be wrong by a large part of
# the illustration. A backend that is a `FigureLocator` gets a second request for each
# illustration, with only the part of the page image that has it. That part is the box of
# the layout and a margin at each side: this distance on the 0-to-1000 grid of the page
# image, or this ratio of the size of the box when that is more.
FIGURE_SEARCH_MARGIN = 100.0
FIGURE_SEARCH_MARGIN_RATIO = 0.5
# The part of the page image must show paper around the illustration. When the new box is
# nearer than this to a side of the part, a label of the illustration can be outside the
# part. The part then becomes larger at that side, and the backend gets it again.
FIGURE_SEARCH_MIN_GAP = 30.0
# The maximum number of requests for the box of one illustration, and for the boxes of
# all illustrations of one page. A page layout can have many blocks that are not
# illustrations of the page, for example each ornament.
FIGURE_SEARCH_REQUESTS = 3
FIGURE_SEARCH_PAGE_REQUESTS = 12
# The box of a second request is not better than the box of the page layout in each case.
# A second box that has this many times the area of the first box, or less than the
# inverse, is in doubt.
FIGURE_BOX_AREA_RATIO = 4.0
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
    flow: bool = False,
    ocr_output_directory: Path | None = None,
) -> None:
    """Write a PDF or a photo of a page as one LaTeX document.

    The illustrations go to a directory next to the document. `rotate` is the direction
    of the pages: the function turns each page image clockwise by that angle before
    recognition. Without it, the backend finds the direction of each page. Each source
    page is one page of the document. With `flow`, the text flows across the pages of the
    document.

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
            if isinstance(backend, FigureLocator):
                layout = _locate_figures(
                    backend, layout, page_image, work_directory / "regions", page_number
                )
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
            temporary_tex.write_text(render_latex(layouts, flow=flow), encoding="utf-8")
            figure_count = _install_result(
                temporary_tex, output_tex, work_directory / "figures", figure_directory
            )
        finally:
            temporary_tex.unlink(missing_ok=True)
    logger.info("Wrote LaTeX document to %s", output_tex)
    if not flow:
        # Scanwich starts no TeX engine, so it does not know the height of a page.
        logger.info(
            "Each of the %d source page(s) is one page of the document. A TeX engine "
            "writes a warning for a source page that it makes smaller to keep it on one page",
            len(layouts),
        )
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


def _locate_figures(
    backend: FigureLocator,
    layout: PageLayout,
    page_image: Path,
    work_directory: Path,
    page_number: int,
) -> PageLayout:
    """Return the layout with the box that a second request gives for each illustration."""
    if not any(isinstance(block, Figure) for block in layout.blocks):
        return layout
    work_directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    figure_number = 0
    requests_left = FIGURE_SEARCH_PAGE_REQUESTS
    without_request = 0
    with Image.open(page_image) as image:
        for block in layout.blocks:
            if isinstance(block, Figure):
                figure_number += 1
                if requests_left:
                    box, requests = _locate_figure(
                        backend,
                        image,
                        block,
                        work_directory / "region.png",
                        f"illustration {figure_number} of page {page_number}",
                        min(FIGURE_SEARCH_REQUESTS, requests_left),
                    )
                    requests_left -= requests
                    block = replace(block, box=box)
                else:
                    without_request += 1
            blocks.append(block)
    if without_request:
        logger.warning(
            "Page %d has %d illustration(s) in its layout, and one page has a limit of %d "
            "requests for the boxes of illustrations; %d illustration(s) use the box of "
            "the page layout. Compare them with the source",
            page_number,
            figure_number,
            FIGURE_SEARCH_PAGE_REQUESTS,
            without_request,
        )
    return replace(layout, blocks=tuple(blocks))


def _locate_figure(
    backend: FigureLocator,
    image: Image.Image,
    figure: Figure,
    region_path: Path,
    name: str,
    max_requests: int,
) -> tuple[Box, int]:
    """Return the box of one illustration on the grid of the page image, and the requests.

    The backend gets a part of the page image around the box of the layout. It gets a
    larger part when the first part does not show the full illustration with paper around
    it, until `max_requests` requests. The function keeps the box of the layout when the
    backend gives no box, and it does not use a new box that is in doubt without a warning.
    """
    region = _search_region(figure.box)
    box = figure.box
    for request in range(1, max_requests + 1):
        crop_box(image, region).save(region_path, format="PNG", compress_level=1)
        try:
            location = backend.locate_figure(
                region_path, estimate=_box_in_region(box, region), caption=figure.caption
            )
        except Exception as error:
            # The page has a layout, so a failure of this request does not stop the
            # conversion, for each type of error of a backend.
            logger.warning(
                "The request for the box of %s failed (%s: %s); the crop uses the box that "
                "the backend gave before. Compare the illustration with the source",
                name,
                type(error).__name__,
                error,
            )
            return _checked_box(figure.box, box, name), request
        if location is None:
            logger.warning(
                "The backend found no illustration in the part of the page image for %s; "
                "the crop uses the box that the backend gave before. Compare the "
                "illustration with the source",
                name,
            )
            return _checked_box(figure.box, box, name), request
        box = _box_on_page(location.box, region)
        open_sides = _open_sides(box, region, location.cut_sides)
        if not open_sides:
            break
        if request == max_requests:
            still_cut = [side for side in BOX_SIDES if side in open_sides & location.cut_sides]
            if still_cut:
                # A crop that is too large loses nothing. A crop that cuts the
                # illustration loses a part of it.
                box = replace(box, **{side: getattr(region, side) for side in still_cut})
                logger.warning(
                    "After %d requests, the backend still reports that %s continues at "
                    "the %s; the crop is larger at such a side. Compare the illustration "
                    "with the source",
                    request,
                    name,
                    ", ".join(still_cut),
                )
                # This box is large on purpose, and it has its warning.
                return box, request
            break
        region = _grown_region(region, open_sides)
    box = _checked_box(figure.box, box, name)
    logger.info(
        "The box of %s is %s after %d request(s) for it; the page layout gave %s",
        name,
        _box_text(box),
        request,
        _box_text(figure.box),
    )
    return box, request


def _checked_box(first: Box, second: Box, name: str) -> Box:
    """Return the box for the crop, from the box of the page layout and a second box.

    The second box is the better one in most cases. It is in doubt when it has no part
    of the first box, and when its area is very different. A crop that is too large loses
    nothing, and a crop that cuts the illustration loses a part of it. Thus the crop then
    has the two boxes. The first box alone is not the safe one: a box of a page layout
    can be on the caption below a small illustration, and the second box is then correct.
    """
    if second == first:
        return first
    overlap_width = min(first.right, second.right) - max(first.left, second.left)
    overlap_height = min(first.bottom, second.bottom) - max(first.top, second.top)
    ratio = (second.width * second.height) / (first.width * first.height)
    if overlap_width <= 0 or overlap_height <= 0:
        doubt = "has no part of"
    elif 1 / FIGURE_BOX_AREA_RATIO <= ratio <= FIGURE_BOX_AREA_RATIO:
        return second
    else:
        doubt = f"is much {'smaller' if ratio < 1 else 'larger'} than"
    logger.warning(
        "The second box of %s (%s) %s the box of the page layout (%s); the crop has the "
        "two boxes. Compare the illustration with the source",
        name,
        _box_text(second),
        doubt,
        _box_text(first),
    )
    return Box(
        left=min(first.left, second.left),
        top=min(first.top, second.top),
        right=max(first.right, second.right),
        bottom=max(first.bottom, second.bottom),
    )


def _search_region(box: Box) -> Box:
    """Return the part of the page that the first request for an illustration shows."""
    margin_x = max(FIGURE_SEARCH_MARGIN, FIGURE_SEARCH_MARGIN_RATIO * box.width)
    margin_y = max(FIGURE_SEARCH_MARGIN, FIGURE_SEARCH_MARGIN_RATIO * box.height)
    return Box(
        left=max(0.0, box.left - margin_x),
        top=max(0.0, box.top - margin_y),
        right=min(GRID_MAX, box.right + margin_x),
        bottom=min(GRID_MAX, box.bottom + margin_y),
    )


def _open_sides(box: Box, region: Box, cut_sides: frozenset[str]) -> set[str]:
    """Return the sides where a larger part of the page can show more of an illustration.

    At such a side, the backend reports that the illustration continues, or the box is
    near the side of the part. A side at the edge of the page cannot move.
    """
    gaps = {
        "left": box.left - region.left,
        "top": box.top - region.top,
        "right": region.right - box.right,
        "bottom": region.bottom - box.bottom,
    }
    at_page_edge = {
        "left": region.left <= 0.0,
        "top": region.top <= 0.0,
        "right": region.right >= GRID_MAX,
        "bottom": region.bottom >= GRID_MAX,
    }
    return {
        side
        for side in BOX_SIDES
        if not at_page_edge[side] and (side in cut_sides or gaps[side] < FIGURE_SEARCH_MIN_GAP)
    }


def _grown_region(region: Box, sides: set[str]) -> Box:
    return Box(
        left=max(0.0, region.left - FIGURE_SEARCH_MARGIN) if "left" in sides else region.left,
        top=max(0.0, region.top - FIGURE_SEARCH_MARGIN) if "top" in sides else region.top,
        right=(
            min(GRID_MAX, region.right + FIGURE_SEARCH_MARGIN) if "right" in sides else region.right
        ),
        bottom=(
            min(GRID_MAX, region.bottom + FIGURE_SEARCH_MARGIN)
            if "bottom" in sides
            else region.bottom
        ),
    )


def _box_in_region(box: Box, region: Box) -> Box:
    """Return a box of the page grid on the 0-to-1000 grid of a part of the page."""
    scale_x = GRID_MAX / region.width
    scale_y = GRID_MAX / region.height
    return Box(
        left=(box.left - region.left) * scale_x,
        top=(box.top - region.top) * scale_y,
        right=(box.right - region.left) * scale_x,
        bottom=(box.bottom - region.top) * scale_y,
    )


def _box_on_page(box: Box, region: Box) -> Box:
    """Return a box of the grid of a part of the page on the 0-to-1000 grid of the page."""
    scale_x = region.width / GRID_MAX
    scale_y = region.height / GRID_MAX
    # One decimal place is less than one pixel of a page image, and the layout JSON
    # stays easy to read.
    return Box(
        left=round(region.left + box.left * scale_x, 1),
        top=round(region.top + box.top * scale_y, 1),
        right=round(region.left + box.right * scale_x, 1),
        bottom=round(region.top + box.bottom * scale_y, 1),
    )


def _box_text(box: Box) -> str:
    return f"left {box.left:.0f}, top {box.top:.0f}, right {box.right:.0f}, bottom {box.bottom:.0f}"


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


def _install_result(
    temporary_tex: Path, output_tex: Path, new_directory: Path, figure_directory: Path
) -> int:
    """Put the new document and its illustrations in place, and return their number.

    An earlier document includes its illustrations by name, so the document and the
    illustrations are one result. A failure must not leave a part of the new result in
    the place of the earlier one. The function has three steps:

    1. It copies each new illustration to the figure directory, under a temporary name.
       This is the step that needs space on the disk. The earlier result does not change.
    2. It gives a temporary name to each earlier illustration, gives the new
       illustrations their names, and replaces the document. Each of these is a rename
       in one directory.
    3. It removes the earlier illustrations.

    When step 1 or step 2 fails, the function puts the earlier result back and raises the
    error. It does not remove a file of the figure directory with a different name.
    """
    new_figures = _figure_files(new_directory)
    old_figures = _figure_files(figure_directory)
    made_directory = bool(new_figures) and not figure_directory.exists()
    if new_figures:
        figure_directory.mkdir(parents=True, exist_ok=True)
    # A conversion that the system stopped can leave a new illustration here. It is not
    # a part of a result.
    _remove_temporary_figures(figure_directory, NEW_FIGURE_SUFFIX)
    waiting: list[tuple[Path, Path]] = []
    set_aside: list[tuple[Path, Path]] = []
    in_place: list[Path] = []
    try:
        for new_figure in new_figures:
            figure = figure_directory / new_figure.name
            temporary = figure.with_name(f".{figure.name}{NEW_FIGURE_SUFFIX}")
            waiting.append((temporary, figure))
            # The temporary directory can be on a different file system.
            shutil.copyfile(new_figure, temporary)
        for old_figure in old_figures:
            temporary = old_figure.with_name(f".{old_figure.name}{OLD_FIGURE_SUFFIX}")
            old_figure.replace(temporary)
            set_aside.append((temporary, old_figure))
        for temporary, figure in waiting:
            temporary.replace(figure)
            in_place.append(figure)
        temporary_tex.replace(output_tex)
    except BaseException:
        _restore_earlier_figures(waiting, set_aside, in_place)
        if made_directory:
            # The directory is empty when the function made it and removed its files.
            with contextlib.suppress(OSError):
                figure_directory.rmdir()
        raise
    # The new result is complete. The earlier illustrations have no document now.
    _remove_temporary_figures(figure_directory, OLD_FIGURE_SUFFIX)
    if not new_figures and figure_directory.is_dir() and not any(figure_directory.iterdir()):
        figure_directory.rmdir()
    return len(new_figures)


def _restore_earlier_figures(
    waiting: Sequence[tuple[Path, Path]],
    set_aside: Sequence[tuple[Path, Path]],
    in_place: Sequence[Path],
) -> None:
    """Remove the new illustrations, and give the earlier illustrations their names again.

    The function continues after an error of one file, so that it puts back as much as
    is possible. An earlier illustration that it cannot put back stays in the figure
    directory under its temporary name, and an `ERROR` log record names it.
    """
    for path in [*in_place, *(temporary for temporary, _ in waiting)]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.error("Could not remove the new illustration %s", path)
    for temporary, old_figure in set_aside:
        try:
            temporary.replace(old_figure)
        except OSError:
            logger.error(
                "Could not put back the illustration %s of the earlier document; it is the file %s",
                old_figure,
                temporary,
            )


def _remove_temporary_figures(figure_directory: Path, suffix: str) -> None:
    """Remove the temporary files of illustrations that have this suffix.

    A file that stays is not a defect of the result, so an error is only a warning.
    """
    if not figure_directory.is_dir():
        return
    for path in figure_directory.iterdir():
        is_temporary = TEMPORARY_FIGURE_FILE_NAME.fullmatch(path.name) is not None
        if is_temporary and path.name.endswith(suffix) and path.is_file():
            try:
                path.unlink()
            except OSError:
                logger.warning("Could not remove the temporary file %s", path)


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
