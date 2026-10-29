"""Read a page upright and turned, and put the words on the original page.

A detector that only finds upright text misses a word printed at 90 or 270 degrees.
The extra passes turn the page so that direction is upright. The boxes are then mapped
back onto the original page.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from scanwich.images import turn_clockwise
from scanwich.models import OcrRegion, Point

# 180 is not a default: pass it with the `rotations` option when a page is upside down.
DEFAULT_ROTATIONS = (90, 270)
# A turned pass can see an upright word again. It replaces that word only when the
# existing reading is weak and the new one is clearly more confident. A small gain is
# not enough: a weak but correct horizontal word must not be overwritten by a misread.
WEAK_CONFIDENCE = 0.5
REPLACEMENT_MARGIN = 0.2
# Two regions are the same spot when at least this share of one box lies inside the other.
OVERLAP_FRACTION = 0.5

# recognize_image(path, degrees) reads one image. degrees is 0 for the upright page.
RecognizeImage = Callable[[Path, int], Sequence[OcrRegion]]
AsyncRecognizeImage = Callable[[Path, int], Awaitable[Sequence[OcrRegion]]]


def rotations_from_option(value: Any) -> tuple[int, ...]:
    """Return the extra clockwise angles, without the upright pass."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError("the 'rotations' backend option must be a list of 90, 180, and 270")
    rotations: list[int] = []
    for angle in value:
        if isinstance(angle, bool) or angle not in (90, 180, 270):
            raise ValueError(
                "rotations must be 90, 180, or 270 degrees; "
                f"the upright pass always runs, and {angle!r} is not one of those turns"
            )
        if angle not in rotations:
            rotations.append(angle)
    return tuple(rotations)


def recognize_turned(
    image_path: Path,
    rotations: Sequence[int],
    recognize_image: RecognizeImage,
    *,
    reader_name: str,
    log: logging.Logger,
) -> list[OcrRegion]:
    """Return the text of one page, including words that are upright only after a turn.

    `recognize_image` returns regions in the pixel coordinates of the image file it reads.
    """
    started = time.monotonic()
    regions = list(recognize_image(image_path, 0))
    if not rotations:
        return regions
    log.info(
        "The upright pass found %d text region(s) in %.1f seconds",
        len(regions),
        time.monotonic() - started,
    )
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        _log_extra_passes(log, reader_name, rotations)
        with TemporaryDirectory(prefix="scanwich-ocr-") as temporary_directory:
            directory = Path(temporary_directory)
            for degrees in rotations:
                pass_started = time.monotonic()
                turned_path = directory / f"turn-{degrees}.png"
                _save_turn(image, turned_path, degrees)
                turned_regions = list(recognize_image(turned_path, degrees))
                regions = _keep_turned_regions(
                    regions, turned_regions, degrees=degrees, width=width, height=height
                )
                _log_pass(log, degrees, len(turned_regions), pass_started, len(regions))
    return regions


async def recognize_turned_async(
    image_path: Path,
    rotations: Sequence[int],
    recognize_image: AsyncRecognizeImage,
    *,
    reader_name: str,
    log: logging.Logger,
) -> list[OcrRegion]:
    """Return the text of one page, as `recognize_turned` does, awaiting each pass."""
    started = time.monotonic()
    regions = list(await recognize_image(image_path, 0))
    if not rotations:
        return regions
    log.info(
        "The upright pass found %d text region(s) in %.1f seconds",
        len(regions),
        time.monotonic() - started,
    )

    def page_size() -> tuple[int, int]:
        with Image.open(image_path) as image:
            image.load()
            return image.size

    width, height = await asyncio.to_thread(page_size)
    _log_extra_passes(log, reader_name, rotations)
    with TemporaryDirectory(prefix="scanwich-ocr-") as temporary_directory:
        directory = Path(temporary_directory)
        for degrees in rotations:
            pass_started = time.monotonic()
            turned_path = directory / f"turn-{degrees}.png"

            def write(degrees: int = degrees, turned_path: Path = turned_path) -> None:
                with Image.open(image_path) as image:
                    image.load()
                    _save_turn(image, turned_path, degrees)

            await asyncio.to_thread(write)
            turned_regions = list(await recognize_image(turned_path, degrees))
            regions = _keep_turned_regions(
                regions, turned_regions, degrees=degrees, width=width, height=height
            )
            _log_pass(log, degrees, len(turned_regions), pass_started, len(regions))
    return regions


def _log_extra_passes(log: logging.Logger, reader_name: str, rotations: Sequence[int]) -> None:
    log.info(
        "Reading the page again at %s degrees so rotated text is upright; "
        "%s reads this page %d times",
        _degrees_text(rotations),
        reader_name,
        1 + len(rotations),
    )


def _log_pass(log: logging.Logger, degrees: int, found: int, started: float, kept: int) -> None:
    log.info(
        "The %d-degree pass found %d text region(s) in %.1f seconds; %d region(s) are kept",
        degrees,
        found,
        time.monotonic() - started,
        kept,
    )


def _save_turn(image: Image.Image, path: Path, degrees: int) -> None:
    # The same fast compression the page images use. The file is temporary.
    turn_clockwise(image, degrees).save(path, format="PNG", compress_level=1)


def _keep_turned_regions(
    regions: list[OcrRegion],
    turned_regions: Sequence[OcrRegion],
    *,
    degrees: int,
    width: int,
    height: int,
) -> list[OcrRegion]:
    mapped = [region_on_page(region, degrees, width, height) for region in turned_regions]
    return merge_regions(regions, mapped)


def _degrees_text(rotations: Sequence[int]) -> str:
    if len(rotations) == 1:
        return str(rotations[0])
    return f"{', '.join(str(angle) for angle in rotations[:-1])} and {rotations[-1]}"


def region_on_page(region: OcrRegion, degrees: int, width: int, height: int) -> OcrRegion:
    """Return a region from a turned page image, in the original page's coordinates.

    `degrees` is the clockwise angle the page was turned by before recognition. The point
    order stays the order the reader gave, which is the order around the upright glyphs.
    """
    return OcrRegion(
        text=region.text,
        polygon=tuple(point_on_page(point, degrees, width, height) for point in region.polygon),  # type: ignore[arg-type]
        confidence=region.confidence,
    )


def point_on_page(point: Point, degrees: int, width: int, height: int) -> Point:
    """Map a point on a page turned clockwise by `degrees` back to the original page."""
    if degrees == 90:
        return Point(point.y, height - point.x)
    if degrees == 180:
        return Point(width - point.x, height - point.y)
    if degrees == 270:
        return Point(width - point.y, point.x)
    raise ValueError(f"unsupported rotation {degrees}")


def merge_regions(regions: Sequence[OcrRegion], extra: Sequence[OcrRegion]) -> list[OcrRegion]:
    """Add regions from one turned pass, without a second copy of the same spot.

    A new region that lands on one existing region replaces it only when that reading is
    weak and the new one is clearly more confident. A clear upright word, and a weak but
    still plausible one, therefore stay. A region with no confidence does not replace
    another one. A new region that covers more than one existing region is left out: it
    is a line of several words, and replacing them would drop the upright readings.
    """
    merged = list(regions)
    for region in extra:
        overlaps = [
            index
            for index, kept in enumerate(merged)
            if overlap_fraction(region, kept) >= OVERLAP_FRACTION
        ]
        if not overlaps:
            merged.append(region)
        elif len(overlaps) == 1 and _prefer_new(merged[overlaps[0]], region):
            merged[overlaps[0]] = region
    return merged


def _prefer_new(current: OcrRegion, new: OcrRegion) -> bool:
    new_confidence = 0.0 if new.confidence is None else new.confidence
    current_confidence = 0.0 if current.confidence is None else current.confidence
    # A reader that gives no confidence must not overwrite a word with a guess that is
    # also unscored. Both confidences are then 0.
    if current.confidence is None or new.confidence is None:
        return False
    return (
        current_confidence < WEAK_CONFIDENCE
        and new_confidence >= current_confidence + REPLACEMENT_MARGIN
    )


def overlap_fraction(region: OcrRegion, other: OcrRegion) -> float:
    """Return the larger share of either axis-aligned box that lies inside the other."""
    left_a, top_a, right_a, bottom_a = _bounds(region)
    left_b, top_b, right_b, bottom_b = _bounds(other)
    width = min(right_a, right_b) - max(left_a, left_b)
    height = min(bottom_a, bottom_b) - max(top_a, top_b)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    fractions = []
    area_a = (right_a - left_a) * (bottom_a - top_a)
    area_b = (right_b - left_b) * (bottom_b - top_b)
    if area_a > 0:
        fractions.append(intersection / area_a)
    if area_b > 0:
        fractions.append(intersection / area_b)
    return max(fractions, default=0.0)


def _bounds(region: OcrRegion) -> tuple[float, float, float, float]:
    xs = [point.x for point in region.polygon]
    ys = [point.y for point in region.polygon]
    return min(xs), min(ys), max(xs), max(ys)
