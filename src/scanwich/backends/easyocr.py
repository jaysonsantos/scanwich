from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from scanwich.images import turn_clockwise
from scanwich.models import OcrRegion, Point

logger = logging.getLogger(__name__)

# EasyOCR's detector finds text that is upright in the image it is given. A word printed
# at 90 or 270 degrees is missed on the upright pass. The extra passes turn the page so
# that direction is upright, then the boxes are mapped back onto the original page.
# 180 is not a default: pass it with the `rotations` option when a page is upside down.
DEFAULT_ROTATIONS = (90, 270)
# A turned pass can see an upright word again. It replaces that word only when the
# existing reading is weak and the new one is clearly more confident. A small gain is
# not enough: a weak but correct horizontal word must not be overwritten by a misread.
WEAK_CONFIDENCE = 0.5
REPLACEMENT_MARGIN = 0.2
# Two regions are the same spot when at least this share of one box lies inside the other.
OVERLAP_FRACTION = 0.5
# On a turned page the glyphs may be upside down. EasyOCR then recognizes each crop both
# ways and keeps the reading with the higher confidence. The box itself does not move.
UPSIDE_DOWN = (180,)
# EasyOCR otherwise shrinks the long side to this size. Nearby vertical labels then become
# one word. A turned pass keeps the page's own long side when it is larger than this.
DEFAULT_CANVAS_SIZE = 2560
# The detection image is not larger than this. A bigger page is still shrunk, but less
# than the default, and the cost of a turned pass stays bounded.
ROTATED_CANVAS_LIMIT = 4096


class EasyOcrBackend:
    """EasyOCR provider, imported lazily so other providers need not install it."""

    def __init__(self, *, languages: Sequence[str], options: Mapping[str, Any]) -> None:
        self._languages = list(languages)
        reader_options = dict(options)
        self._readtext_options = reader_options.pop("readtext", {})
        if not isinstance(self._readtext_options, dict):
            raise TypeError("EasyOCR's 'readtext' backend option must be a JSON object")
        self._rotations = _rotations_from_option(reader_options.pop("rotations", DEFAULT_ROTATIONS))
        reader_options.setdefault("gpu", False)
        self._reader_options = reader_options
        self._reader: Any | None = None

    def recognize(self, image_path: Path) -> list[OcrRegion]:
        started = time.monotonic()
        regions = self._read(str(image_path), self._readtext_options)
        if not self._rotations:
            return regions
        logger.info(
            "The upright pass found %d text region(s) in %.1f seconds",
            len(regions),
            time.monotonic() - started,
        )
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            logger.info(
                "Reading the page again at %s degrees so rotated text is upright; "
                "EasyOCR reads this page %d times",
                _degrees_text(self._rotations),
                1 + len(self._rotations),
            )
            with TemporaryDirectory(prefix="scanwich-ocr-") as temporary_directory:
                directory = Path(temporary_directory)
                for degrees in self._rotations:
                    pass_started = time.monotonic()
                    turned_path = directory / f"turn-{degrees}.png"
                    # The same fast compression the page images use. The file is temporary.
                    turn_clockwise(image, degrees).save(
                        turned_path,
                        format="PNG",
                        compress_level=1,
                    )
                    options = dict(self._readtext_options)
                    options.setdefault("rotation_info", list(UPSIDE_DOWN))
                    # Full resolution keeps a gap between labels that sit in a row once the
                    # page is turned. The upright pass keeps EasyOCR's own detection size.
                    long_side = max(width, height)
                    options.setdefault(
                        "canvas_size",
                        min(max(long_side, DEFAULT_CANVAS_SIZE), ROTATED_CANVAS_LIMIT),
                    )
                    turned_regions = self._read(str(turned_path), options)
                    mapped = [
                        _region_on_page(region, degrees, width, height) for region in turned_regions
                    ]
                    regions = _merge_regions(regions, mapped)
                    logger.info(
                        "The %d-degree pass found %d text region(s) in %.1f seconds; "
                        "%d region(s) are kept",
                        degrees,
                        len(turned_regions),
                        time.monotonic() - pass_started,
                        len(regions),
                    )
        return regions

    def _get_reader(self) -> Any:
        if self._reader is not None:
            return self._reader

        logger.info("Importing EasyOCR")
        started_at = time.monotonic()
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError("the EasyOCR backend requires the 'easyocr' package") from error

        logger.info(
            "Initializing EasyOCR for languages %s; first use may download models and take a while",
            ", ".join(self._languages),
        )
        self._reader = easyocr.Reader(self._languages, **self._reader_options)
        logger.info("EasyOCR ready after %.1f seconds", time.monotonic() - started_at)
        return self._reader

    def _read(self, image: str, options: Mapping[str, Any]) -> list[OcrRegion]:
        raw_regions = self._get_reader().readtext(
            image,
            detail=1,
            paragraph=False,
            **options,
        )
        regions: list[OcrRegion] = []
        for raw_polygon, raw_text, raw_confidence in raw_regions:
            text = str(raw_text).strip()
            if not text:
                continue
            points = tuple(Point(float(x), float(y)) for x, y in raw_polygon)
            if len(points) != 4:
                raise ValueError(f"EasyOCR returned a polygon with {len(points)} points")
            regions.append(
                OcrRegion(
                    text=text,
                    polygon=points,  # type: ignore[arg-type]
                    confidence=float(raw_confidence),
                )
            )
        return regions


def _rotations_from_option(value: Any) -> tuple[int, ...]:
    """Return the extra clockwise angles, without the upright pass."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = [value]
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError("EasyOCR's 'rotations' backend option must be a list of 90, 180, and 270")
    rotations: list[int] = []
    for angle in value:
        if isinstance(angle, bool) or angle not in (90, 180, 270):
            raise ValueError(
                "EasyOCR rotations must be 90, 180, or 270 degrees; "
                f"the upright pass always runs, and {angle!r} is not one of those turns"
            )
        if angle not in rotations:
            rotations.append(angle)
    return tuple(rotations)


def _degrees_text(rotations: Sequence[int]) -> str:
    if len(rotations) == 1:
        return str(rotations[0])
    return f"{', '.join(str(angle) for angle in rotations[:-1])} and {rotations[-1]}"


def _region_on_page(region: OcrRegion, degrees: int, width: int, height: int) -> OcrRegion:
    """Return a region from a turned page image, in the original page's coordinates.

    `degrees` is the clockwise angle the page was turned by before recognition. The point
    order stays the order EasyOCR gave, which is the order around the upright glyphs.
    """
    return OcrRegion(
        text=region.text,
        polygon=tuple(_point_on_page(point, degrees, width, height) for point in region.polygon),  # type: ignore[arg-type]
        confidence=region.confidence,
    )


def _point_on_page(point: Point, degrees: int, width: int, height: int) -> Point:
    """Map a point on a page turned clockwise by `degrees` back to the original page."""
    if degrees == 90:
        return Point(point.y, height - point.x)
    if degrees == 180:
        return Point(width - point.x, height - point.y)
    if degrees == 270:
        return Point(width - point.y, point.x)
    raise ValueError(f"unsupported rotation {degrees}")


def _merge_regions(regions: list[OcrRegion], extra: Sequence[OcrRegion]) -> list[OcrRegion]:
    """Add regions from one turned pass, without a second copy of the same spot.

    A new region that lands on one existing region replaces it only when that reading is
    weak and the new one is clearly more confident. A clear upright word, and a weak but
    still plausible one, therefore stay. A new region that covers more than one existing
    region is left out: it is a line of several words, and replacing them would drop the
    upright readings.
    """
    merged = list(regions)
    for region in extra:
        overlaps = [
            index
            for index, kept in enumerate(merged)
            if _overlap_fraction(region, kept) >= OVERLAP_FRACTION
        ]
        if not overlaps:
            merged.append(region)
        elif len(overlaps) == 1 and _prefer_new(merged[overlaps[0]], region):
            merged[overlaps[0]] = region
    return merged


def _prefer_new(current: OcrRegion, new: OcrRegion) -> bool:
    new_confidence = 0.0 if new.confidence is None else new.confidence
    current_confidence = 0.0 if current.confidence is None else current.confidence
    return (
        current_confidence < WEAK_CONFIDENCE
        and new_confidence >= current_confidence + REPLACEMENT_MARGIN
    )


def _overlap_fraction(region: OcrRegion, other: OcrRegion) -> float:
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


def factory(
    *,
    languages: Sequence[str],
    options: Mapping[str, Any],
) -> EasyOcrBackend:
    return EasyOcrBackend(languages=languages, options=options)
