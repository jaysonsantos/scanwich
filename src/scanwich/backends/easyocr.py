from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from scanwich.models import OcrRegion, Point
from scanwich.rotated_text import (
    DEFAULT_ROTATIONS,
    merge_regions,
    recognize_turned,
    rotations_from_option,
)

logger = logging.getLogger(__name__)

# On a turned page the glyphs may be upside down. EasyOCR then recognizes each crop both
# ways and keeps the reading with the higher confidence. The box itself does not move.
UPSIDE_DOWN = (180,)
# EasyOCR otherwise shrinks the long side to this size. Nearby vertical labels then become
# one word. A turned pass keeps the page's own long side when it is larger than this.
DEFAULT_CANVAS_SIZE = 2560
# The detection image is not larger than this. A bigger page is still shrunk, but less
# than the default, and the cost of a pass stays bounded.
ROTATED_CANVAS_LIMIT = 4096
# A large page is read as this many tiles on a side, so a small label is not lost in the
# shrink EasyOCR applies to a whole page. `tiles=1` reads the page in one piece.
DEFAULT_TILES = 1
# Tiles overlap by this share of a tile, so a label on a boundary is still whole in one tile.
TILE_OVERLAP = 0.15
MAX_TILES = 6


class EasyOcrBackend:
    """EasyOCR provider, imported lazily so other providers need not install it."""

    def __init__(self, *, languages: Sequence[str], options: Mapping[str, Any]) -> None:
        self._languages = list(languages)
        reader_options = dict(options)
        self._readtext_options = reader_options.pop("readtext", {})
        if not isinstance(self._readtext_options, dict):
            raise TypeError("EasyOCR's 'readtext' backend option must be a JSON object")
        self._rotations = rotations_from_option(reader_options.pop("rotations", DEFAULT_ROTATIONS))
        self._tiles = _tiles_from_option(reader_options.pop("tiles", DEFAULT_TILES))
        self._refine = _bool_from_option(reader_options.pop("refine", True))
        reader_options.setdefault("gpu", False)
        self._reader_options = reader_options
        self._reader: Any | None = None

    def recognize(self, image_path: Path) -> list[OcrRegion]:
        return recognize_turned(
            image_path,
            self._rotations,
            self._recognize_file,
            reader_name="EasyOCR",
            log=logger,
        )

    def _recognize_file(self, image_path: Path, degrees: int) -> list[OcrRegion]:
        options = dict(self._readtext_options)
        if degrees:
            options.setdefault("rotation_info", list(UPSIDE_DOWN))
            # Full resolution keeps a gap between the vertical labels the turn puts in a
            # row. The upright pass keeps EasyOCR's own detection size: raising it on
            # this page rewrote correct horizontal labels as single-character mistakes.
            with Image.open(image_path) as image:
                long_side = max(image.size)
            options.setdefault(
                "canvas_size",
                min(max(long_side, DEFAULT_CANVAS_SIZE), ROTATED_CANVAS_LIMIT),
            )
        if self._tiles <= 1:
            regions = self._read(str(image_path), options)
        else:
            regions = self._read_tiles(image_path, options)
        # A second upright read at the page's own size finds small horizontal labels,
        # such as LSY, that the 2560-pixel read misses or reads too weakly. It is merged
        # in afterwards, so a confident word from the first read is not replaced by a
        # less sure reading of the same spot.
        if (
            degrees == 0
            and self._refine
            and self._tiles <= 1
            and "canvas_size" not in self._readtext_options
        ):
            with Image.open(image_path) as image:
                long_side = max(image.size)
            if long_side > DEFAULT_CANVAS_SIZE:
                finer = dict(options)
                finer["canvas_size"] = min(long_side, ROTATED_CANVAS_LIMIT)
                logger.info(
                    "Reading the upright page again at %d pixels on the long side",
                    finer["canvas_size"],
                )
                regions = merge_regions(regions, self._read(str(image_path), finer))
        return regions

    def _read_tiles(self, image_path: Path, options: Mapping[str, Any]) -> list[OcrRegion]:
        """Read overlapping tiles and return the regions in the whole image's coordinates."""
        with Image.open(image_path) as image:
            image.load()
            boxes = _tile_boxes(image.size, self._tiles)
            logger.info(
                "Reading this image as %d overlapping tile(s) so small labels stay large enough",
                len(boxes),
            )
            regions: list[OcrRegion] = []
            with TemporaryDirectory(prefix="scanwich-tile-") as temporary_directory:
                directory = Path(temporary_directory)
                for index, (left, top, right, bottom) in enumerate(boxes):
                    tile_path = directory / f"tile-{index}.png"
                    image.crop((left, top, right, bottom)).save(
                        tile_path, format="PNG", compress_level=1
                    )
                    for region in self._read(str(tile_path), options):
                        # A label found in two tiles is the same spot. The second copy is dropped.
                        regions = merge_regions(regions, [_shift_region(region, left, top)])
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


def _bool_from_option(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("EasyOCR's 'refine' backend option must be true or false")
    return value


def _tiles_from_option(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("EasyOCR's 'tiles' backend option must be an integer")
    if value < 1 or value > MAX_TILES:
        raise ValueError(f"EasyOCR tiles must be from 1 to {MAX_TILES}")
    return value


def _tile_boxes(size: tuple[int, int], tiles: int) -> list[tuple[int, int, int, int]]:
    """Return overlapping crops that cover an image, `tiles` along each side."""
    width, height = size
    if tiles <= 1:
        return [(0, 0, width, height)]
    tile_width = min(width, math.ceil(width / tiles * (1 + TILE_OVERLAP)))
    tile_height = min(height, math.ceil(height / tiles * (1 + TILE_OVERLAP)))

    def origins(length: int, tile: int) -> list[int]:
        if tile >= length:
            return [0]
        return [round(index * (length - tile) / (tiles - 1)) for index in range(tiles)]

    return [
        (x, y, min(width, x + tile_width), min(height, y + tile_height))
        for y in origins(height, tile_height)
        for x in origins(width, tile_width)
    ]


def _shift_region(region: OcrRegion, dx: int, dy: int) -> OcrRegion:
    return OcrRegion(
        text=region.text,
        polygon=tuple(Point(point.x + dx, point.y + dy) for point in region.polygon),  # type: ignore[arg-type]
        confidence=region.confidence,
    )


def factory(
    *,
    languages: Sequence[str],
    options: Mapping[str, Any],
) -> EasyOcrBackend:
    return EasyOcrBackend(languages=languages, options=options)
