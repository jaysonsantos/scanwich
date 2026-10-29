from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from scanwich.models import OcrRegion, Point
from scanwich.rotated_text import DEFAULT_ROTATIONS, recognize_turned, rotations_from_option

logger = logging.getLogger(__name__)

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
        self._rotations = rotations_from_option(reader_options.pop("rotations", DEFAULT_ROTATIONS))
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
            # Full resolution keeps a gap between labels that sit in a row once the page
            # is turned. The long side is the same after a quarter turn. The upright pass
            # keeps EasyOCR's own detection size.
            with Image.open(image_path) as image:
                long_side = max(image.size)
            options.setdefault(
                "canvas_size",
                min(max(long_side, DEFAULT_CANVAS_SIZE), ROTATED_CANVAS_LIMIT),
            )
        return self._read(str(image_path), options)

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


def factory(
    *,
    languages: Sequence[str],
    options: Mapping[str, Any],
) -> EasyOcrBackend:
    return EasyOcrBackend(languages=languages, options=options)
