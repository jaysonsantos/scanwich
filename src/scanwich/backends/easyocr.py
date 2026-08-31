from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scanwich.models import OcrRegion, Point

logger = logging.getLogger(__name__)


class EasyOcrBackend:
    """EasyOCR provider, imported lazily so other providers need not install it."""

    def __init__(self, *, languages: Sequence[str], options: Mapping[str, Any]) -> None:
        self._languages = list(languages)
        reader_options = dict(options)
        self._readtext_options = reader_options.pop("readtext", {})
        if not isinstance(self._readtext_options, dict):
            raise TypeError("EasyOCR's 'readtext' backend option must be a JSON object")
        reader_options.setdefault("gpu", False)
        self._reader_options = reader_options
        self._reader: Any | None = None

    def recognize(self, image_path: Path) -> list[OcrRegion]:
        reader = self._get_reader()
        raw_regions = reader.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
            **self._readtext_options,
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


def factory(
    *,
    languages: Sequence[str],
    options: Mapping[str, Any],
) -> EasyOcrBackend:
    return EasyOcrBackend(languages=languages, options=options)
