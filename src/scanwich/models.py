"""Backend-neutral OCR result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Point:
    """A pixel coordinate measured from the image's top-left corner."""

    x: float
    y: float

    def to_json(self) -> list[float]:
        return [self.x, self.y]


@dataclass(frozen=True)
class OcrRegion:
    """Text recognized inside a clockwise quadrilateral, starting at top-left."""

    text: str
    polygon: tuple[Point, Point, Point, Point]
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("OCR region text must not be empty")
        if len(self.polygon) != 4:
            raise ValueError("OCR region polygon must contain exactly four points")

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "text": self.text,
            "polygon": [point.to_json() for point in self.polygon],
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result
