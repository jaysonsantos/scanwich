from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from scanwich.models import OcrRegion

ENTRY_POINT_GROUP = "scanwich.ocr_backends"


@runtime_checkable
class OcrBackend(Protocol):
    """Contract implemented by OCR providers."""

    def recognize(self, image_path: Path) -> Sequence[OcrRegion]:
        """Recognize text in one page image."""


class OcrBackendFactory(Protocol):
    def __call__(
        self,
        *,
        languages: Sequence[str],
        options: Mapping[str, Any],
    ) -> OcrBackend: ...


def available_backends() -> list[str]:
    return sorted({entry_point.name for entry_point in entry_points(group=ENTRY_POINT_GROUP)})


def load_backend(
    name: str,
    *,
    languages: Sequence[str],
    options: Mapping[str, Any] | None = None,
) -> OcrBackend:
    matches = list(entry_points(group=ENTRY_POINT_GROUP, name=name))
    if not matches:
        installed = ", ".join(available_backends()) or "none"
        raise ValueError(f"unknown OCR backend {name!r}; installed backends: {installed}")
    if len(matches) > 1:
        providers = ", ".join(entry_point.value for entry_point in matches)
        raise ValueError(f"multiple OCR backends are registered as {name!r}: {providers}")

    factory: OcrBackendFactory = matches[0].load()
    backend = factory(languages=tuple(languages), options=dict(options or {}))
    if not isinstance(backend, OcrBackend):
        raise TypeError(f"OCR backend {name!r} does not implement recognize(image_path)")
    return backend
