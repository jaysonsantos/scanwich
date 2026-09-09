from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import EntryPoint, entry_points
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
    """Callable that builds a backend.

    A factory may expose ``request_options``: the option names that callers, such as the
    HTTP API, are allowed to set for a single request. Factories without that attribute
    accept configuration options only.
    """

    request_options: Iterable[str]

    def __call__(
        self,
        *,
        languages: Sequence[str],
        options: Mapping[str, Any],
    ) -> OcrBackend: ...


def available_backends() -> list[str]:
    return sorted({entry_point.name for entry_point in entry_points(group=ENTRY_POINT_GROUP)})


def _find_entry_point(name: str) -> EntryPoint:
    matches = list(entry_points(group=ENTRY_POINT_GROUP, name=name))
    if not matches:
        installed = ", ".join(available_backends()) or "none"
        raise ValueError(f"unknown OCR backend {name!r}; installed backends: {installed}")
    if len(matches) > 1:
        providers = ", ".join(entry_point.value for entry_point in matches)
        raise ValueError(f"multiple OCR backends are registered as {name!r}: {providers}")
    return matches[0]


def backend_request_options(name: str) -> frozenset[str]:
    """Return the option names that a backend accepts from one request.

    Backends opt in through a ``request_options`` collection on their factory. Backends
    without that attribute accept no per-request options.
    """
    factory = _find_entry_point(name).load()
    declared = getattr(factory, "request_options", ())
    if isinstance(declared, str) or not isinstance(declared, Iterable):
        raise TypeError(f"OCR backend {name!r} must declare request_options as a collection")
    options = frozenset(declared)
    if not all(isinstance(option, str) for option in options):
        raise TypeError(f"OCR backend {name!r} must declare request_options as option names")
    return options


def load_backend(
    name: str,
    *,
    languages: Sequence[str],
    options: Mapping[str, Any] | None = None,
) -> OcrBackend:
    factory: OcrBackendFactory = _find_entry_point(name).load()
    backend = factory(languages=tuple(languages), options=dict(options or {}))
    if not isinstance(backend, OcrBackend):
        raise TypeError(f"OCR backend {name!r} does not implement recognize(image_path)")
    return backend
