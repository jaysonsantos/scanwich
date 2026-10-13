from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from scanwich.layout import Box, FigureLocation, PageLayout
from scanwich.models import OcrRegion

ENTRY_POINT_GROUP = "scanwich.ocr_backends"


@runtime_checkable
class OcrBackend(Protocol):
    """Contract implemented by OCR providers."""

    def recognize(self, image_path: Path) -> Sequence[OcrRegion]:
        """Recognize text in one page image."""


@runtime_checkable
class LayoutBackend(Protocol):
    """Contract of an OCR provider that also returns the structure of a page.

    A provider implements it in addition to `OcrBackend`. A document writer, such as the
    LaTeX writer, needs it.
    """

    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        """Recognize the blocks of one page image in reading order.

        `upright` is True when the caller knows that the page is upright in the image. The
        provider then does not look for the direction of the page: it reads the page.
        Without it, a provider that gets a page that is not upright can return the side
        of the image where the top of the page is, and no blocks.
        """


@runtime_checkable
class FigureLocator(Protocol):
    """Contract of a provider that finds the exact box of one illustration.

    A provider implements it in addition to `LayoutBackend`. The box of an illustration in
    the layout of a full page is often not exact. With this contract, a document writer
    asks again with the part of the page image that has the illustration.
    """

    def locate_figure(
        self, image_path: Path, *, estimate: Box | None = None, caption: str | None = None
    ) -> FigureLocation | None:
        """Find one illustration in an image that shows a part of a page.

        `estimate` is a first box of the illustration, which can be wrong at each side.
        `caption` is the caption that the page gives to the illustration. The two values
        tell the provider which illustration to find when the image shows more than one.
        The boxes use the 0-to-1000 grid of the image. The result is None when the image
        shows no illustration.
        """


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
