from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import logging
import math
import os
import re
import shutil
import signal
import uuid
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

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
    FigureLocation,
    LayoutFormatError,
    PageLayout,
    page_layout_from_json,
)
from scanwich.ocr import (
    AsyncFigureLocator,
    AsyncLayoutBackend,
    FigureLocator,
    LayoutBackend,
    LayoutSettings,
    NoReplyError,
    OcrBackend,
    RateLimitError,
)
from scanwich.pdf import (
    PNG_COMPRESSION_LEVEL,
    PdfPipelineError,
    assemble_searchable_pdf,
    count_pdf_pages,
    rasterize_pdf,
    rasterize_pdf_page,
)

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
# LaTeX output reads this many pages at the same time, when the caller gives no number.
# A request for a photo of a book page needs 30 to 40 seconds, so a book needs hours with
# one page at a time. The number is small: a larger number needs more memory for page
# images, and the risk that the service limits the rate of the requests is higher.
DEFAULT_PARALLEL_PAGES = 4
# One task makes the page images, and the pages wait for a reader in a queue of this
# size. That task waits when the queue is full. Thus the page images that exist at one
# time are not more than: one for each reader, the pages in the queue, and the one page
# that waits for a place in the queue.
PAGE_QUEUE_SIZE = 1
# When the pages in work get no reply from the service this many times in sequence, the
# cause is probably not a page: it can be the service, the key, the quota, or an option.
# The conversion then starts no more pages. One time is one wave of failed pages: the
# pages in work at one moment fail together. Thus a page counts only when it started
# after the last page that counted. A page that gets a layout from its requests starts
# the count again. A page with a reply that is not usable, or with an answer of the
# service about its content, does not count: the service works.
NO_REPLY_WAVES = 3
# When the first this many pages of a conversion that get a reply of the service fail,
# and no page has a layout, the conversion starts no more pages. A page with no reply is
# for the rule of the waves, because its cause is probably the service. A service that
# gives only replies that Scanwich cannot use, for example after a wrong model option,
# then has a cost for 10 pages and not for a book. 10 difficult pages in sequence at the
# start of a book are improbable. A layout directory with a saved layout turns the rule
# off, so that a book whose first pages are difficult does not stop at each run.
FIRST_FAILED_PAGES = 10
# The reasons of a stop of the conversion before its last page.
STOP_NO_REPLY = "no reply"
STOP_FIRST_PAGES = "first pages"
# The waits in seconds after the service answers that it gets too many requests. All
# requests of the conversion wait. After the last wait, the request fails.
RATE_LIMIT_WAITS = (30.0, 60.0, 120.0)
# A service can ask for a longer wait. This is the maximum.
MAX_RATE_LIMIT_WAIT = 300.0
# `--ocr-output-dir` has one file for the layout of each page. The value of this field
# of the file is the record that ties the layout to a page of an input file.
LAYOUT_FILE_SUFFIX = ".json"
SOURCE_FIELD = "source"
SOURCE_RECORD_VERSION = 1
# The name of a layout file has the name of the input file, not longer than this, and
# this many characters of the SHA-256 of the input file.
MAX_LAYOUT_NAME_BYTES = 150
LAYOUT_NAME_HASH_CHARACTERS = 8
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
# A file that is not complete has this suffix until a rename gives it its name.
TEMPORARY_SUFFIX = ".scanwich-tmp"
# The page that the current task or thread works for, as text for a log record. The
# pages of a conversion are in work at the same time, and a backend does not know the
# page of its request.
LOG_PAGE: contextvars.ContextVar[str | None] = contextvars.ContextVar("scanwich_page", default=None)

logger = logging.getLogger(__name__)

Result = TypeVar("Result")


class PageLayoutError(RuntimeError):
    """A page, or the full document, has no usable layout."""


class OutputDirectoryError(RuntimeError):
    """Scanwich cannot write to a directory that the conversion needs."""


class ConversionStopped(KeyboardInterrupt):
    """A signal stopped a conversion to LaTeX. `signal_number` is the signal.

    The conversion ended in a clean way: no task, no thread, and no work file stays, and
    each layout that was saved stays.

    `document_complete` is True for a signal in the install of the result. The install
    then continued to its end, so the new document and its illustrations are complete.
    When it is False, the earlier document and its illustrations did not change.
    """

    def __init__(self, signal_number: int, *, document_complete: bool = False) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number
        self.document_complete = document_complete


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
    inputs: Path | Sequence[Path],
    output_tex: Path,
    *,
    backend: LayoutBackend,
    dpi: int | None = None,
    rotate: int | None = None,
    flow: bool = False,
    ocr_output_directory: Path | None = None,
    ignore_saved_layouts: bool = False,
    parallel_pages: int = DEFAULT_PARALLEL_PAGES,
    stop_signals: Collection[int] = (),
) -> None:
    """Write PDFs and photos of pages as one LaTeX document.

    `inputs` is one file or a sequence of files. The pages of the document have the
    sequence of the files, and a PDF gives its pages at its place. The illustrations go
    to a directory next to the document. `rotate` is the direction of the pages: the
    function turns each page image clockwise by that angle before recognition. Without
    it, the backend finds the direction of each page. Each source page is one page of the
    document. With `flow`, the text flows across the pages of the document.

    With `ocr_output_directory`, the function saves the layout of each page there, and a
    page that has a saved layout there needs no request. `ignore_saved_layouts` gives a
    new layout to each page. The function reads `parallel_pages` pages at the same time.

    The function raises `PageLayoutError` when a page has no usable layout, and when no
    page has content. It then writes no document.

    `stop_signals` are signal numbers, for example `signal.SIGTERM`. Such a signal stops
    the conversion as Ctrl-C does, and the function raises `ConversionStopped`. This
    needs the main thread. In a different thread, a signal keeps its usual effect. A
    stop signal or Ctrl-C in the install of the result does not stop the install: the
    function raises `ConversionStopped` after it, with `document_complete` True.

    The function starts an event loop of its own. A thread with a running event loop
    cannot call it: use a different thread, for example with `asyncio.to_thread`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "convert_to_latex starts an event loop of its own, so a thread with a running "
            "event loop cannot call it; call it from a different thread, for example with "
            "asyncio.to_thread"
        )
    given_paths = [inputs] if isinstance(inputs, (str, os.PathLike)) else list(inputs)
    if not given_paths:
        raise ValueError("LaTeX output needs one input file or more")
    given_paths = [Path(path) for path in given_paths]
    input_paths = [path.expanduser().resolve() for path in given_paths]
    output_tex = output_tex.expanduser().resolve()
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(f"input file does not exist: {input_path}")
        if input_path == output_tex:
            raise ValueError("input and output paths must be different")
    if dpi is not None and dpi <= 0:
        raise ValueError("DPI must be greater than zero")
    if rotate is not None and rotate not in PAGE_ROTATIONS:
        raise ValueError("a page image turns by 0, 90, 180, or 270 degrees")
    if type(parallel_pages) is not int or parallel_pages < 1:
        raise ValueError("the number of parallel pages must be 1 or more")
    if ignore_saved_layouts and ocr_output_directory is None:
        raise ValueError("without a directory for the layouts, there is no saved layout to ignore")
    if not isinstance(backend, LayoutBackend):
        raise TypeError(
            f"{type(backend).__name__} does not implement recognize_layout(image_path); "
            "LaTeX output needs an OCR backend that recognizes the layout of a page"
        )

    with _StopSignals(stop_signals) as stop:
        _convert_with_stop_signals(
            given_paths,
            input_paths,
            output_tex,
            backend=backend,
            dpi=dpi,
            rotate=rotate,
            flow=flow,
            ocr_output_directory=ocr_output_directory,
            ignore_saved_layouts=ignore_saved_layouts,
            parallel_pages=parallel_pages,
            stop=stop,
        )
    if stop.signal_number is not None:
        # Each other signal raised the stop before the install. The result is complete.
        raise ConversionStopped(stop.signal_number, document_complete=True)


def _convert_with_stop_signals(
    given_paths: Sequence[Path],
    input_paths: Sequence[Path],
    output_tex: Path,
    *,
    backend: LayoutBackend,
    dpi: int | None,
    rotate: int | None,
    flow: bool,
    ocr_output_directory: Path | None,
    ignore_saved_layouts: bool,
    parallel_pages: int,
    stop: _StopSignals,
) -> None:
    """Do the work of `convert_to_latex`, which has checked the arguments.

    The stop signals have their handlers for the full time of this function: each part
    has work files. The work files go away before the handlers of the caller come back.
    From the start of the install of the result, a signal only records the stop.
    """
    # A directory that Scanwich cannot write to must stop the conversion here, before
    # the first request. The conversion checks the layout directory before its first
    # request: a read-only layout directory with a layout for each page needs no write.
    _check_directory(output_tex.parent, "the directory of the document")
    layout_directory = None
    if ocr_output_directory is not None:
        layout_directory = ocr_output_directory.expanduser().resolve()
    figure_directory = _figure_directory(output_tex)
    pages = _source_pages(given_paths, input_paths)
    logger.info("The document has %d page(s) from %d input file(s)", len(pages), len(input_paths))
    _warn_about_the_sequence_of_names(given_paths)
    if len(pages) > 1 and layout_directory is None:
        # The user must know this before the first request, not after a failure.
        logger.warning(
            "The conversion has %d pages and saves no layouts: a failure at one page loses "
            "the layouts of the pages before it. With --ocr-output-dir DIRECTORY, a new run "
            "sends requests only for the pages that have no saved layout",
            len(pages),
        )
    with (
        TemporaryDirectory(prefix="scanwich-") as temporary_directory,
        # The illustrations of a book are large in sum, and they stay until the document
        # is complete. The temporary directory of a system can be in memory, so the
        # illustrations wait next to the document.
        TemporaryDirectory(
            prefix=f".{output_tex.name}.", suffix=TEMPORARY_SUFFIX, dir=output_tex.parent
        ) as new_figures_directory,
    ):
        work_directory = Path(temporary_directory)
        conversion = _Conversion(
            backend=backend,
            pages=pages,
            work_directory=work_directory,
            new_figures_directory=Path(new_figures_directory),
            figure_directory_name=figure_directory.name,
            dpi=dpi,
            rotate=rotate,
            layout_directory=layout_directory,
            ignore_saved_layouts=ignore_saved_layouts,
            parallel_pages=parallel_pages,
            stop=stop,
        )
        try:
            asyncio.run(conversion.run())
        except asyncio.CancelledError:
            if stop.signal_number is None:
                raise
            raise ConversionStopped(stop.signal_number) from None
        finally:
            stop.detach()
        # A signal can come after the last page and before the end of the event loop.
        stop.raise_if_stopped()
        layouts = conversion.layouts_in_sequence()

        if not any(layout.blocks for layout in layouts):
            source = (
                str(input_paths[0])
                if len(input_paths) == 1
                else f"the {len(input_paths)} input files"
            )
            raise PageLayoutError(
                f"no page of {source} has content in its layout, so there is no document"
            )
        logger.info("Writing LaTeX document")
        # The file is in the directory of the document, so the last step only renames it.
        temporary_tex = output_tex.with_name(f".{output_tex.name}{TEMPORARY_SUFFIX}")
        try:
            temporary_tex.write_text(render_latex(layouts, flow=flow), encoding="utf-8")
            # A stop between two renames of the install can leave the new document with the
            # earlier illustrations, or lose an earlier illustration. The install and the
            # removal of the work files need a short time, so a stop waits for them.
            stop.defer()
            figure_count = _install_result(
                temporary_tex, output_tex, Path(new_figures_directory), figure_directory
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


class _StopSignals:
    """The handlers of the stop signals, for the time that a conversion has work files.

    In the event loop, a stop signal cancels the conversion, as Ctrl-C does in
    `asyncio.run`. Before and after the event loop, it raises `ConversionStopped` at once.
    The handlers stay until the work files are gone, and the handlers of the caller then
    come back.

    - From `defer()` to the end, a stop signal and Ctrl-C only record the stop.
    - A second signal has no effect while the conversion stops. A terminal that closes
      sends SIGHUP two times, in less than 10 ms.
    - A signal that the caller ignores at the start gets no handler, so that it keeps no
      effect. `nohup` and a parent of the process give `SIG_IGN` to SIGHUP in this way.
    - A thread that is not the main thread cannot set signal handlers. Each signal then
      keeps its usual effect.
    """

    def __init__(self, signal_numbers: Collection[int]) -> None:
        self._signal_numbers = tuple(signal_numbers)
        self._handlers_of_the_caller: dict[int, Any] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[Any] | None = None
        self._deferred = False
        # The signal that stopped the conversion.
        self.signal_number: int | None = None

    def __enter__(self) -> _StopSignals:
        for number in self._signal_numbers:
            handler = signal.getsignal(number)
            if handler == signal.SIG_IGN:
                continue
            try:
                signal.signal(number, self._handle)
            except ValueError:
                continue
            self._handlers_of_the_caller[number] = handler
        return self

    def __exit__(self, *details: object) -> None:
        for number, handler in self._handlers_of_the_caller.items():
            # A handler that the caller did not set in Python is not known.
            with contextlib.suppress(ValueError, TypeError, OSError):
                signal.signal(number, signal.SIG_DFL if handler is None else handler)

    def attach(self, task: asyncio.Task[Any]) -> None:
        """Make a stop signal cancel this task of the running event loop."""
        self._loop = asyncio.get_running_loop()
        self._task = task

    def detach(self) -> None:
        self._loop = self._task = None

    def raise_if_stopped(self) -> None:
        if self.signal_number is not None:
            raise ConversionStopped(self.signal_number)

    def defer(self) -> None:
        """From now on, a stop signal and Ctrl-C only record the stop."""
        self._deferred = True
        # Ctrl-C gets the same handler when it has the usual handler of Python, which
        # raises `KeyboardInterrupt`. A different handler of the caller stays.
        if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
            return
        try:
            signal.signal(signal.SIGINT, self._handle)
        except ValueError:
            return
        self._handlers_of_the_caller[signal.SIGINT] = signal.default_int_handler

    def _handle(self, number: int, frame: object) -> None:
        if self.signal_number is not None:
            return
        self.signal_number = number
        if self._deferred:
            return
        if self._loop is None or self._task is None:
            raise ConversionStopped(number)
        # The handler runs in the thread of the event loop, between two steps of a task.
        # The cancel waits for the next step of the loop.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._task.cancel)


def _check_directory(directory: Path, name: str) -> None:
    """Make a directory, and raise `OutputDirectoryError` when a file cannot go there."""
    probe = directory / f".{uuid.uuid4().hex}{TEMPORARY_SUFFIX}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        raise OutputDirectoryError(
            f"Scanwich cannot write to {name} {directory}: {error}"
        ) from error


@dataclass(frozen=True)
class _SourcePage:
    """One page of the document, and the input file that has it."""

    # The number of the page in the document. The first page is 1.
    number: int
    # The input file, and the same file as the caller gave it. A message has the second.
    path: Path
    given: Path
    # The number of the page in the input file. An image file has one page.
    page_of_input: int
    from_pdf: bool

    @property
    def label(self) -> str:
        if self.from_pdf:
            return f"{self.given}, page {self.page_of_input}"
        return str(self.given)


@dataclass(frozen=True)
class _SourceRecord:
    """The record that ties a saved layout to a page of an input file."""

    # The SHA-256 of the input file, and the number of the page in that file.
    digest: str
    page: int
    # The clockwise angle that the page image turned by before the backend read it.
    turn: int
    # The backend and the settings that gave the layout.
    reading: Mapping[str, Any]


@dataclass(frozen=True)
class _SavedLayout:
    path: Path
    layout: PageLayout
    record: _SourceRecord


@dataclass(frozen=True)
class _PageJob:
    """One page in the queue between the task that makes page images and the readers."""

    page: _SourcePage
    # The SHA-256 of the input file. It is None when the conversion saves no layouts.
    digest: str | None
    saved: _SavedLayout | None
    # The image file of the page in the work directory. A page with a saved layout and
    # no illustration needs none.
    image_path: Path | None


def _source_pages(given_paths: Sequence[Path], input_paths: Sequence[Path]) -> list[_SourcePage]:
    """Return the pages of the document in the sequence of the input files."""
    pages: list[_SourcePage] = []
    for given, input_path in zip(given_paths, input_paths, strict=True):
        from_pdf = is_pdf(input_path)
        count = count_pdf_pages(input_path) if from_pdf else 1
        if count < 1:
            raise PdfPipelineError(f"PDFium found no pages in {input_path}")
        pages_before = len(pages)
        pages.extend(
            _SourcePage(
                number=pages_before + page_of_input,
                path=input_path,
                given=given,
                page_of_input=page_of_input,
                from_pdf=from_pdf,
            )
            for page_of_input in range(1, count + 1)
        )
    return pages


def _warn_about_the_sequence_of_names(given_paths: Sequence[Path]) -> None:
    """Warn when the sequence of the input files is probably not that of the pages.

    A shell sorts the names of a pattern by their characters. `page10.jpg` is then
    before `page2.jpg`.
    """
    names = [str(path) for path in given_paths]
    by_numbers = sorted(names, key=_name_with_numbers)
    if names != sorted(names) or names == by_numbers:
        return
    first = next(index for index, name in enumerate(names) if name != by_numbers[index])
    logger.warning(
        "The input files are in the sequence of their characters, and the numbers in their "
        "names give a different sequence: %s is before %s. The pages of the document have "
        "the sequence of the command. Compare it with the source",
        names[first],
        by_numbers[first],
    )


def _name_with_numbers(name: str) -> list[tuple[int, int | str]]:
    """Return a sort key that compares the numbers in a name by their value."""
    # Only the digits 0 to 9 make a number here. `²` is a digit for `str.isdigit`, and
    # it is not a number for `int`.
    parts = re.split(r"([0-9]+)", name)
    return [(1, int(part)) if index % 2 else (0, part) for index, part in enumerate(parts)]


@contextlib.contextmanager
def _log_page(page: _SourcePage, page_count: int) -> Iterator[None]:
    """Name the page in `LOG_PAGE` for the log records of the work in this block."""
    token = LOG_PAGE.set(f"page {page.number}/{page_count}")
    try:
        yield
    finally:
        LOG_PAGE.reset(token)


class _BackendCalls:
    """The requests of one conversion to its backend, from the tasks of an event loop.

    A backend with the async methods gets its calls at the same time. A backend with
    only the synchronous methods gets one call at a time, and all its calls come from
    one thread: the pipeline does not know that such a backend permits calls from more
    than one thread. That thread is not the thread that made the backend.

    When the service answers that it gets too many requests, all the requests of the
    conversion wait, and the request then goes again.
    """

    def __init__(self, backend: LayoutBackend) -> None:
        self._backend = backend
        # The one thread for the calls of a backend with only synchronous methods.
        self._thread: ThreadPoolExecutor | None = None
        # The time of the event loop before which no request starts.
        self._resume_at = 0.0
        self.locates_figures = isinstance(backend, FigureLocator)

    async def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        backend = self._backend
        if isinstance(backend, AsyncLayoutBackend):
            return await self._send(
                lambda: backend.recognize_layout_async(image_path, upright=upright)
            )
        return await self._send(
            lambda: self._in_thread(backend.recognize_layout, image_path, upright=upright)
        )

    async def locate_figure(
        self, image_path: Path, *, estimate: Box | None, caption: str | None
    ) -> FigureLocation | None:
        backend = self._backend
        if isinstance(backend, AsyncFigureLocator):
            return await self._send(
                lambda: backend.locate_figure_async(image_path, estimate=estimate, caption=caption)
            )
        assert isinstance(backend, FigureLocator)
        return await self._send(
            lambda: self._in_thread(
                backend.locate_figure, image_path, estimate=estimate, caption=caption
            )
        )

    async def close(self) -> None:
        """Let a backend with the async methods close its connections.

        An error at this point is not an error of a page, so it is only a warning. It
        must not replace the error of a page, and it must not stop a complete document.
        """
        close = getattr(self._backend, "aclose", None)
        if isinstance(self._backend, AsyncLayoutBackend) and callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                logger.warning(
                    "The backend did not close its connections (%s: %s)",
                    type(error).__name__,
                    error,
                )
        if self._thread is not None:
            # A call that waits for the thread does not start. The thread ends after the
            # call that is in work.
            thread, self._thread = self._thread, None
            await asyncio.to_thread(thread.shutdown, wait=True, cancel_futures=True)

    async def _in_thread(self, function: Callable[..., Result], *args: Any, **options: Any) -> Any:
        if self._thread is None:
            self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scanwich-backend")
        # The thread gets the context of the task, which has the page for the log records.
        call = functools.partial(contextvars.copy_context().run, function, *args, **options)
        return await asyncio.get_running_loop().run_in_executor(self._thread, call)

    async def _send(self, call: Callable[[], Awaitable[Result]]) -> Result:
        waits = iter(RATE_LIMIT_WAITS)
        loop = asyncio.get_running_loop()
        while True:
            # A wait of one request can become longer while this request waits.
            while (delay := self._resume_at - loop.time()) > 0:
                await asyncio.sleep(delay)
            try:
                return await call()
            except RateLimitError as error:
                wait = next(waits, None)
                if wait is None:
                    raise
                wait = _rate_limit_wait(wait, error.retry_after)
                self._resume_at = max(self._resume_at, loop.time() + wait)
                logger.warning(
                    "The service gets too many requests. All pages wait %.0f seconds before "
                    "their next request, and then this request goes again",
                    wait,
                )
                if error.resume is not None:
                    # The call had replies before the refusal. The provider continues it,
                    # so that those replies count for its limit of requests.
                    call = error.resume


def _rate_limit_wait(wait: float, retry_after: float | None) -> float:
    """Return the wait in seconds after the service refused a request for the rate.

    `wait` is the wait of the conversion, and `retry_after` is the wait that the service
    asks for. The longer one is in use, to the maximum.
    """
    if retry_after is not None:
        wait = max(wait, retry_after)
    return min(wait, MAX_RATE_LIMIT_WAIT)


class _Conversion:
    """The work for the pages of one conversion to LaTeX.

    One task makes the page images and puts each page in a queue. Reader tasks take the
    pages from the queue, get their layouts, and crop their illustrations. The queue has
    a maximum size, so the first task waits when the readers are not ready. The result
    does not depend on the sequence of the replies: each layout goes to the number of
    its page.
    """

    def __init__(
        self,
        *,
        backend: LayoutBackend,
        pages: Sequence[_SourcePage],
        work_directory: Path,
        new_figures_directory: Path,
        figure_directory_name: str,
        dpi: int | None,
        rotate: int | None,
        layout_directory: Path | None,
        ignore_saved_layouts: bool,
        parallel_pages: int,
        stop: _StopSignals | None = None,
    ) -> None:
        self._backend = backend
        self._stop_signals = stop
        self._pages = pages
        self._work_directory = work_directory
        self._new_figures_directory = new_figures_directory
        self._figure_directory_name = figure_directory_name
        self._dpi = dpi
        self._rotate = rotate
        self._layout_directory = layout_directory
        self._ignore_saved_layouts = ignore_saved_layouts
        self._parallel_pages = parallel_pages
        settings = backend.layout_settings() if isinstance(backend, LayoutSettings) else {}
        # The settings go to a JSON file. This form compares with the form of a file.
        self._settings: dict[str, Any] = json.loads(json.dumps(dict(settings)))
        self._digests: dict[Path, str] = {}
        self._saved_index: dict[tuple[str, int], list[Path]] = {}
        self._layouts: dict[int, PageLayout] = {}
        # The message of each failed page. The error itself does not stay: its traceback
        # has the frames of the backend, and they hold the request with the page image.
        self._failures: dict[int, str] = {}
        # The pages that got their layout from requests, and the pages that used a
        # saved layout.
        self._read_pages = 0
        self._saved_pages = 0
        # The reason of a stop of the conversion before its last page.
        self._stop_reason: str | None = None
        # The pages whose requests failed with a reply of the service, and with no reply.
        self._failed_readings = 0
        self._pages_without_reply = 0
        # The layout directory is checked before the first request of the conversion.
        self._layout_directory_is_checked = False
        # The waves in sequence of pages with no reply from the service, and the time of
        # the event loop when the last page that counted failed.
        self._waves_without_reply = 0
        self._last_counted_no_reply = -math.inf
        # The settings of saved layouts that are not the settings of this conversion.
        self._other_settings: dict[str, int] = {}
        self._other_settings_pages = 0

    async def run(self) -> None:
        """Get the layout of each page. A page that fails is then without a layout."""
        # From here, a stop signal cancels this task. The index below has work files.
        if self._stop_signals is not None:
            conversion_task = asyncio.current_task()
            assert conversion_task is not None
            self._stop_signals.attach(conversion_task)
        # The primitives of asyncio belong to the event loop, so they start here.
        self._stop = asyncio.Event()
        self._layout_directory_check = asyncio.Lock()
        self._calls = _BackendCalls(self._backend)
        if self._layout_directory is not None and not self._ignore_saved_layouts:
            self._saved_index = await asyncio.to_thread(_saved_layout_index, self._layout_directory)
        queue: asyncio.Queue[_PageJob | None] = asyncio.Queue(maxsize=PAGE_QUEUE_SIZE)
        readers = min(self._parallel_pages, len(self._pages))
        logger.info(
            "Reading %d page(s) at the same time; not more than %d page image(s) exist at one time",
            readers,
            readers + PAGE_QUEUE_SIZE + 1,
        )
        tasks = [asyncio.create_task(self._produce(queue, readers), name="scanwich-page-images")]
        tasks += [
            asyncio.create_task(self._read(queue), name=f"scanwich-page-reader-{number}")
            for number in range(1, readers + 1)
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            # An error of one task and a stop of the conversion come here. No task stays.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._calls.close()
        if self._layout_directory is not None:
            logger.info(
                "%d page(s) got a layout from requests, and %d page(s) have a saved layout",
                self._read_pages,
                self._saved_pages,
            )
        if self._other_settings_pages:
            logger.warning(
                "%d saved layout(s) are from other settings than those of this conversion "
                "(%s). The document uses them. To get new layouts, remove their files or use "
                "--ignore-saved-layouts",
                self._other_settings_pages,
                ", ".join(
                    f"{name} of {count} layout(s)"
                    for name, count in sorted(self._other_settings.items())
                ),
            )

    def layouts_in_sequence(self) -> list[PageLayout]:
        """Return the layouts of all pages, or raise `PageLayoutError` when one has none."""
        missing = [page for page in self._pages if page.number not in self._layouts]
        if not missing:
            return [self._layouts[page.number] for page in self._pages]
        lines = [
            f"{len(missing)} of {len(self._pages)} page(s) have no layout, so there is no document:"
        ]
        lines += [
            f"  page {page.number} ({page.label}): {self._failures[page.number]}"
            for page in missing
            if page.number in self._failures
        ]
        not_started = [page.number for page in missing if page.number not in self._failures]
        if not_started:
            lines.append(f"  {len(not_started)} page(s) did not start: {_page_ranges(not_started)}")
        saved = f"The layouts of the pages that are complete are in {self._layout_directory}."
        if self._stop_reason == STOP_FIRST_PAGES:
            lines.append(
                ("The conversion started no more pages: the" if not_started else "The")
                + f" first {FIRST_FAILED_PAGES} pages with a reply of the service failed, and no "
                "page has a layout. The cause is probably the model or an option, for example "
                "a model that does not read images."
            )
            lines.append(
                "Correct the model or the option, and run the same command again. If these "
                "pages are the cause, convert the other files first, with the same layout "
                "directory: a layout directory with a saved layout turns this rule off."
            )
        elif self._stop_reason == STOP_NO_REPLY:
            # All pages can have started before the stop.
            lines.append(
                ("The conversion started no more pages: the" if not_started else "The")
                + f" pages in work got no reply from the service {NO_REPLY_WAVES} times in "
                "sequence. The cause is probably the service, the key, the quota, or an option."
            )
            lines.append(
                f"{saved} When the service works again, the same command sends requests only "
                "for the pages that have no saved layout. If the same pages fail again, they "
                "are the cause: convert the other files first, with the same layout directory."
            )
        elif self._failures and self._pages_without_reply == len(self._failures):
            # Fewer than 3 waves: for example 16 pages in work and 24 pages in all.
            lines.append(
                "No page of this list got a reply from the service. The cause is probably the "
                "service, the key, the quota, or an option."
            )
            lines.append(
                f"{saved} When the service works again, the same command sends requests only "
                "for the pages that have no saved layout."
            )
        else:
            lines.append(
                f"{saved} The same command again sends requests only for the pages of this "
                "list. Read the error of each page first: a page that fails again needs a "
                "correction of its cause."
            )
        raise PageLayoutError("\n".join(lines))

    async def _produce(self, queue: asyncio.Queue[_PageJob | None], readers: int) -> None:
        """Make the page image of each page, and put the page in the queue.

        The task makes one page image at a time, and it waits when the queue is full.
        Thus a page image does not exist long before a reader can take it. PDFium also
        gets one call at a time in this way.
        """
        for page in self._pages:
            if self._stop.is_set():
                break
            with _log_page(page, len(self._pages)):
                try:
                    job = await self._prepare(page)
                except Exception as error:
                    self._page_failed(page, error, read=False)
                    continue
            await queue.put(job)
        # A reader ends when it gets this value.
        for _ in range(readers):
            await queue.put(None)

    async def _prepare(self, page: _SourcePage) -> _PageJob:
        logger.info("Page %d/%d is %s", page.number, len(self._pages), page.label)
        digest = saved = None
        if self._layout_directory is not None:
            if page.path not in self._digests:
                self._digests[page.path] = await asyncio.to_thread(_file_digest, page.path)
            digest = self._digests[page.path]
            if not self._ignore_saved_layouts:
                saved = await asyncio.to_thread(self._find_saved_layout, page, digest)
        if saved is not None and not _has_figures(saved.layout):
            return _PageJob(page, digest, saved, None)
        # A saved layout has its boxes on the page image after the turn of its record.
        turn = saved.record.turn if saved is not None else self._rotate or 0
        image_path = self._work_directory / "pages" / f"page-{page.number:06d}.png"
        try:
            await asyncio.to_thread(_make_page_image, page, image_path, dpi=self._dpi, turn=turn)
        except Exception:
            image_path.unlink(missing_ok=True)
            raise
        return _PageJob(page, digest, saved, image_path)

    def _find_saved_layout(self, page: _SourcePage, digest: str) -> _SavedLayout | None:
        """Return the saved layout of a page, or None for a page that has none.

        The file with the name of the page is the first place. A file with a different
        name can have the layout also, for an input file that has a new name.
        """
        assert self._layout_directory is not None
        path = self._layout_directory / _layout_file_name(page, digest)
        if not path.is_file():
            other_paths = self._saved_index.get((digest, page.page_of_input))
            if not other_paths:
                return None
            path = other_paths[0]
            if len(other_paths) > 1:
                logger.warning(
                    "%d layout files are for page %d (%s), and none has the usual name %s. "
                    "The conversion uses %s. Remove the files that you do not want",
                    len(other_paths),
                    page.number,
                    ", ".join(other_path.name for other_path in other_paths),
                    _layout_file_name(page, digest),
                    path.name,
                )
        return _read_saved_layout(path, page, digest, self._rotate)

    async def _read(self, queue: asyncio.Queue[_PageJob | None]) -> None:
        """Take pages from the queue, and get the layout and the illustrations of each."""
        while (job := await queue.get()) is not None:
            if self._stop.is_set():
                # The conversion starts no more pages. This page stays without a layout.
                if job.image_path is not None:
                    job.image_path.unlink(missing_ok=True)
                continue
            with _log_page(job.page, len(self._pages)):
                started = asyncio.get_running_loop().time()
                try:
                    layout = await self._convert_page(job)
                except OutputDirectoryError:
                    # No page can have a layout file, so this is the error of the conversion.
                    raise
                except Exception as error:
                    self._page_failed(job.page, error, read=job.saved is None, started=started)
                else:
                    self._layouts[job.page.number] = layout
                    if job.saved is None:
                        self._read_pages += 1
                        # The service gives replies.
                        self._waves_without_reply = 0
                    else:
                        self._saved_pages += 1

    async def _convert_page(self, job: _PageJob) -> PageLayout:
        """Return the layout of one page, and write its illustrations as image files.

        The page image goes away at the end, so that the page images of a long document
        do not all exist at the same time.
        """
        page = job.page
        region_path = self._work_directory / "regions" / f"page-{page.number:06d}.png"
        try:
            if job.saved is None:
                layout = await self._read_layout(job, region_path)
            else:
                layout = job.saved.layout
                logger.info(
                    "Page %d has the saved layout %s, so it needs no request",
                    page.number,
                    job.saved.path.name,
                )
                self._note_other_settings(page, job.saved)
            if not layout.blocks:
                logger.warning(
                    "Page %d has no content in its layout; compare it with the source",
                    page.number,
                )
            if job.image_path is None:
                return layout
            return await asyncio.to_thread(
                _crop_figures,
                layout,
                job.image_path,
                self._new_figures_directory,
                page_number=page.number,
                figure_directory_name=self._figure_directory_name,
            )
        finally:
            if job.image_path is not None:
                job.image_path.unlink(missing_ok=True)
            region_path.unlink(missing_ok=True)

    async def _read_layout(self, job: _PageJob, region_path: Path) -> PageLayout:
        """Return the layout that the backend gives for a page, and save it."""
        page = job.page
        assert job.image_path is not None
        if self._layout_directory is not None:
            # A request has a cost, and its layout needs the directory.
            await self._check_layout_directory()
        logger.info(
            "Recognizing the layout of page %d/%d with %s",
            page.number,
            len(self._pages),
            type(self._backend).__name__,
        )
        layout, turn = await _recognize_upright_layout(
            self._calls, job.image_path, page.number, self._rotate
        )
        if self._calls.locates_figures:
            layout = await _locate_figures(
                self._calls, layout, job.image_path, region_path, page.number
            )
        logger.info("Recognized %d block(s) on page %d", len(layout.blocks), page.number)
        layout = await asyncio.to_thread(_checked_layout, page, layout)
        if self._layout_directory is not None:
            assert job.digest is not None
            record = _SourceRecord(
                digest=job.digest, page=page.page_of_input, turn=turn, reading=self._reading(page)
            )
            # The layout is complete here. The file is there before the crops start, so
            # that a later failure does not lose a layout that has a cost.
            path = await asyncio.to_thread(
                _write_layout_file, self._layout_directory, page, record, layout
            )
            logger.info("Wrote layout JSON for page %d to %s", page.number, path.name)
        return layout

    async def _check_layout_directory(self) -> None:
        """Raise `OutputDirectoryError` when a layout file cannot go to the directory."""
        assert self._layout_directory is not None
        async with self._layout_directory_check:
            if not self._layout_directory_is_checked:
                await asyncio.to_thread(
                    _check_directory, self._layout_directory, "the layout directory"
                )
                self._layout_directory_is_checked = True

    def _reading(self, page: _SourcePage) -> dict[str, Any]:
        """Return the backend and the settings that give the layout of a page."""
        reading: dict[str, Any] = {"backend": type(self._backend).__name__}
        if page.from_pdf:
            reading["dpi"] = self._dpi
        reading["settings"] = self._settings
        return reading

    def _note_other_settings(self, page: _SourcePage, saved: _SavedLayout) -> None:
        """Count a saved layout that is from other settings than those of this conversion."""
        now, then = _flat_reading(self._reading(page)), _flat_reading(saved.record.reading)
        names = sorted(name for name in now.keys() | then.keys() if now.get(name) != then.get(name))
        if not names:
            return
        self._other_settings_pages += 1
        for name in names:
            self._other_settings[name] = self._other_settings.get(name, 0) + 1
        logger.info(
            "The saved layout of page %d is from other settings (%s)",
            page.number,
            ", ".join(names),
        )

    def _page_failed(
        self, page: _SourcePage, error: Exception, *, read: bool, started: float = -math.inf
    ) -> None:
        """Record a page that has no layout. `read` is True when its requests failed.

        `started` is the time of the event loop when the work for the page started.

        A conversion that saves its layouts continues with the other pages: one new run
        then completes the document, and it sends requests only for the pages that
        failed. A conversion that saves no layouts stops, because each more page has a
        cost and a new run sends its request again.

        The conversion also starts no more pages when the service gives no reply for
        some waves of pages in sequence. Without this rule, a service that stops in the
        middle of a book makes each later page wait for its timeout. It also starts no
        more pages when its first pages with a reply all fail and no page has a layout.
        """
        if self._layout_directory is None:
            if self._layouts:
                logger.warning(
                    "Page %d failed, and the conversion saves no layouts: the layouts of %d "
                    "page(s) are lost. With --ocr-output-dir, a new run sends requests only "
                    "for the pages that have no saved layout",
                    page.number,
                    len(self._layouts),
                )
            raise error
        message = str(error)
        self._failures[page.number] = message
        logger.error(
            "Page %d/%d (%s) has no layout: %s", page.number, len(self._pages), page.label, message
        )
        if not read:
            return
        if not isinstance(error, NoReplyError):
            # A page with no reply is for the rule of the waves below: its cause is
            # probably the service, not the model.
            self._failed_readings += 1
            no_layout = not (self._read_pages or self._saved_pages or self._saved_index)
            if (
                no_layout
                and self._failed_readings >= FIRST_FAILED_PAGES
                and not self._stop.is_set()
            ):
                self._stop.set()
                self._stop_reason = STOP_FIRST_PAGES
                logger.error(
                    "The first %d pages with a reply of the service failed, and no page has a "
                    "layout, so the conversion starts no more pages",
                    self._failed_readings,
                )
            return
        self._pages_without_reply += 1
        if started > self._last_counted_no_reply:
            # A page that was in work when the last page counted belongs to that wave.
            self._waves_without_reply += 1
            self._last_counted_no_reply = asyncio.get_running_loop().time()
        if self._waves_without_reply >= NO_REPLY_WAVES and not self._stop.is_set():
            self._stop.set()
            self._stop_reason = STOP_NO_REPLY
            logger.error(
                "The pages in work got no reply from the service %d times in sequence, so the "
                "conversion starts no more pages",
                self._waves_without_reply,
            )


def _has_figures(layout: PageLayout) -> bool:
    return any(isinstance(block, Figure) for block in layout.blocks)


def _flat_reading(reading: Mapping[str, Any]) -> dict[str, Any]:
    """Return the backend and the settings of a record as one set of names and values."""
    settings = reading.get("settings")
    flat = {name: value for name, value in reading.items() if name != "settings"}
    if isinstance(settings, Mapping):
        flat.update(settings)
    return flat


def _page_ranges(numbers: Sequence[int]) -> str:
    """Return page numbers as text, with a range for numbers in sequence: "4-6, 9"."""
    ranges: list[list[int]] = []
    for number in numbers:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ", ".join(str(first) if first == last else f"{first}-{last}" for first, last in ranges)


def _make_page_image(page: _SourcePage, image_path: Path, *, dpi: int | None, turn: int) -> None:
    """Write the image file of one page, turned clockwise by `turn` degrees."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if page.from_pdf:
        rasterize_pdf_page(page.path, page.page_of_input - 1, image_path, dpi=dpi)
        if turn:
            _turn_page_image(image_path, turn)
        return
    logger.info("Reading %s as the image of one page", page.path)
    image = read_photo(page.path)
    logger.info("The page image has %dx%d pixels", image.width, image.height)
    turn_clockwise(image, turn).save(image_path, format="PNG", compress_level=PNG_COMPRESSION_LEVEL)


def _file_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _layout_file_name(page: _SourcePage, digest: str) -> str:
    """Return the name of the file for the saved layout of a page.

    The name has the name of the input file and a part of the hash of its bytes. Thus
    the layout of a page does not depend on the place of the page in the document, and
    an input file that changes gets a new layout file.
    """
    name = page.given.name
    while len(os.fsencode(name)) > MAX_LAYOUT_NAME_BYTES:
        name = name[:-1]
    if page.from_pdf:
        name += f"-page-{page.page_of_input:04d}"
    return f"{name}-{digest[:LAYOUT_NAME_HASH_CHARACTERS]}{LAYOUT_FILE_SUFFIX}"


def _source_record_from_json(value: Any) -> _SourceRecord | None:
    """Return the record of a saved layout, or None for a value that is not such a record."""
    if not isinstance(value, dict) or value.get("record") != SOURCE_RECORD_VERSION:
        return None
    digest, page, turn = value.get("sha256"), value.get("page"), value.get("turn")
    if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
        return None
    if type(page) is not int or page < 1 or type(turn) is not int or turn not in PAGE_ROTATIONS:
        return None
    reading = value.get("reading")
    return _SourceRecord(
        digest=digest, page=page, turn=turn, reading=reading if isinstance(reading, dict) else {}
    )


def _saved_layout_index(directory: Path) -> dict[tuple[str, int], list[Path]]:
    """Return the layout files of a directory, by input file hash and page of the input.

    More than one file can have the same record, for example a copy that a person made.
    The files of one record are in the sequence of their names.
    """
    index: dict[tuple[str, int], list[Path]] = {}
    if not directory.is_dir():
        return index
    for path in sorted(directory.glob(f"*{LAYOUT_FILE_SUFFIX}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # The directory can have files that are not layouts of Scanwich.
            continue
        record = _source_record_from_json(
            payload.get(SOURCE_FIELD) if isinstance(payload, dict) else None
        )
        if record is not None:
            index.setdefault((record.digest, record.page), []).append(path)
    return index


def _read_saved_layout(
    path: Path, page: _SourcePage, digest: str, rotate: int | None
) -> _SavedLayout:
    """Return the saved layout of a page from its file.

    A file that is not usable gives `PageLayoutError`, and no new request: a new layout
    would replace a file that can have the work of a person.
    """
    remedy = "Correct the file, or remove it to get a new layout of the page"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PageLayoutError(
            f"the saved layout {path} is not valid JSON: {error.msg} at line {error.lineno}, "
            f"column {error.colno}. {remedy}"
        ) from None
    except (OSError, UnicodeDecodeError) as error:
        raise PageLayoutError(f"could not read the saved layout {path}: {error}") from error
    record = _source_record_from_json(
        payload.get(SOURCE_FIELD) if isinstance(payload, dict) else None
    )
    if record is None:
        raise PageLayoutError(
            f"the saved layout {path} has no usable {SOURCE_FIELD!r} record, so Scanwich does "
            f"not know which page it is for. {remedy}"
        )
    if (record.digest, record.page) != (digest, page.page_of_input):
        raise PageLayoutError(
            f"the saved layout {path} is for a different input file or a different page of "
            f"it. {remedy}"
        )
    if rotate is not None and record.turn != rotate:
        raise PageLayoutError(
            f"the saved layout {path} is for the page image after a turn of {record.turn} "
            f"degrees, and --rotate gives {rotate} degrees. Remove the file, or use "
            "--ignore-saved-layouts, to get a new layout"
        )
    try:
        layout = _layout_of_file(payload)
    except LayoutFormatError as error:
        hint = f" {error.hint}" if error.hint else ""
        raise PageLayoutError(
            f"the saved layout {path} is not usable: {error}.{hint} {remedy}"
        ) from None
    return _SavedLayout(path=path, layout=layout, record=record)


def _layout_of_file(payload: Mapping[str, Any]) -> PageLayout:
    """Return the layout that the JSON object of a layout file has.

    The function raises `LayoutFormatError` for a layout that a conversion cannot use.
    """
    layout = page_layout_from_json(
        {name: value for name, value in payload.items() if name != SOURCE_FIELD}
    )
    if layout.page_top != PAGE_TOP_UPRIGHT:
        raise LayoutFormatError(f"page_top must be {PAGE_TOP_UPRIGHT!r} in a saved layout")
    return layout


def _checked_layout(page: _SourcePage, layout: PageLayout) -> PageLayout:
    """Return the layout of a backend as the check of a layout file reads it.

    A backend can return a layout as an object, without the check of the JSON form. Each
    conversion gives each such layout this check, with and without a layout directory:
    a layout that the check refuses gives `PageLayoutError`. The document is from the
    result, so that a later conversion with the saved layout gives the same document.
    """
    try:
        return _layout_of_file(json.loads(_json_text(layout.to_json())))
    except LayoutFormatError as error:
        hint = f" {error.hint}" if error.hint else ""
        raise PageLayoutError(
            f"the backend gave a layout for page {page.number} that the check of a layout "
            f"file refuses: {error}.{hint}"
        ) from None


def _write_layout_file(
    directory: Path, page: _SourcePage, record: _SourceRecord, layout: PageLayout
) -> Path:
    """Save the layout of a page with the record of its source, and return the file.

    The layout is the result of `_checked_layout`, so a later conversion can read it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _layout_file_name(page, record.digest)
    payload = {
        SOURCE_FIELD: {
            "record": SOURCE_RECORD_VERSION,
            "file": page.given.name,
            "sha256": record.digest,
            "page": record.page,
            "turn": record.turn,
            "reading": record.reading,
        },
        **layout.to_json(),
    }
    text = _json_text(payload)
    # A stop in the middle of the write must not leave a part of a file. Two conversions
    # can write the layout of one page at the same time, so each write has a name of
    # its own for its temporary file.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:16]}{TEMPORARY_SUFFIX}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


async def _recognize_upright_layout(
    calls: _BackendCalls, page_image: Path, page_number: int, rotate: int | None
) -> tuple[PageLayout, int]:
    """Return the layout of a page, and the clockwise angle that its image turned by.

    The boxes of the layout fit the page image file. With `rotate`, the caller gives the
    direction of the page: the page image has that turn, and the backend gets the page
    as upright. Without `rotate`, the backend finds the direction, and the function
    turns the page image file one time when the backend tells it to.

    The function raises `PageLayoutError` for a page that has no usable layout, so that a
    page does not disappear from the document without a failure.
    """
    if rotate is not None:
        layout = await calls.recognize_layout(page_image, upright=True)
        if layout.page_top == PAGE_TOP_UPRIGHT:
            return layout, rotate
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
        return replace(layout, page_top=PAGE_TOP_UPRIGHT), rotate

    layout = await calls.recognize_layout(page_image)
    if layout.page_top == PAGE_TOP_UPRIGHT:
        return layout, 0
    first_side = layout.page_top
    degrees = UPRIGHT_TURNS[first_side]
    logger.info(
        "The top of page %d is at the %s of its image; turning the image %d degrees "
        "clockwise and recognizing the page again",
        page_number,
        first_side,
        degrees,
    )
    await asyncio.to_thread(_turn_page_image, page_image, degrees)
    layout = await calls.recognize_layout(page_image)
    if layout.page_top != PAGE_TOP_UPRIGHT:
        raise PageLayoutError(
            f"page {page_number} has no usable layout: the backend reported the top of the "
            f"page at the {first_side} of the image and, after a turn of {degrees} degrees "
            f"clockwise, at the {layout.page_top}; set --rotate to give the direction of the "
            "pages"
        )
    return layout, degrees


def _turn_page_image(page_image: Path, degrees: int) -> None:
    with Image.open(page_image) as image:
        turned = turn_clockwise(image, degrees)
    turned.save(page_image, format="PNG", compress_level=PNG_COMPRESSION_LEVEL)


def _save_region(page_image: Path, region: Box, region_path: Path) -> None:
    """Write the part of a page image that a request for an illustration shows."""
    region_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(page_image) as image:
        crop_box(image, region).save(region_path, format="PNG", compress_level=1)


async def _locate_figures(
    calls: _BackendCalls,
    layout: PageLayout,
    page_image: Path,
    region_path: Path,
    page_number: int,
) -> PageLayout:
    """Return the layout with the box that a second request gives for each illustration."""
    if not _has_figures(layout):
        return layout
    blocks = []
    figure_number = 0
    requests_left = FIGURE_SEARCH_PAGE_REQUESTS
    without_request = 0
    for block in layout.blocks:
        if isinstance(block, Figure):
            figure_number += 1
            if requests_left:
                box, requests = await _locate_figure(
                    calls,
                    page_image,
                    block,
                    region_path,
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


async def _locate_figure(
    calls: _BackendCalls,
    page_image: Path,
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
        # The crop uses the processor, so it is in a thread.
        await asyncio.to_thread(_save_region, page_image, region, region_path)
        try:
            location = await calls.locate_figure(
                region_path, estimate=_box_in_region(box, region), caption=figure.caption
            )
        except RateLimitError:
            # The service takes no request now. The page then has no layout, so that a
            # new run asks again. A saved layout with the first box gets no second request.
            raise
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
    output_path.write_text(_json_text(payload), encoding="utf-8")


def _json_text(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    # A file in UTF-8 cannot have one half of a character. JSON has an escape for it.
    text = SURROGATE.sub(lambda match: f"\\u{ord(match.group()):04x}", text)
    return text + "\n"


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
