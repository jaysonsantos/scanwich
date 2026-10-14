"""LaTeX output for a book: many input files, saved layouts, and parallel pages."""

import asyncio
import contextlib
import gc
import hashlib
import json
import os
import shutil
import signal
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
from reportlab.pdfgen import canvas

from scanwich import pipeline
from scanwich.layout import (
    Box,
    Figure,
    FigureLocation,
    Footnote,
    Heading,
    PageFurniture,
    PageLayout,
    Paragraph,
    Table,
    TableCell,
)
from scanwich.models import OcrRegion
from scanwich.ocr import (
    AsyncFigureLocator,
    AsyncLayoutBackend,
    FigureLocator,
    LayoutBackend,
    LayoutSettings,
    NoReplyError,
    RateLimitError,
)
from scanwich.pipeline import (
    DEFAULT_PARALLEL_PAGES,
    NO_REPLY_WAVES,
    PAGE_QUEUE_SIZE,
    ConversionStopped,
    OutputDirectoryError,
    PageLayoutError,
    convert_to_latex,
)

LOGGER_NAME = "scanwich.pipeline"
RED = (255, 0, 0)
WHITE = (255, 255, 255)
PAGE_HEIGHT = 100
# The box of the red part of a page image, on the grid of the page.
RED_BOX = Box(750, 0, 1000, 500)


def page_width(number: int) -> int:
    """Return the width of the image of a page. The width tells a backend which page it is."""
    return 200 + 10 * number


def text_of(number: int) -> str:
    return f"Text of page {number}."


class PageBackend:
    """A backend with only the synchronous methods.

    The longer side of a page image tells the page, so the layout of a page does not
    depend on the sequence of the requests. A page without a given layout has one
    paragraph with the number of the page.
    """

    def __init__(
        self,
        layouts: Mapping[int, PageLayout] | None = None,
        *,
        failing: Collection[int] = (),
        no_reply: Collection[int] = (),
        delay: float = 0.0,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        self.layouts = dict(layouts or {})
        # The service answers a failing page with a reply that is not usable. It gives
        # no reply for a page of `no_reply`.
        self.failing = set(failing)
        self.no_reply = set(no_reply)
        self.delay = delay
        self.settings = settings
        # The pages in the sequence of the requests, and of the replies.
        self.requests: list[int] = []
        self.finished: list[int] = []
        self.upright: list[bool] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.max_page_images = 0
        self._count = threading.Lock()

    def recognize(self, image_path: Path) -> list[OcrRegion]:
        return []

    def start(self, image_path: Path, upright: bool) -> int:
        with Image.open(image_path) as image:
            page = (max(image.size) - 200) // 10
        with self._count:
            self.requests.append(page)
            self.upright.append(upright)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.count_page_images(image_path)
        return page

    def count_page_images(self, image_path: Path) -> None:
        # The page images of a conversion are the files of one directory.
        count = len(list(image_path.parent.glob("*.png")))
        with self._count:
            self.max_page_images = max(self.max_page_images, count)

    def end(self, page: int) -> PageLayout:
        with self._count:
            self.in_flight -= 1
            self.finished.append(page)
        if page in self.failing:
            raise RuntimeError(f"the service did not read page {page}")
        if page in self.no_reply:
            raise NoReplyError(f"the service gave no reply for page {page}")
        return self.layouts.get(page, PageLayout(blocks=(Paragraph(text_of(page)),)))

    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        page = self.start(image_path, upright)
        time.sleep(self.delay)
        self.count_page_images(image_path)
        return self.end(page)

    def layout_settings(self) -> Mapping[str, object]:
        return {} if self.settings is None else self.settings


class AsyncPageBackend(PageBackend):
    """A backend with the async methods. `delays` gives a delay to the reply of a page."""

    def __init__(
        self,
        layouts: Mapping[int, PageLayout] | None = None,
        *,
        delays: Mapping[int, float] | None = None,
        on_request: Callable[[int], None] | None = None,
        **options: object,
    ) -> None:
        super().__init__(layouts, **options)  # type: ignore[arg-type]
        self.delays = dict(delays or {})
        self.on_request = on_request
        self.started_at: dict[int, float] = {}
        self.closed = 0

    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        raise AssertionError("a backend with the async methods gets no synchronous call")

    async def recognize_layout_async(
        self, image_path: Path, *, upright: bool = False
    ) -> PageLayout:
        page = self.start(image_path, upright)
        self.started_at.setdefault(page, time.monotonic())
        try:
            if self.on_request is not None:
                self.on_request(page)
            await asyncio.sleep(self.delays.get(page, self.delay))
            self.count_page_images(image_path)
        except BaseException:
            with self._count:
                self.in_flight -= 1
            raise
        return self.end(page)

    async def aclose(self) -> None:
        self.closed += 1
        # The backend closes its connections after the last request ended.
        self.in_flight_at_close = self.in_flight


class NoRequestBackend(PageBackend):
    """A backend that must get no request."""

    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        self.start(image_path, upright)
        raise AssertionError("a page with a saved layout needs no request")

    def locate_figure(self, image_path: Path, **hints: object) -> FigureLocation | None:
        self.requests.append(0)
        raise AssertionError("a page with a saved layout needs no request for a box")


class BookTestCase(TestCase):
    def setUp(self) -> None:
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name).resolve()
        self.pages = self.directory / "pages"
        self.pages.mkdir()
        self.layouts = self.directory / "layouts"
        self.output = self.directory / "out" / "book.tex"

    def photo(self, number: int, name: str | None = None) -> Path:
        """Write the photo of a page: white, with a red part at its top-right corner."""
        path = self.pages / (name or f"photo-{number:02d}.png")
        width = page_width(number)
        image = Image.new("RGB", (width, PAGE_HEIGHT), WHITE)
        image.paste(RED, (width * 3 // 4, 0, width, PAGE_HEIGHT // 2))
        image.save(path)
        return path

    def photos(self, count: int) -> list[Path]:
        return [self.photo(number) for number in range(1, count + 1)]

    def pdf(self, numbers: Collection[int], name: str = "part.pdf") -> Path:
        """Write a PDF. At 72 DPI, the image of each page has the width of its number."""
        path = self.pages / name
        source = canvas.Canvas(str(path))
        for number in numbers:
            source.setPageSize((page_width(number), PAGE_HEIGHT))
            source.drawString(10, 50, f"page {number}")
            source.showPage()
        source.save()
        return path

    def convert(self, inputs: object, backend: object, **options: object) -> None:
        options.setdefault("ocr_output_directory", self.layouts)
        convert_to_latex(inputs, self.output, backend=backend, **options)  # type: ignore[arg-type]

    def document(self) -> str:
        return self.output.read_text(encoding="utf-8")

    def layout_files(self) -> list[str]:
        return sorted(path.name for path in self.layouts.iterdir())

    def layout_file(self, source: Path, page: int | None = None) -> Path:
        """Return the layout file of an input file, or of one page of a PDF."""
        part = hashlib.sha256(source.read_bytes()).hexdigest()[:8]
        page_part = "" if page is None else f"-page-{page:04d}"
        return self.layouts / f"{source.name}{page_part}-{part}.json"

    def figure_files(self) -> dict[str, bytes]:
        directory = self.output.with_name("book-figures")
        if not directory.is_dir():
            return {}
        return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


class ManyInputsTests(BookTestCase):
    def test_the_inputs_give_one_document_in_the_sequence_of_the_command(self) -> None:
        # A photo, a PDF with 2 pages, and a photo. The rules for the pages of one PDF
        # are also the rules across the files.
        inputs = [self.photo(1), self.pdf([2, 3]), self.photo(4)]
        backend = PageBackend(
            {
                1: PageLayout(blocks=(Paragraph("The first file ends in the middle of a sen-"),)),
                2: PageLayout(
                    blocks=(Paragraph("tence, and the paragraph continues", continued=True),)
                ),
                3: PageLayout(
                    blocks=(
                        Paragraph("on the next page of the PDF.", continued=True),
                        Figure(box=RED_BOX, caption="Fig. 3"),
                    )
                ),
                4: PageLayout(blocks=(Paragraph("The last file."), Figure(box=RED_BOX))),
            }
        )

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            self.convert(inputs, backend, dpi=72, parallel_pages=1)

        self.assertEqual(backend.requests, [1, 2, 3, 4])
        document = self.document()
        # The word that the end of the first file divides is complete on its page.
        self.assertIn(
            "\\begin{sourcepage}{1}\n\nThe first file ends in the middle of a sentence,\n",
            document,
        )
        self.assertIn("\\begin{sourcepage}{2}\n\n\\noindent and the paragraph continues", document)
        self.assertIn("\\begin{sourcepage}{3}\n\n\\noindent on the next page of the PDF.", document)
        self.assertIn("\\begin{sourcepage}{4}\n\nThe last file.", document)
        # Each source page is one page of the document.
        self.assertEqual(document.count("\\begin{sourcepage}"), 4)
        # The name of an illustration has the page of the document, not of its file.
        self.assertEqual(
            sorted(self.figure_files()), ["page-0003-figure-01.jpg", "page-0004-figure-01.jpg"]
        )
        self.assertIn("{book-figures/page-0003-figure-01.jpg}", document)
        self.assertEqual(
            self.layout_files(),
            sorted(
                [
                    self.layout_file(inputs[0]).name,
                    self.layout_file(inputs[1], 1).name,
                    self.layout_file(inputs[1], 2).name,
                    self.layout_file(inputs[2]).name,
                ]
            ),
        )
        progress = "\n".join(captured_logs.output)
        self.assertIn("The document has 4 page(s) from 3 input file(s)", progress)
        self.assertIn(f"Page 3/4 is {inputs[1]}, page 2", progress)
        self.assertIn(f"Page 4/4 is {inputs[2]}", progress)

    def test_one_input_is_still_one_argument(self) -> None:
        self.convert(self.photo(1), PageBackend())
        self.assertIn(text_of(1), self.document())

    def test_each_input_must_exist_before_the_first_request(self) -> None:
        backend = PageBackend()
        with self.assertRaisesRegex(FileNotFoundError, "input file does not exist: .*missing.png"):
            self.convert([self.photo(1), self.pages / "missing.png"], backend)
        with self.assertRaisesRegex(ValueError, "needs one input file or more"):
            self.convert([], backend)
        with self.assertRaisesRegex(ValueError, "must be different"):
            convert_to_latex([self.photo(1), self.photo(2)], self.photo(2), backend=backend)
        self.assertEqual(backend.requests, [])

    def test_document_without_content_names_the_number_of_inputs(self) -> None:
        backend = PageBackend({1: PageLayout(), 2: PageLayout()})
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(PageLayoutError, "no page of the 2 input files has content"),
        ):
            self.convert(self.photos(2), backend)

    def test_warns_when_the_sequence_of_the_names_is_not_that_of_their_numbers(self) -> None:
        # A shell sorts the names of a pattern by their characters.
        names = ["page1.png", "page10.png", "page2.png"]
        inputs = [self.photo(number, name) for number, name in enumerate(names, start=1)]
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            self.convert(inputs, PageBackend())
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:The input files are in the sequence of their "
                    "characters, and the numbers in their names give a different sequence: "
                    f"{inputs[1]} is before {inputs[2]}. The pages of the document have the "
                    "sequence of the command. Compare it with the source"
                )
            ],
        )
        # The sequence of the command stays.
        self.assertLess(self.document().index(text_of(2)), self.document().index(text_of(3)))

    def test_no_warning_for_names_in_the_sequence_of_their_numbers(self) -> None:
        for names in (["page01.png", "page02.png", "page10.png"], ["b.png", "a.png", "c.png"]):
            with self.subTest(names=names), self.assertNoLogs(LOGGER_NAME, level="WARNING"):
                inputs = [self.photo(number, name) for number, name in enumerate(names, start=1)]
                self.convert(inputs, PageBackend())

    def test_name_with_a_digit_that_is_not_a_decimal_digit(self) -> None:
        # `str.isdigit` is true for such a character, and `int` does not read it.
        names = ["1\u00b23.png", "10\u00b2.png", "a\u00b2b.png", "\u0663.png"]
        inputs = [self.photo(number, name) for number, name in enumerate(names, start=1)]
        backend = PageBackend()
        self.convert(inputs, backend, parallel_pages=1)
        self.assertEqual(backend.requests, [1, 2, 3, 4])

    def test_conversion_of_many_pages_that_saves_no_layouts_warns_at_its_start(self) -> None:
        # A failure at page 199 of 200 loses 198 layouts. The user must know this before
        # the first request, not after the failure.
        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            self.convert(self.photos(3), PageBackend(), ocr_output_directory=None)
        warning = (
            f"WARNING:{LOGGER_NAME}:The conversion has 3 pages and saves no layouts: a failure "
            "at one page loses the layouts of the pages before it. With --ocr-output-dir "
            "DIRECTORY, a new run sends requests only for the pages that have no saved layout"
        )
        self.assertIn(warning, captured_logs.output)
        first_request = next(
            index
            for index, record in enumerate(captured_logs.output)
            if "Recognizing the layout of page" in record
        )
        self.assertLess(captured_logs.output.index(warning), first_request)

        # One page has no pages before it, and a conversion that saves its layouts loses none.
        with self.assertNoLogs(LOGGER_NAME, level="WARNING"):
            self.convert(self.photos(1), PageBackend(), ocr_output_directory=None)
            self.convert(self.photos(3), PageBackend())

    def test_caller_with_a_running_event_loop_gets_a_clear_error(self) -> None:
        inputs = self.photos(1)
        backend = PageBackend()

        async def call_in_the_loop() -> None:
            self.convert(inputs, backend)

        with self.assertRaisesRegex(
            RuntimeError,
            "convert_to_latex starts an event loop of its own, so a thread with a running "
            "event loop cannot call it; call it from a different thread, for example with "
            "asyncio.to_thread",
        ):
            asyncio.run(call_in_the_loop())
        self.assertEqual(backend.requests, [])
        self.assertFalse(self.layouts.exists())

        # The way that the message names.
        async def call_in_a_thread() -> None:
            await asyncio.to_thread(self.convert, inputs, backend)

        asyncio.run(call_in_a_thread())
        self.assertEqual(backend.requests, [1])
        self.assertIn(text_of(1), self.document())


class SavedLayoutTests(BookTestCase):
    def test_new_run_sends_no_request_for_a_page_that_has_a_saved_layout(self) -> None:
        inputs = [self.photo(1), self.pdf([2, 3]), self.photo(4)]
        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=RED_BOX, caption="Fig. 1")))
        self.convert(inputs, PageBackend({3: figure, 4: figure}), dpi=72)
        first_document, first_figures = self.document(), self.figure_files()
        first_layouts = {path.name: path.read_bytes() for path in self.layouts.iterdir()}
        self.assertEqual(
            sorted(first_figures), ["page-0003-figure-01.jpg", "page-0004-figure-01.jpg"]
        )

        backend = NoRequestBackend()
        made: list[int] = []
        make_page_image = pipeline._make_page_image

        def counting(page: object, *arguments: object, **options: object) -> None:
            made.append(page.number)  # type: ignore[attr-defined]
            make_page_image(page, *arguments, **options)

        with (
            patch.object(pipeline, "_make_page_image", counting),
            self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs,
        ):
            self.convert(inputs, backend, dpi=72)

        self.assertEqual(backend.requests, [])
        self.assertIsInstance(backend, FigureLocator)
        # Only a page with an illustration needs its page image, for the crop.
        self.assertEqual(made, [3, 4])
        # The document and its illustrations are the same, and no layout file changed.
        self.assertEqual(self.document(), first_document)
        self.assertEqual(self.figure_files(), first_figures)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.layouts.iterdir()}, first_layouts
        )
        progress = "\n".join(captured_logs.output)
        self.assertIn(
            f"Page 2 has the saved layout {self.layout_file(inputs[1], 1).name}, so it needs "
            "no request",
            progress,
        )
        self.assertIn(
            "0 page(s) got a layout from requests, and 4 page(s) have a saved layout", progress
        )

    def test_document_from_layouts_that_a_person_changed_needs_no_request(self) -> None:
        inputs = self.photos(2)
        self.convert(inputs, PageBackend())
        layout_file = self.layout_file(inputs[1])
        saved = json.loads(layout_file.read_text(encoding="utf-8"))
        saved["blocks"][0]["text"] = "A person corrected this text."
        saved["blocks"].append({"type": "figure", "box": RED_BOX.to_json()})
        layout_file.write_text(json.dumps(saved), encoding="utf-8")

        backend = NoRequestBackend()
        self.convert(inputs, backend)

        self.assertEqual(backend.requests, [])
        document = self.document()
        self.assertIn("A person corrected this text.", document)
        self.assertNotIn(text_of(2), document)
        # The illustration that the person added is a crop of the page image.
        self.assertEqual(list(self.figure_files()), ["page-0002-figure-01.jpg"])
        # Scanwich does not write a layout file that it uses.
        self.assertEqual(json.loads(layout_file.read_text(encoding="utf-8")), saved)

    def test_record_ties_a_layout_to_its_input(self) -> None:
        photo, pdf = self.photo(1), self.pdf([2])
        backend = PageBackend(settings={"model": "provider/model-a", "languages": ("fr", "la")})
        self.assertIsInstance(backend, LayoutSettings)
        self.convert([photo, pdf], backend, dpi=72, rotate=0)

        record = json.loads(self.layout_file(photo).read_text(encoding="utf-8"))["source"]
        self.assertEqual(
            record,
            {
                "record": 1,
                "file": "photo-01.png",
                "sha256": hashlib.sha256(photo.read_bytes()).hexdigest(),
                "page": 1,
                "turn": 0,
                "reading": {
                    "backend": "PageBackend",
                    "settings": {"model": "provider/model-a", "languages": ["fr", "la"]},
                },
            },
        )
        record = json.loads(self.layout_file(pdf, 1).read_text(encoding="utf-8"))["source"]
        self.assertEqual((record["file"], record["page"]), ("part.pdf", 1))
        # The image of a PDF page depends on the DPI.
        self.assertEqual(record["reading"]["dpi"], 72)

    def test_changed_input_gets_a_new_layout(self) -> None:
        inputs = self.photos(2)
        self.convert(inputs, PageBackend())
        old_file = self.layout_file(inputs[1])
        # The same name and the same size, with different bytes.
        with Image.open(inputs[1]) as image:
            changed = image.copy()
        changed.putpixel((0, PAGE_HEIGHT - 1), (0, 0, 255))
        changed.save(inputs[1])

        backend = PageBackend({2: PageLayout(blocks=(Paragraph("The new photo."),))})
        self.convert(inputs, backend)

        self.assertEqual(backend.requests, [2])
        self.assertIn("The new photo.", self.document())
        # The layout of the earlier bytes stays in its file.
        self.assertTrue(old_file.is_file())
        self.assertNotEqual(self.layout_file(inputs[1]), old_file)
        self.assertEqual(len(self.layout_files()), 3)

    def test_input_with_a_new_name_or_a_new_place_keeps_its_layout(self) -> None:
        inputs = self.photos(3)
        self.convert(inputs, PageBackend())
        renamed = inputs[2].rename(self.pages / "a-new-name.png")
        added = self.photo(4)

        # The page with the new name is now the first page, and a new page is in the middle.
        backend = PageBackend()
        self.convert([renamed, inputs[0], added, inputs[1]], backend)

        self.assertEqual(backend.requests, [4])
        document = self.document()
        places = [document.index(text_of(number)) for number in (3, 1, 4, 2)]
        self.assertEqual(places, sorted(places))

    def test_rotate_that_is_not_the_turn_of_a_saved_layout_is_a_failure(self) -> None:
        photo = self.photo(1)
        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=Box(0, 0, 500, 250))))
        self.convert(photo, PageBackend({1: figure}), rotate=90)
        layout_file = self.layout_file(photo)
        saved, first_figures = layout_file.read_bytes(), self.figure_files()
        self.assertEqual(json.loads(saved)["source"]["turn"], 90)

        backend = NoRequestBackend()
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaisesRegex(
                PageLayoutError,
                "is for the page image after a turn of 90 degrees, and --rotate gives 270 "
                "degrees. Remove the file, or use --ignore-saved-layouts, to get a new layout",
            ),
        ):
            self.convert(photo, backend, rotate=270)
        self.assertEqual(backend.requests, [])
        self.assertEqual(layout_file.read_bytes(), saved)

        # Without `rotate`, and with the same angle, the page image gets the turn of the
        # record, so the crop is the same.
        for rotate in (None, 90):
            with self.subTest(rotate=rotate):
                self.convert(photo, backend, rotate=rotate)
                self.assertEqual(backend.requests, [])
                self.assertEqual(self.figure_files(), first_figures)

        backend = PageBackend({1: figure})
        self.convert(photo, backend, rotate=270, ignore_saved_layouts=True)
        self.assertEqual(backend.requests, [1])
        self.assertEqual(json.loads(layout_file.read_text())["source"]["turn"], 270)
        self.assertNotEqual(self.figure_files(), first_figures)

    def test_saved_layout_has_the_turn_that_the_backend_found(self) -> None:
        class TurnedPageBackend(PageBackend):
            """Report a page whose top is at the right of the first image."""

            def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
                layout = super().recognize_layout(image_path, upright=upright)
                with Image.open(image_path) as image:
                    return layout if image.height > image.width else PageLayout(page_top="right")

        photo = self.photo(1)
        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=Box(0, 0, 500, 250))))
        backend = TurnedPageBackend({1: figure})
        self.convert(photo, backend)
        self.assertEqual(backend.requests, [1, 1])
        first_figures = self.figure_files()
        self.assertEqual(json.loads(self.layout_file(photo).read_text())["source"]["turn"], 270)

        backend = NoRequestBackend()
        self.convert(photo, backend)

        self.assertEqual(backend.requests, [])
        self.assertEqual(self.figure_files(), first_figures)

    def test_layout_from_other_settings_is_in_use_with_a_warning(self) -> None:
        # A person reads the pages that failed with a different model. The pages that
        # are complete must not get a request again.
        inputs = self.photos(3)
        first = PageBackend(settings={"model": "provider/model-a", "effort": "low"}, failing={2})
        with self.assertLogs(LOGGER_NAME, level="ERROR"), self.assertRaises(PageLayoutError):
            self.convert(inputs, first)

        second = PageBackend(settings={"model": "provider/model-b", "effort": "low"})
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            self.convert(inputs, second)

        self.assertEqual(second.requests, [2])
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:2 saved layout(s) are from other settings than those of "
                    "this conversion (model of 2 layout(s)). The document uses them. To get new "
                    "layouts, remove their files or use --ignore-saved-layouts"
                )
            ],
        )
        for number in (1, 2, 3):
            self.assertIn(text_of(number), self.document())

    def test_ignore_saved_layouts_sends_a_request_for_each_page(self) -> None:
        inputs = self.photos(2)
        self.convert(inputs, PageBackend())
        backend = PageBackend({1: PageLayout(blocks=(Paragraph("A new reading."),))})
        self.convert(inputs, backend, ignore_saved_layouts=True)
        self.assertEqual(sorted(backend.requests), [1, 2])
        self.assertIn("A new reading.", self.document())
        self.assertEqual(len(self.layout_files()), 2)
        # The new layout is the saved layout now.
        self.convert(inputs, NoRequestBackend())
        self.assertIn("A new reading.", self.document())

        with self.assertRaisesRegex(ValueError, "there is no saved layout to ignore"):
            self.convert(inputs, backend, ocr_output_directory=None, ignore_saved_layouts=True)

    def test_layout_file_that_is_not_usable_is_a_failure_and_stays(self) -> None:
        inputs = self.photos(2)
        self.convert(inputs, PageBackend())
        layout_file = self.layout_file(inputs[0])
        good = json.loads(layout_file.read_text(encoding="utf-8"))
        without_record = {name: value for name, value in good.items() if name != "source"}
        other_page = {**good, "source": {**good["source"], "sha256": "0" * 64}}
        unknown_field = {**good, "blocks": [{"type": "paragraph", "text": "a", "note": "b"}]}
        cases = {
            '{"source": ': "is not valid JSON: Expecting value at line 1, column 12",
            json.dumps(without_record): "has no usable 'source' record",
            json.dumps({**good, "source": {**good["source"], "record": 2}}): (
                "has no usable 'source' record"
            ),
            json.dumps(other_page): "is for a different input file or a different page of it",
            json.dumps(unknown_field): (
                "is not usable: block 1 (paragraph) has text in a field that the layout does not read"
            ),
            json.dumps({**good, "page_top": "left"}): "page_top must be 'top' in a saved layout",
        }
        for text, message in cases.items():
            with self.subTest(message=message):
                layout_file.write_text(text, encoding="utf-8")
                backend = NoRequestBackend()
                with (
                    self.assertLogs(LOGGER_NAME, level="ERROR"),
                    self.assertRaises(PageLayoutError) as raised,
                ):
                    self.convert(inputs, backend)
                error = str(raised.exception)
                self.assertIn("1 of 2 page(s) have no layout, so there is no document:", error)
                self.assertIn(f"page 1 ({inputs[0]}): the saved layout {layout_file} ", error)
                self.assertIn(message, error)
                self.assertIn("Correct the file, or remove it to get a new layout", error)
                # No request replaces a file that can have the work of a person.
                self.assertEqual(backend.requests, [])
                self.assertEqual(layout_file.read_text(encoding="utf-8"), text)

        # Without the file, the page gets a new layout.
        layout_file.unlink()
        backend = PageBackend()
        self.convert(inputs, backend)
        self.assertEqual(backend.requests, [1])

    def test_layout_that_scanwich_cannot_read_again_is_not_saved(self) -> None:
        # A backend can return a layout as an object, without the check of the JSON
        # form. The next conversion reads the file with that check.
        text = Paragraph("Text.")
        layouts = {
            "text has a control character": PageLayout(blocks=(Paragraph("a \x08 b"),)),
            "box must use the 0-to-1000 grid of the image, not its pixels": PageLayout(
                blocks=(text, Figure(box=Box(100, 100, 1500, 2000)))
            ),
            "has a mark and no text": PageLayout(blocks=(text, Footnote("", mark="1"))),
            "position must be one of": PageLayout(
                blocks=(text, PageFurniture("page_number", "12", "outer"))
            ),
        }
        inputs = self.photos(2)
        for message, layout in layouts.items():
            backend = PageBackend({2: layout})
            with (
                self.subTest(message=message),
                self.assertLogs(LOGGER_NAME, level="ERROR"),
                self.assertRaises(PageLayoutError) as raised,
            ):
                self.convert(inputs, backend, ignore_saved_layouts=True)
            error = str(raised.exception)
            self.assertIn(
                f"page 2 ({inputs[1]}): the backend gave a layout for page 2 that the check "
                "of a layout file refuses: block ",
                error,
            )
            self.assertIn(message, error)
            # The page has no file, so the next conversion sends its request again.
            self.assertEqual(self.layout_files(), [self.layout_file(inputs[0]).name])
            self.assertFalse(self.output.exists())

    def test_document_is_from_the_layout_that_the_file_gives(self) -> None:
        # The check of a layout file can change a layout that a backend returns as an
        # object. The first conversion and each later one must give the same document.
        layout = PageLayout(
            blocks=(
                Heading("A heading of level 5", level=5),
                Paragraph("  Text with spaces at its ends.  "),
                # The document writer does not know this alignment. The check of a layout
                # file reads it as the usual alignment.
                Paragraph("A paragraph with a different alignment.", align="justify"),
                Table(rows=((TableCell("a"), TableCell("b")), (TableCell("c"),))),
            )
        )
        inputs = self.photos(1)
        self.convert(inputs, PageBackend({1: layout}))
        first_document = self.document()

        backend = NoRequestBackend()
        self.convert(inputs, backend)

        self.assertEqual(backend.requests, [])
        self.assertEqual(self.document(), first_document)
        self.assertIn("Text with spaces at its ends.", first_document)

        # With no layout directory, the layout gets the same check, and the document is
        # the same. Before, the alignment stopped the document writer with a `KeyError`.
        self.convert(inputs, PageBackend({1: layout}), ocr_output_directory=None)
        self.assertEqual(self.document(), first_document)

    def test_directory_that_scanwich_cannot_write_to_stops_before_the_first_request(self) -> None:
        # The first file of the layout directory comes after the requests of a page.
        inputs = self.photos(2)
        a_file = self.directory / "a-file"
        a_file.write_text("not a directory")
        read_only = self.directory / "read-only"
        read_only.mkdir(mode=0o500)
        self.addCleanup(read_only.chmod, 0o700)
        cases = {
            "the layout directory is a file": {"ocr_output_directory": a_file},
            "the layout directory is in a file": {"ocr_output_directory": a_file / "layouts"},
            "the directory of the document is in a file": {"output": a_file / "out" / "book.tex"},
        }
        # The system administrator can write to each directory.
        if os.geteuid() != 0:
            cases["the layout directory is read-only"] = {"ocr_output_directory": read_only}
            cases["a new layout directory in a read-only directory"] = {
                "ocr_output_directory": read_only / "layouts"
            }
            cases["the directory of the document is read-only"] = {"output": read_only / "book.tex"}
        for name, options in cases.items():
            backend = PageBackend()
            output = options.pop("output", self.output)
            options.setdefault("ocr_output_directory", self.layouts)
            is_layout_directory = "layout" in name.split(" is ")[0]
            with (
                self.subTest(case=name),
                self.assertRaisesRegex(
                    OutputDirectoryError,
                    "Scanwich cannot write to the "
                    + ("layout directory" if is_layout_directory else "directory of the document")
                    + " /",
                ),
            ):
                convert_to_latex(inputs, output, backend=backend, **options)
            self.assertEqual(backend.requests, [])
        self.assertEqual(list(read_only.iterdir()), [])
        self.assertEqual(a_file.read_text(), "not a directory")

    def test_read_only_layout_directory_with_each_layout_needs_no_write(self) -> None:
        # A person makes the layout directory read-only, for example to keep the layouts
        # that they corrected. A new run with a layout for each page sends no request.
        if os.geteuid() == 0:
            self.skipTest("the system administrator can write to each directory")
        inputs = self.photos(3)
        self.convert(inputs, PageBackend())
        first_document = self.document()
        self.layouts.chmod(0o500)
        self.addCleanup(self.layouts.chmod, 0o700)

        backend = NoRequestBackend()
        self.convert(inputs, backend)
        self.assertEqual(backend.requests, [])
        self.assertEqual(self.document(), first_document)

        # A page that needs a request needs the directory. The check comes before the
        # first request.
        backend = PageBackend()
        with self.assertRaisesRegex(
            OutputDirectoryError, "Scanwich cannot write to the layout directory"
        ):
            self.convert([*inputs, self.photo(4)], backend)
        self.assertEqual(backend.requests, [])
        self.assertEqual(self.document(), first_document)

    def test_two_writers_of_one_layout_file_do_not_fail(self) -> None:
        # Two conversions can have one layout directory, for example a document with
        # pages and a document with `--flow` from the same files.
        inputs = self.photos(1)
        self.convert(inputs, PageBackend())
        (page,) = pipeline._source_pages(inputs, inputs)
        layout_file = self.layout_file(inputs[0])
        record = pipeline._source_record_from_json(
            json.loads(layout_file.read_text(encoding="utf-8"))["source"]
        )
        assert record is not None
        layout = PageLayout(blocks=(Paragraph(text_of(1)),))
        errors: list[Exception] = []

        def write() -> None:
            for _ in range(150):
                try:
                    pipeline._write_layout_file(self.layouts, page, record, layout)
                except Exception as error:
                    errors.append(error)

        threads = [threading.Thread(target=write) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        # The file is complete, and no temporary file stays.
        self.assertEqual(self.layout_files(), [layout_file.name])
        self.assertEqual(
            json.loads(layout_file.read_text(encoding="utf-8"))["blocks"][0]["text"], text_of(1)
        )

    def test_write_that_fails_leaves_no_temporary_file(self) -> None:
        inputs = self.photos(2)
        replace = Path.replace

        def full_disk(source: Path, target: Path) -> Path:
            if source.name.endswith(".scanwich-tmp") and target.suffix == ".json":
                raise OSError(28, "No space left on device", str(target))
            return replace(source, target)

        with (
            patch.object(Path, "replace", full_disk),
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaisesRegex(PageLayoutError, "No space left on device"),
        ):
            self.convert(inputs, PageBackend())
        # No part of a layout file stays, and the directory has no file at all.
        self.assertEqual([path.name for path in self.layouts.iterdir()], [])

    def test_two_layout_files_with_one_record_give_a_warning(self) -> None:
        inputs = self.photos(2)
        self.convert(inputs, PageBackend())
        layout_file = self.layout_file(inputs[1])
        shutil.copy(layout_file, self.layouts / "copy-b.json")
        layout_file.rename(self.layouts / "copy-a.json")

        backend = PageBackend()
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            self.convert(inputs, backend)

        self.assertEqual(backend.requests, [])
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:2 layout files are for page 2 (copy-a.json, "
                    f"copy-b.json), and none has the usual name {layout_file.name}. The "
                    "conversion uses copy-a.json. Remove the files that you do not want"
                )
            ],
        )
        # One file with the record and a different name is the usual case of an input
        # file with a new name. It gives no warning.
        (self.layouts / "copy-b.json").unlink()
        with self.assertNoLogs(LOGGER_NAME, level="WARNING"):
            self.convert(inputs, backend)

    def test_other_files_of_the_layout_directory_stay(self) -> None:
        self.layouts.mkdir()
        other_files = {
            "page-000001.json": '[{"text": "regions of a searchable PDF"}]',
            "notes.json": "not JSON",
            "list.json": '{"source": "a text"}',
        }
        for name, text in other_files.items():
            (self.layouts / name).write_text(text, encoding="utf-8")
        inputs = self.photos(1)
        for _ in range(2):
            self.convert(inputs, PageBackend())
        self.assertEqual(
            self.layout_files(), sorted([*other_files, self.layout_file(inputs[0]).name])
        )
        for name, text in other_files.items():
            self.assertEqual((self.layouts / name).read_text(encoding="utf-8"), text)


class FailedPageTests(BookTestCase):
    def test_one_failed_page_does_not_stop_the_other_pages(self) -> None:
        inputs = self.photos(5)
        self.output.parent.mkdir()
        self.output.write_text("the earlier document")
        backend = PageBackend(failing={3})
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR") as captured_logs,
            self.assertRaises(PageLayoutError) as raised,
        ):
            self.convert(inputs, backend, parallel_pages=1)

        # The pages after the failed page have their requests.
        self.assertEqual(backend.requests, [1, 2, 3, 4, 5])
        self.assertEqual(
            str(raised.exception),
            "1 of 5 page(s) have no layout, so there is no document:\n"
            f"  page 3 ({inputs[2]}): the service did not read page 3\n"
            f"The layouts of the pages that are complete are in {self.layouts.resolve()}. "
            "The same command again sends requests only for the pages of this list. Read the "
            "error of each page first: a page that fails again needs a correction of its "
            "cause.",
        )
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"ERROR:{LOGGER_NAME}:Page 3/5 ({inputs[2]}) has no layout: the service did "
                    "not read page 3"
                )
            ],
        )
        # A page does not disappear: there is no new document.
        self.assertEqual(self.output.read_text(), "the earlier document")
        self.assertEqual(
            self.layout_files(),
            sorted(self.layout_file(inputs[index]).name for index in (0, 1, 3, 4)),
        )

        # The new run sends a request only for the page that failed.
        backend = PageBackend()
        self.convert(inputs, backend)
        self.assertEqual(backend.requests, [3])
        places = [self.document().index(text_of(number)) for number in range(1, 6)]
        self.assertEqual(places, sorted(places))

    def test_conversion_that_saves_no_layouts_stops_at_the_first_failed_page(self) -> None:
        # Each more page has a cost, and a new run sends its request again.
        backend = PageBackend(failing={2})
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
            self.assertRaises(RuntimeError) as raised,
        ):
            self.convert(self.photos(4), backend, ocr_output_directory=None, parallel_pages=1)

        # The error is the error of the backend.
        self.assertNotIsInstance(raised.exception, PageLayoutError)
        self.assertEqual(str(raised.exception), "the service did not read page 2")
        self.assertEqual(backend.requests, [1, 2])
        # The first record is the warning at the start of the conversion.
        self.assertEqual(
            captured_logs.output[1:],
            [
                (
                    f"WARNING:{LOGGER_NAME}:Page 2 failed, and the conversion saves no layouts: "
                    "the layouts of 1 page(s) are lost. With --ocr-output-dir, a new run sends "
                    "requests only for the pages that have no saved layout"
                )
            ],
        )
        self.assertFalse(self.output.exists())

    def test_pages_that_the_service_answered_do_not_stop_the_conversion(self) -> None:
        # 3 bad pages at the start of a book, for example a cover and two plates. The
        # service works, so the other pages get their requests.
        inputs = self.photos(8)
        for run in (1, 2, 3):
            backend = PageBackend(failing={1, 2, 3})
            with (
                self.subTest(run=run),
                self.assertLogs(LOGGER_NAME, level="ERROR"),
                self.assertRaises(PageLayoutError) as raised,
            ):
                self.convert(inputs, backend, parallel_pages=1)
            # Each run after the first one sends requests only for the 3 bad pages.
            self.assertEqual(backend.requests, [1, 2, 3, 4, 5, 6, 7, 8] if run == 1 else [1, 2, 3])
            self.assertEqual(len(self.layout_files()), 5)
            error = str(raised.exception)
            self.assertIn("3 of 8 page(s) have no layout, so there is no document:", error)
            self.assertNotIn("did not start", error)
            self.assertTrue(
                error.endswith(
                    f"The layouts of the pages that are complete are in {self.layouts}. The "
                    "same command again sends requests only for the pages of this list. Read "
                    "the error of each page first: a page that fails again needs a correction "
                    "of its cause."
                ),
                error,
            )

    def test_failure_with_no_saved_layouts_cancels_the_pages_in_work(self) -> None:
        # The conversion has no result then, so each more request is only a cost.
        # Page 1 fails while the requests of 3 other pages are in work.
        backend = AsyncPageBackend(failing={1}, delay=0.4, delays={1: 0.1})

        def too_long(*arguments: object) -> None:
            raise AssertionError("the conversion did not end after the failure of a page")

        # Without the cancel of its tasks, the conversion waits with no end. The test
        # must then fail, and not wait.
        handler = signal.signal(signal.SIGALRM, too_long)
        self.addCleanup(signal.signal, signal.SIGALRM, handler)
        signal.alarm(10)
        self.addCleanup(signal.alarm, 0)
        started = time.monotonic()
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(RuntimeError, "the service did not read page 1"),
        ):
            self.convert(self.photos(12), backend, ocr_output_directory=None, parallel_pages=4)

        # The 3 other pages in work do not come to their end, and no other page starts.
        self.assertEqual(backend.finished, [1])
        self.assertEqual(sorted(backend.requests), [1, 2, 3, 4])
        self.assertEqual(backend.in_flight, 0)
        # Their requests ended before the backend closed its connections.
        self.assertEqual((backend.closed, backend.in_flight_at_close), (1, 0))
        self.assertLess(time.monotonic() - started, 0.4)

    def test_conversion_starts_no_more_pages_when_the_service_gives_no_reply(self) -> None:
        # A service that stops in the middle of a book. Without this rule, each later
        # page waits for its timeout.
        self.assertEqual(NO_REPLY_WAVES, 3)
        inputs = self.photos(12)
        made: list[int] = []
        make_page_image = pipeline._make_page_image

        def counting(page: object, *arguments: object, **options: object) -> None:
            made.append(page.number)  # type: ignore[attr-defined]
            make_page_image(page, *arguments, **options)

        backend = PageBackend(no_reply=range(5, 13))
        with (
            patch.object(pipeline, "_make_page_image", counting),
            self.assertLogs(LOGGER_NAME, level="ERROR") as captured_logs,
            self.assertRaises(PageLayoutError) as raised,
        ):
            self.convert(inputs, backend, parallel_pages=1)

        self.assertEqual(backend.requests, [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(self.layout_files()), 4)
        self.assertEqual(
            str(raised.exception),
            "8 of 12 page(s) have no layout, so there is no document:\n"
            f"  page 5 ({inputs[4]}): the service gave no reply for page 5\n"
            f"  page 6 ({inputs[5]}): the service gave no reply for page 6\n"
            f"  page 7 ({inputs[6]}): the service gave no reply for page 7\n"
            "  5 page(s) did not start: 8-12\n"
            "The conversion started no more pages: the pages in work got no reply from the "
            "service 3 times in sequence. The cause is probably the service, the key, the "
            "quota, or an option.\n"
            f"The layouts of the pages that are complete are in {self.layouts}. When the "
            "service works again, the same command sends requests only for the pages that "
            "have no saved layout. If the same pages fail again, they are the cause: convert "
            "the other files first, with the same layout directory.",
        )
        self.assertEqual(
            captured_logs.output[-1],
            f"ERROR:{LOGGER_NAME}:The pages in work got no reply from the service 3 times in "
            "sequence, so the conversion starts no more pages",
        )
        # After the stop, the conversion makes no more page images: not more than the
        # pages that were in the queue.
        self.assertLessEqual(max(made), 7 + PAGE_QUEUE_SIZE + 1)

        # The same rule at the start of a run: the first 4 pages have saved layouts.
        backend = PageBackend(no_reply=range(5, 13))
        with self.assertLogs(LOGGER_NAME, level="ERROR"), self.assertRaises(PageLayoutError):
            self.convert(inputs, backend, parallel_pages=1)
        self.assertEqual(backend.requests, [5, 6, 7])

        # When the service gives replies again, the same command completes the book.
        backend = PageBackend()
        self.convert(inputs, backend)
        self.assertEqual(sorted(backend.requests), [5, 6, 7, 8, 9, 10, 11, 12])

    def test_page_with_a_layout_starts_the_count_of_pages_with_no_reply_again(self) -> None:
        backend = PageBackend(no_reply={1, 2, 4, 5, 7, 8})
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaisesRegex(PageLayoutError, "6 of 9 page.s. have no layout") as raised,
        ):
            self.convert(self.photos(9), backend, parallel_pages=1)
        self.assertEqual(backend.requests, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertNotIn("did not start", str(raised.exception))

    def test_page_with_a_reply_that_is_not_usable_does_not_count(self) -> None:
        # It also does not start the count again: only a page with a layout does that.
        backend = PageBackend(no_reply={1, 3, 5}, failing={2, 4})
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaisesRegex(PageLayoutError, "3 page.s. did not start: 6-8"),
        ):
            self.convert(self.photos(8), backend, parallel_pages=1)
        self.assertEqual(backend.requests, [1, 2, 3, 4, 5])

    def test_service_that_stops_fails_only_the_pages_in_work(self) -> None:
        readers = 4
        backend = AsyncPageBackend(no_reply=range(9, 41), delay=0.01)
        with self.assertLogs(LOGGER_NAME, level="ERROR"), self.assertRaises(PageLayoutError):
            self.convert(self.photos(40), backend, parallel_pages=readers)
        self.assertEqual(len(self.layout_files()), 8)
        # Each wave is not more than the pages in work. After the third wave, the pages
        # in work have their requests, and no more.
        self.assertLessEqual(len(backend.requests), 8 + (NO_REPLY_WAVES + 1) * readers)

    def test_one_outage_of_the_service_counts_one_time(self) -> None:
        # The pages in work at one moment fail together. A short outage must not stop a
        # run: 3 or 4 pages with no reply are one wave.
        outage = (0.25, 0.4)
        started = time.monotonic()

        class OutageBackend(AsyncPageBackend):
            async def recognize_layout_async(
                self, image_path: Path, *, upright: bool = False
            ) -> PageLayout:
                if outage[0] <= time.monotonic() - started < outage[1]:
                    page = self.start(image_path, upright)
                    # A client that tries a request again needs some time to fail.
                    await asyncio.sleep(0.2)
                    self.in_flight -= 1
                    raise NoReplyError(f"the service gave no reply for page {page}")
                return await super().recognize_layout_async(image_path, upright=upright)

        backend = OutageBackend(delay=0.1)
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaises(PageLayoutError) as raised,
        ):
            self.convert(self.photos(24), backend, parallel_pages=4)

        error = str(raised.exception)
        failed = error.count("the service gave no reply")
        self.assertGreater(failed, 1)
        # Each page started and each other page has its layout.
        self.assertNotIn("did not start", error)
        self.assertNotIn("got no reply from the service 3 times", error)
        self.assertEqual(len(self.layout_files()), 24 - failed)

    def test_slow_pages_at_the_same_time_count_one_time(self) -> None:
        # A page that needs more time than the timeout of the service. Three such pages
        # in work at one moment are one wave.
        class SlowBackend(AsyncPageBackend):
            async def recognize_layout_async(
                self, image_path: Path, *, upright: bool = False
            ) -> PageLayout:
                page = self.start(image_path, upright)
                await asyncio.sleep(0.3 if page in (5, 6, 7) else 0.02)
                self.in_flight -= 1
                if page in (5, 6, 7):
                    raise NoReplyError(f"the request for page {page} timed out")
                return PageLayout(blocks=(Paragraph(text_of(page)),))

        for run in (1, 2):
            backend = SlowBackend()
            with (
                self.subTest(run=run),
                self.assertLogs(LOGGER_NAME, level="ERROR"),
                self.assertRaises(PageLayoutError) as raised,
            ):
                self.convert(self.photos(12), backend, parallel_pages=4)
            self.assertIn("3 of 12 page(s) have no layout", str(raised.exception))
            self.assertNotIn("did not start", str(raised.exception))
            self.assertEqual(len(self.layout_files()), 9)

    def test_stop_after_the_last_page_tells_no_pages_that_did_not_start(self) -> None:
        backend = PageBackend(no_reply={1, 2, 3})
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaises(PageLayoutError) as raised,
        ):
            self.convert(self.photos(3), backend, parallel_pages=1)
        error = str(raised.exception)
        self.assertNotIn("did not start", error)
        self.assertNotIn("started no more pages", error)
        self.assertIn(
            "The pages in work got no reply from the service 3 times in sequence. The cause is "
            "probably the service, the key, the quota, or an option.\n",
            error,
        )

    def test_conversion_starts_no_more_pages_when_its_first_pages_all_fail(self) -> None:
        # A service that gives only replies that Scanwich cannot use, for example after
        # a wrong model option. Without this rule, each page of a book has a cost.
        self.assertEqual(pipeline.FIRST_FAILED_PAGES, 10)
        inputs = self.photos(24)
        for readers in (1, 4):
            backend = PageBackend(failing=range(1, 25))
            layouts = self.directory / f"layouts-{readers}"
            with (
                self.subTest(readers=readers),
                self.assertLogs(LOGGER_NAME, level="ERROR") as captured_logs,
                self.assertRaises(PageLayoutError) as raised,
            ):
                self.convert(inputs, backend, parallel_pages=readers, ocr_output_directory=layouts)
            # The pages in work at the stop end, and no other page starts.
            self.assertGreaterEqual(len(backend.requests), 10)
            self.assertLessEqual(len(backend.requests), 10 + readers)
            error = str(raised.exception)
            self.assertIn(
                "The conversion started no more pages: the first 10 pages that needed a request "
                "failed, and no page has a layout. The cause is probably the model or an option, "
                "for example a model that does not read images.\n"
                "Correct the model or the option, and run the same command again. If these pages "
                "are the cause, convert the other files first, with the same layout directory: a "
                "layout directory with a saved layout turns this rule off.",
                error,
            )
            self.assertIn(
                f"ERROR:{LOGGER_NAME}:The first 10 pages that needed a request failed, and no "
                "page has a layout, so the conversion starts no more pages",
                captured_logs.output,
            )

    def test_book_with_difficult_first_pages_continues_with_saved_layouts(self) -> None:
        # The way around that the message gives. It must not bring back a stop at each run.
        inputs = self.photos(14)
        difficult = range(1, 11)
        with self.assertLogs(LOGGER_NAME, level="ERROR"), self.assertRaises(PageLayoutError):
            self.convert(inputs, PageBackend(failing=difficult), parallel_pages=1)
        self.assertEqual(self.layout_files(), [])

        # The other files first, with the same layout directory.
        self.convert(inputs[10:], PageBackend())
        backend = PageBackend(failing=difficult)
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaises(PageLayoutError) as raised,
        ):
            self.convert(inputs, backend, parallel_pages=1)
        self.assertEqual(backend.requests, list(difficult))
        self.assertIn("10 of 14 page(s) have no layout", str(raised.exception))
        self.assertNotIn("did not start", str(raised.exception))

        # A page with a layout from this run also turns the rule off.
        backend = PageBackend(failing=range(2, 13))
        with self.assertLogs(LOGGER_NAME, level="ERROR"), self.assertRaises(PageLayoutError):
            self.convert(self.photos(13), backend, parallel_pages=1, ignore_saved_layouts=True)
        self.assertEqual(len(backend.requests), 13)

    def test_failed_page_keeps_nothing_of_its_request(self) -> None:
        # The error of a failed page has its traceback, and the frames of a backend hold
        # the request with the page image: megabytes for each failed page of a book.
        class Request:
            """Stands for the image text of a request."""

        requests: list[weakref.ref[Request]] = []
        alive_at_the_last_page: list[bool] = []

        class HoldingBackend(AsyncPageBackend):
            async def recognize_layout_async(
                self, image_path: Path, *, upright: bool = False
            ) -> PageLayout:
                page = self.start(image_path, upright)
                self.in_flight -= 1
                if page in (1, 6):
                    if page == 6:
                        gc.collect()
                        alive_at_the_last_page.extend(ref() is not None for ref in requests)
                    return PageLayout(blocks=(Paragraph(text_of(page)),))
                request = Request()
                requests.append(weakref.ref(request))
                raise RuntimeError(f"the service did not read page {page}")

        with (
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaises(PageLayoutError) as raised,
        ):
            self.convert(self.photos(6), HoldingBackend(), parallel_pages=1)

        self.assertEqual(alive_at_the_last_page, [False, False, False, False])
        # The error of the conversion has the messages, and not the errors of the pages.
        self.assertIn("the service did not read page 2", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        gc.collect()
        self.assertEqual([ref() is not None for ref in requests], [False] * 4)

    def test_input_that_is_not_an_image_is_a_failed_page(self) -> None:
        notes = self.pages / "notes.txt"
        notes.write_text("not a page")
        inputs = [self.photo(1), notes, self.photo(3)]
        backend = PageBackend()
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR"),
            self.assertRaisesRegex(
                PageLayoutError, r"page 2 \(.*notes.txt\): .*is not a PDF and not an image"
            ),
        ):
            self.convert(inputs, backend)
        self.assertEqual(sorted(backend.requests), [1, 3])


class ParallelPagesTests(BookTestCase):
    def test_backend_contract_for_async_calls_is_an_addition(self) -> None:
        self.assertIsInstance(AsyncPageBackend(), LayoutBackend)
        self.assertIsInstance(AsyncPageBackend(), AsyncLayoutBackend)
        self.assertNotIsInstance(PageBackend(), AsyncLayoutBackend)
        self.assertNotIsInstance(AsyncPageBackend(), AsyncFigureLocator)

    def test_page_images_that_exist_at_one_time_have_a_limit(self) -> None:
        # The limit does not depend on the number of pages: one image for each reader,
        # the pages in the queue, and the one page that waits for a place in the queue.
        readers = 3
        limit = readers + PAGE_QUEUE_SIZE + 1
        self.assertEqual(limit, 5)
        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=RED_BOX)))
        for count in (10, 40):
            with self.subTest(pages=count):
                half = count // 2
                inputs = [
                    self.pdf(range(1, half + 1), f"part-{count}.pdf"),
                    *(self.photo(number) for number in range(half + 1, count + 1)),
                ]
                waiting: list[list[str]] = []
                in_system_directory: list[Path] = []

                def note_illustrations(
                    page: int,
                    waiting: list[list[str]] = waiting,
                    in_system_directory: list[Path] = in_system_directory,
                ) -> None:
                    # The illustrations of the pages that are complete.
                    waiting.append(
                        sorted(
                            path.name
                            for path in self.output.parent.glob(".book.tex.*.scanwich-tmp/*")
                        )
                    )
                    in_system_directory.extend(Path(tempfile.gettempdir()).rglob("*.jpg"))

                backend = AsyncPageBackend(
                    dict.fromkeys(range(1, count + 1, 4), figure),
                    delay=0.02,
                    on_request=note_illustrations,
                )
                with TemporaryDirectory(dir=self.directory) as work_directory:
                    with patch("tempfile.tempdir", work_directory):
                        self.convert(
                            inputs,
                            backend,
                            dpi=72,
                            parallel_pages=readers,
                            ocr_output_directory=self.directory / f"layouts-{count}",
                        )
                    # No page image and no work directory stays.
                    self.assertEqual(list(Path(work_directory).iterdir()), [])

                self.assertEqual(sorted(backend.requests), list(range(1, count + 1)))
                self.assertEqual(backend.max_in_flight, readers)
                self.assertLessEqual(backend.max_page_images, limit)
                # The page images of the readers exist at the same time.
                self.assertGreaterEqual(backend.max_page_images, readers)
                self.assertEqual(self.document().count("\\begin{sourcepage}"), count)
                self.assertEqual(len(self.figure_files()), len(range(1, count + 1, 4)))
                self.assertEqual(backend.closed, 1)
                # The illustrations are many in sum, and they stay until the end. They
                # wait next to the document, because the temporary directory of a system
                # can be in memory.
                self.assertIn("page-0001-figure-01.jpg", waiting[-1])
                self.assertEqual(in_system_directory, [])
                self.assertEqual(
                    sorted(path.name for path in self.output.parent.iterdir()),
                    ["book-figures", "book.tex"],
                )

    def test_page_images_have_the_same_limit_with_saved_layouts(self) -> None:
        # A page with a saved layout and an illustration needs its page image for the crop.
        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=RED_BOX)))
        inputs = self.photos(30)
        self.convert(inputs, PageBackend(dict.fromkeys(range(1, 31), figure)))
        first_figures = self.figure_files()
        self.assertEqual(len(first_figures), 30)

        counts: list[int] = []
        work_directory = self.directory / "work"
        work_directory.mkdir()

        def counting_crop(layout: PageLayout, page_image: Path, *args: object, **options: object):
            counts.append(len(list(page_image.parent.glob("*.png"))))
            time.sleep(0.005)
            return crop_figures(layout, page_image, *args, **options)

        crop_figures = pipeline._crop_figures
        backend = NoRequestBackend()
        with (
            patch("tempfile.tempdir", str(work_directory)),
            patch.object(pipeline, "_crop_figures", counting_crop),
        ):
            self.convert(inputs, backend, parallel_pages=2)

        self.assertEqual(backend.requests, [])
        self.assertEqual(len(counts), 30)
        self.assertLessEqual(max(counts), 2 + PAGE_QUEUE_SIZE + 1)
        self.assertEqual(self.figure_files(), first_figures)
        self.assertEqual(list(work_directory.iterdir()), [])

    def test_document_does_not_depend_on_the_sequence_of_the_replies(self) -> None:
        count = 6
        inputs = [self.pdf([1, 2]), *(self.photo(number) for number in range(3, count + 1))]
        layouts = {
            1: PageLayout(blocks=(Paragraph("The page ends in the middle of a sen-"),)),
            2: PageLayout(
                blocks=(
                    Paragraph("tence, and continues", continued=True),
                    Figure(box=RED_BOX, caption="Fig. 1"),
                )
            ),
            3: PageLayout(blocks=(Paragraph("on the third page.", continued=True),)),
            5: PageLayout(blocks=(Figure(box=Box(0, 0, 500, 500)), Figure(box=RED_BOX))),
        }
        results = {}
        for readers in (1, count):
            # The first page has the last reply.
            backend = AsyncPageBackend(
                layouts, delays={number: 0.05 * (count - number) for number in range(1, count + 1)}
            )
            self.output = self.directory / f"out-{readers}" / "book.tex"
            self.convert(
                inputs,
                backend,
                dpi=72,
                parallel_pages=readers,
                ocr_output_directory=self.directory / f"layouts-{readers}",
            )
            self.assertEqual(backend.max_in_flight, readers)
            results[readers] = (self.document(), self.figure_files(), backend.finished)

        document, figures, finished = results[1]
        self.assertEqual(finished, [1, 2, 3, 4, 5, 6])
        parallel_document, parallel_figures, parallel_finished = results[count]
        # With one reader for each page, the reply of the last page is the first one.
        self.assertEqual(parallel_finished, [6, 5, 4, 3, 2, 1])
        self.assertEqual(parallel_document, document)
        self.assertEqual(parallel_figures, figures)
        self.assertEqual(
            sorted(figures),
            ["page-0002-figure-01.jpg", "page-0005-figure-01.jpg", "page-0005-figure-02.jpg"],
        )
        self.assertIn("The page ends in the middle of a sentence,\n", document)

    def test_default_is_more_than_one_page_and_one_page_is_possible(self) -> None:
        self.assertEqual(DEFAULT_PARALLEL_PAGES, 4)
        inputs = self.photos(8)
        backend = AsyncPageBackend(delay=0.02)
        self.convert(inputs, backend)
        self.assertEqual(backend.max_in_flight, 4)

        backend = AsyncPageBackend(delay=0.005)
        self.convert(inputs, backend, parallel_pages=1, ignore_saved_layouts=True)
        self.assertEqual(backend.max_in_flight, 1)
        self.assertEqual(backend.requests, [1, 2, 3, 4, 5, 6, 7, 8])

        for parallel_pages in (0, -1, True, 2.0):
            with (
                self.subTest(parallel_pages=parallel_pages),
                self.assertRaisesRegex(ValueError, "parallel pages must be 1 or more"),
            ):
                self.convert(inputs, backend, parallel_pages=parallel_pages)

    def test_work_for_the_processor_does_not_stop_the_event_loop(self) -> None:
        # Each kind of work that uses the processor or the disk is in a thread. A task
        # that needs the event loop at short intervals gets it while that work is in
        # progress.
        work_seconds = 0.3

        class TurnedFigureBackend(AsyncPageBackend):
            """Each page is not upright in its first image, and it has one illustration."""

            async def recognize_layout_async(
                self, image_path: Path, *, upright: bool = False
            ) -> PageLayout:
                await super().recognize_layout_async(image_path, upright=upright)
                with Image.open(image_path) as image:
                    if image.width > image.height:
                        return PageLayout(page_top="right")
                return PageLayout(blocks=(Paragraph("Text."), Figure(box=Box(250, 250, 750, 750))))

            async def locate_figure_async(
                self, image_path: Path, **hints: object
            ) -> FigureLocation:
                return FigureLocation(Box(100, 100, 900, 900))

            def locate_figure(self, image_path: Path, **hints: object) -> FigureLocation:
                raise AssertionError("a backend with the async methods gets no synchronous call")

        # The kinds of work, and True for a kind that only a conversion with saved
        # layouts has.
        kinds = {
            "_saved_layout_index": False,
            "_file_digest": False,
            "_make_page_image": False,
            "_turn_page_image": False,
            "_save_region": False,
            "_checked_layout": False,
            "_write_layout_file": False,
            "_crop_figures": False,
            "_read_saved_layout": True,
        }
        for name, with_saved_layouts in kinds.items():
            gaps: list[float] = []
            calls: list[str] = []

            class HeartbeatCalls(pipeline._BackendCalls):
                """Start a task of the event loop at the start of the conversion."""

                def __init__(self, backend: LayoutBackend, gaps: list[float] = gaps) -> None:
                    super().__init__(backend)
                    self._gaps = gaps
                    self._heartbeat = asyncio.create_task(self.beat())

                async def beat(self) -> None:
                    last = time.monotonic()
                    while True:
                        await asyncio.sleep(0.005)
                        self._gaps.append(time.monotonic() - last)
                        last = time.monotonic()

                async def close(self) -> None:
                    self._heartbeat.cancel()
                    await super().close()

            function = getattr(pipeline, name)

            def slow(
                *arguments: object,
                function: Callable[..., object] = function,
                calls: list[str] = calls,
                **options: object,
            ) -> object:
                # A wait in the thread of the event loop stops the loop for this time.
                calls.append(threading.current_thread().name)
                time.sleep(work_seconds)
                return function(*arguments, **options)

            with self.subTest(work=name):
                inputs = self.photos(2)
                layouts = self.directory / f"layouts-{name}"
                self.output = self.directory / f"out-{name}" / "book.tex"
                if with_saved_layouts:
                    self.convert(inputs, TurnedFigureBackend(), ocr_output_directory=layouts)
                with (
                    patch.object(pipeline, "_BackendCalls", HeartbeatCalls),
                    patch.object(pipeline, name, slow),
                ):
                    self.convert(inputs, TurnedFigureBackend(), ocr_output_directory=layouts)

                self.assertEqual(len(self.figure_files()), 2)
                # The work was in progress while the task ran, and not in its thread.
                self.assertGreater(len(calls), 0)
                self.assertNotIn(threading.main_thread().name, calls)
                self.assertGreater(len(gaps), 20)
                self.assertLess(max(gaps), work_seconds / 2)

    def test_region_files_of_box_requests_do_not_stay(self) -> None:
        # A box request has an image file with a part of the page.
        readers = 3
        counts: list[int] = []
        at_the_end: list[int] = []

        class RegionBackend(AsyncPageBackend):
            async def locate_figure_async(
                self, image_path: Path, **hints: object
            ) -> FigureLocation:
                await asyncio.sleep(0.01)
                counts.append(len(list(image_path.parent.glob("*.png"))))
                self.region_directory = image_path.parent
                return FigureLocation(Box(100, 100, 900, 900))

            def locate_figure(self, image_path: Path, **hints: object) -> FigureLocation:
                raise AssertionError("a backend with the async methods gets no synchronous call")

            async def aclose(self) -> None:
                # All pages are complete here, and the work directory is still there.
                at_the_end.append(len(list(self.region_directory.iterdir())))
                await super().aclose()

        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=Box(250, 250, 750, 750))))
        backend = RegionBackend(dict.fromkeys(range(1, 13), figure), delay=0.005)
        self.convert(self.photos(12), backend, parallel_pages=readers)

        self.assertEqual(len(counts), 12)
        # Not more than one region file for each page in work.
        self.assertLessEqual(max(counts), readers)
        self.assertEqual(at_the_end, [0])

    def test_backend_with_only_synchronous_methods_gets_one_call_at_a_time(self) -> None:
        # The pipeline does not know that such a backend permits calls from more than
        # one thread.
        backend = PageBackend(delay=0.01)
        self.convert(self.photos(8), backend, parallel_pages=4)
        self.assertEqual(sorted(backend.requests), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(backend.max_in_flight, 1)
        self.assertEqual(self.document().count("\\begin{sourcepage}"), 8)

    def test_backend_with_only_synchronous_methods_gets_its_calls_from_one_thread(self) -> None:
        # Such a backend can have an object that is for one thread only, for example a
        # database connection.
        class ThreadBackend(PageBackend):
            def __init__(self, *arguments: object, **options: object) -> None:
                super().__init__(*arguments, **options)  # type: ignore[arg-type]
                self.threads: list[int] = []
                self.thread_objects: list[threading.Thread] = []
                self.pages_of_the_log: list[str | None] = []

            def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
                self.threads.append(threading.get_ident())
                self.thread_objects.append(threading.current_thread())
                # A log record from this thread names its page.
                self.pages_of_the_log.append(pipeline.LOG_PAGE.get())
                return super().recognize_layout(image_path, upright=upright)

            def locate_figure(self, image_path: Path, **hints: object) -> FigureLocation:
                self.threads.append(threading.get_ident())
                return FigureLocation(Box(100, 100, 900, 900))

        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=Box(250, 250, 750, 750))))
        threads_before = threading.active_count()
        backend = ThreadBackend(dict.fromkeys(range(1, 9), figure), delay=0.002)
        conversions: list[object] = []

        class KeptConversion(pipeline._Conversion):
            """A conversion that stays in memory, so that its thread cannot end with it."""

            def __init__(self, **options: object) -> None:
                super().__init__(**options)  # type: ignore[arg-type]
                conversions.append(self)

        with patch.object(pipeline, "_Conversion", KeptConversion):
            self.convert(self.photos(8), backend, parallel_pages=4)

        self.assertEqual(len(backend.threads), 16)
        self.assertEqual(len(set(backend.threads)), 1)
        # The thread is not the thread of the caller, which made the backend.
        self.assertNotIn(threading.get_ident(), backend.threads)
        # The thread ends with the conversion.
        self.assertFalse(backend.thread_objects[0].is_alive())
        self.assertEqual(threading.active_count(), threads_before)
        self.assertEqual(
            sorted(backend.pages_of_the_log), [f"page {number}/8" for number in range(1, 9)]
        )

    def test_stop_waits_for_the_call_of_a_synchronous_provider(self) -> None:
        # The call of such a provider cannot get a cancel. The work files must stay until
        # it ends, and no thread of the conversion stays after the stop.
        calls: list[str] = []

        class SlowCallBackend(PageBackend):
            def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
                layout = super().recognize_layout(image_path, upright=upright)
                if len(self.requests) == 2:
                    calls.append("start")
                    signal.raise_signal(signal.SIGTERM)
                    time.sleep(0.3)
                    calls.append("end")
                return layout

        def handler_of_the_caller(*arguments: object) -> None:
            raise AssertionError("the conversion did not take the signal")

        self.addCleanup(
            signal.signal, signal.SIGTERM, signal.signal(signal.SIGTERM, handler_of_the_caller)
        )
        threads_before = threading.active_count()
        with self.assertRaises(ConversionStopped):
            self.convert(
                self.photos(4), SlowCallBackend(), parallel_pages=2, stop_signals=(signal.SIGTERM,)
            )
        self.assertEqual(calls, ["start", "end"])
        self.assertEqual(threading.active_count(), threads_before)
        self.assertFalse(self.output.parent.exists() and any(self.output.parent.iterdir()))

    def test_error_of_aclose_is_a_warning(self) -> None:
        class NotClosingBackend(AsyncPageBackend):
            async def aclose(self) -> None:
                raise OSError("the connection did not close")

        # A complete conversion gives its document.
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            self.convert(self.photos(2), NotClosingBackend(), ocr_output_directory=None)
        self.assertEqual(
            captured_logs.output[-1],
            f"WARNING:{LOGGER_NAME}:The backend did not close its connections (OSError: the "
            "connection did not close)",
        )
        self.assertIn(text_of(2), self.document())

        # The error of a page stays the error of the conversion.
        with (
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(RuntimeError, "the service did not read page 2"),
        ):
            self.convert(
                self.photos(3),
                NotClosingBackend(failing={2}),
                ocr_output_directory=None,
                parallel_pages=1,
            )

    def test_one_failed_page_does_not_cancel_the_pages_in_work(self) -> None:
        inputs = self.photos(8)
        # The failed page has the first reply, while 3 other pages are in work.
        backend = AsyncPageBackend(failing={2}, delay=0.05, delays={2: 0.0})
        with self.assertLogs(LOGGER_NAME, level="ERROR"), self.assertRaises(PageLayoutError):
            self.convert(inputs, backend, parallel_pages=4)

        self.assertEqual(backend.finished[0], 2)
        self.assertEqual(sorted(backend.finished), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(self.layout_files()), 7)
        self.assertEqual(backend.closed, 1)

    def test_stop_in_the_middle_keeps_the_layouts_that_are_saved(self) -> None:
        inputs = self.photos(10)
        self.output.parent.mkdir()
        self.output.write_text("the earlier document")
        work_directory = self.directory / "work"
        work_directory.mkdir()
        threads_before = threading.active_count()
        # A process in the background of a shell ignores the signal. The test needs the
        # handler that a command in the foreground has.
        handler = signal.signal(signal.SIGINT, signal.default_int_handler)
        self.addCleanup(signal.signal, signal.SIGINT, handler)

        def stop_at_page_6(page: int) -> None:
            if page == 6:
                # The signal of Ctrl-C.
                signal.raise_signal(signal.SIGINT)

        backend = AsyncPageBackend(delay=0.03, on_request=stop_at_page_6)
        with (
            patch("tempfile.tempdir", str(work_directory)),
            self.assertNoLogs("asyncio", level="WARNING"),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.convert(inputs, backend, parallel_pages=2)

        # The pages before the stop are saved. A page in work at the stop is not.
        saved = len(self.layout_files())
        self.assertIn(saved, (4, 5))
        self.assertLess(max(backend.requests), 8)
        self.assertEqual(self.output.read_text(), "the earlier document")
        # No task, no thread, and no file of the work stays.
        self.assertEqual(backend.in_flight, 0)
        self.assertEqual(backend.closed, 1)
        self.assertEqual(threading.active_count(), threads_before)
        self.assertEqual(list(work_directory.iterdir()), [])
        self.assertEqual(
            [path.name for path in self.layouts.iterdir() if path.name.startswith(".")], []
        )

        # The new run sends requests only for the pages that have no saved layout.
        backend = AsyncPageBackend()
        self.convert(inputs, backend)
        self.assertEqual(len(backend.requests), 10 - saved)
        self.assertEqual(self.document().count("\\begin{sourcepage}"), 10)

    def test_stop_signal_gives_the_same_clean_stop_as_ctrl_c(self) -> None:
        # The signal of a closed terminal and the signal of the system. Without a
        # handler, such a signal ends the process at once, and the work files stay.
        for number in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signal.Signals(number).name):
                inputs = self.photos(10)
                self.output = self.directory / f"out-{number}" / "book.tex"
                layouts = self.directory / f"layouts-{number}"
                work_directory = self.directory / f"work-{number}"
                work_directory.mkdir()
                threads_before = threading.active_count()

                def handler_of_the_caller(*arguments: object) -> None:
                    raise AssertionError("the conversion has the signal")

                handler = signal.signal(number, handler_of_the_caller)
                self.addCleanup(signal.signal, number, handler)

                def stop_at_page_6(page: int, number: int = number) -> None:
                    if page == 6:
                        signal.raise_signal(number)

                backend = AsyncPageBackend(delay=0.03, on_request=stop_at_page_6)
                with (
                    patch("tempfile.tempdir", str(work_directory)),
                    self.assertNoLogs("asyncio", level="WARNING"),
                    self.assertRaises(ConversionStopped) as raised,
                ):
                    self.convert(
                        inputs,
                        backend,
                        parallel_pages=2,
                        ocr_output_directory=layouts,
                        stop_signals=(signal.SIGTERM, signal.SIGHUP),
                    )

                self.assertEqual(raised.exception.signal_number, number)
                # Ctrl-C has the same type of error, so one handler is for the two.
                self.assertIsInstance(raised.exception, KeyboardInterrupt)
                saved = len(list(layouts.glob("*.json")))
                self.assertIn(saved, (4, 5))
                self.assertLess(max(backend.requests), 8)
                # No document, no hidden directory, no work file, no task, and no thread.
                self.assertEqual(list(self.output.parent.iterdir()), [])
                self.assertEqual(list(work_directory.iterdir()), [])
                self.assertEqual(backend.in_flight, 0)
                self.assertEqual(backend.closed, 1)
                self.assertEqual(threading.active_count(), threads_before)
                # The handler of the caller is in place again.
                self.assertIs(signal.getsignal(number), handler_of_the_caller)

                backend = AsyncPageBackend()
                self.convert(inputs, backend, ocr_output_directory=layouts)
                self.assertEqual(len(backend.requests), 10 - saved)

    def stopped_conversion(
        self, signals: list[int], *, at_page: int = 6, **patches: Callable[..., object]
    ) -> tuple[ConversionStopped, Path, AsyncPageBackend]:
        """Convert 10 photos, send the signals at a page, and return the stop.

        Each stop signal has a handler of the caller that fails the test, so that a
        signal that the conversion does not take ends no test process.
        """
        inputs = self.photos(10)
        self.output.parent.mkdir(exist_ok=True)
        self.output.write_text("the earlier document")
        work_directory = self.directory / "work"
        work_directory.mkdir(exist_ok=True)

        # The signals that the handler of the caller got during the conversion, and the
        # signals that had no handler. The conversion must take each signal.
        self.signals_of_the_caller: list[int] = []
        self.signals_with_no_handler: list[int] = []

        def handler_of_the_caller(number: int, frame: object) -> None:
            self.signals_of_the_caller.append(number)
            raise AssertionError("the conversion did not take the signal")

        self.handler_of_the_caller = handler_of_the_caller
        for number in (signal.SIGTERM, signal.SIGHUP):
            self.addCleanup(signal.signal, number, signal.signal(number, handler_of_the_caller))

        def send(page: int) -> None:
            if page == at_page:
                for number in signals:
                    if signal.getsignal(number) in (signal.SIG_DFL, None):
                        # The signal would end the process of the tests.
                        self.signals_with_no_handler.append(number)
                        continue
                    signal.raise_signal(number)

        backend = AsyncPageBackend(delay=0.03, on_request=send)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("tempfile.tempdir", str(work_directory)))
            for name, value in patches.items():
                stack.enter_context(patch.object(pipeline, name, value))
            stack.enter_context(self.assertNoLogs("asyncio", level="WARNING"))
            raised = stack.enter_context(self.assertRaises(ConversionStopped))
            self.convert(
                inputs, backend, parallel_pages=2, stop_signals=(signal.SIGTERM, signal.SIGHUP)
            )
        return raised.exception, work_directory, backend

    def assert_clean_stop(self, work_directory: Path) -> None:
        self.assertEqual(self.signals_of_the_caller, [])
        self.assertEqual(self.signals_with_no_handler, [])
        # No hidden directory and no temporary document next to the earlier document.
        self.assertEqual(self.output.read_text(), "the earlier document")
        self.assertEqual([path.name for path in self.output.parent.iterdir()], ["book.tex"])
        self.assertEqual(list(work_directory.iterdir()), [])
        # The handlers of the caller are in place again.
        for number in (signal.SIGTERM, signal.SIGHUP):
            self.assertIs(signal.getsignal(number), self.handler_of_the_caller)

    def test_two_signals_give_one_clean_stop(self) -> None:
        # A terminal that closes sends SIGHUP two times, in less than 10 ms. The second
        # signal must have no effect while the conversion stops.
        stop, work_directory, backend = self.stopped_conversion(
            [signal.SIGHUP, signal.SIGHUP, signal.SIGTERM]
        )
        self.assertEqual(stop.signal_number, signal.SIGHUP)
        self.assertEqual(backend.in_flight, 0)
        self.assert_clean_stop(work_directory)
        self.assertIn(len(self.layout_files()), (4, 5))

    def test_signal_in_the_parts_before_and_after_the_event_loop(self) -> None:
        # The index of the saved layouts and the document before its install have work
        # files also. A signal there must give the same clean stop.
        index = pipeline._saved_layout_index
        render = pipeline.render_latex

        def index_with_a_signal(*arguments: object) -> object:
            signal.raise_signal(signal.SIGTERM)
            # The index is in a thread. The signal cancels the conversion meanwhile.
            time.sleep(0.05)
            return index(*arguments)

        def render_with_a_signal(*arguments: object, **options: object) -> object:
            signal.raise_signal(signal.SIGTERM)
            return render(*arguments, **options)

        for name, value in (
            ("_saved_layout_index", index_with_a_signal),
            ("render_latex", render_with_a_signal),
        ):
            with self.subTest(part=name):
                stop, work_directory, backend = self.stopped_conversion([], **{name: value})
                self.assertEqual(stop.signal_number, signal.SIGTERM)
                self.assertFalse(stop.document_complete)
                self.assert_clean_stop(work_directory)
                if name == "_saved_layout_index":
                    self.assertEqual(backend.requests, [])
                else:
                    # All pages are complete and saved, and the next run needs no request.
                    self.assertEqual(len(self.layout_files()), 10)
                shutil.rmtree(work_directory)

    def test_signal_at_the_end_of_the_event_loop_gives_no_document(self) -> None:
        # The cancel of the conversion comes after its last step, so it has no effect.
        # The stop must still come before the document.
        run = pipeline._Conversion.run

        async def run_with_a_signal_at_its_end(conversion: object) -> None:
            await run(conversion)  # type: ignore[arg-type]
            signal.raise_signal(signal.SIGTERM)

        with patch.object(pipeline._Conversion, "run", run_with_a_signal_at_its_end):
            stop, work_directory, _ = self.stopped_conversion([])
        self.assertEqual(stop.signal_number, signal.SIGTERM)
        self.assertFalse(stop.document_complete)
        self.assert_clean_stop(work_directory)
        self.assertEqual(len(self.layout_files()), 10)

    def test_signal_in_the_install_gives_the_complete_new_result(self) -> None:
        # A stop between two renames of the install left the new document with the
        # earlier illustrations, or lost an earlier illustration. The install continues
        # to its end, and the stop comes after it.
        inputs = self.photos(4)
        earlier = PageLayout(blocks=(Paragraph("Earlier text."), Figure(box=RED_BOX)))
        new = PageLayout(blocks=(Paragraph("New text."), Figure(box=Box(0, 0, 1000, 1000))))
        self.convert(inputs, PageBackend(dict.fromkeys(range(1, 5), earlier)))
        earlier_result = self.directory / "earlier"
        shutil.copytree(self.output.parent, earlier_result)
        earlier_figures = self.figure_files()
        self.convert(
            inputs,
            PageBackend(dict.fromkeys(range(1, 5), new)),
            ocr_output_directory=self.directory / "new-layouts",
        )
        new_document, new_figures = self.document(), self.figure_files()
        self.assertNotEqual(new_figures, earlier_figures)
        work_directory = self.directory / "work"
        work_directory.mkdir()

        def handler_of_the_caller(number: int, frame: object) -> None:
            raise AssertionError("the conversion did not take the signal")

        for number in (signal.SIGTERM, signal.SIGHUP):
            self.addCleanup(signal.signal, number, signal.signal(number, handler_of_the_caller))
        # Ctrl-C has the usual handler of Python.
        self.addCleanup(
            signal.signal, signal.SIGINT, signal.signal(signal.SIGINT, signal.default_int_handler)
        )
        real_replace = Path.replace
        points: dict[str, Callable[[Path, Path], bool]] = {
            "the document": lambda source, target: Path(target).name == "book.tex",
            "an earlier illustration": (
                lambda source, target: str(target).endswith(pipeline.OLD_FIGURE_SUFFIX)
            ),
            "a new illustration": (
                lambda source, target: str(source).endswith(pipeline.NEW_FIGURE_SUFFIX)
            ),
        }
        for number in (signal.SIGTERM, signal.SIGINT):
            for point, is_the_point in points.items():
                with self.subTest(signal=signal.Signals(number).name, after=point):
                    shutil.rmtree(self.output.parent)
                    shutil.copytree(earlier_result, self.output.parent)
                    sent: list[int] = []

                    def replace(
                        source: Path,
                        target: Path,
                        is_the_point: Callable[[Path, Path], bool] = is_the_point,
                        number: int = number,
                        sent: list[int] = sent,
                    ) -> Path:
                        result = real_replace(source, target)
                        if not sent and is_the_point(source, target):
                            sent.append(number)
                            signal.raise_signal(number)
                        return result

                    try:
                        with (
                            patch.object(Path, "replace", replace),
                            patch("tempfile.tempdir", str(work_directory)),
                        ):
                            self.convert(
                                inputs,
                                NoRequestBackend(),
                                ocr_output_directory=self.directory / "new-layouts",
                                stop_signals=(signal.SIGTERM, signal.SIGHUP),
                            )
                    except KeyboardInterrupt as error:
                        # The test framework stops all tests for a `KeyboardInterrupt`.
                        stop = error
                    else:
                        self.fail("the conversion did not raise the stop")

                    self.assertEqual(sent, [number])
                    self.assertIsInstance(stop, ConversionStopped)
                    self.assertEqual(getattr(stop, "signal_number", None), number)
                    self.assertTrue(getattr(stop, "document_complete", False))
                    # The new document and all its illustrations, and no temporary file.
                    self.assertEqual(self.document(), new_document)
                    self.assertEqual(self.figure_files(), new_figures)
                    self.assertEqual(
                        sorted(path.name for path in self.output.parent.iterdir()),
                        ["book-figures", "book.tex"],
                    )
                    self.assertEqual(list(work_directory.iterdir()), [])
                    # The handlers of the caller are in place again.
                    for each in (signal.SIGTERM, signal.SIGHUP):
                        self.assertIs(signal.getsignal(each), handler_of_the_caller)
                    self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)

    def test_signal_that_the_caller_ignores_keeps_no_effect(self) -> None:
        # `nohup` gives SIG_IGN to SIGHUP, so that a long run continues after the
        # terminal closes.
        inputs = self.photos(6)
        self.addCleanup(signal.signal, signal.SIGHUP, signal.signal(signal.SIGHUP, signal.SIG_IGN))

        def send(page: int) -> None:
            if page == 3:
                signal.raise_signal(signal.SIGHUP)

        backend = AsyncPageBackend(delay=0.01, on_request=send)
        try:
            self.convert(inputs, backend, stop_signals=(signal.SIGTERM, signal.SIGHUP))
        except ConversionStopped:
            # The test framework stops all tests for a `KeyboardInterrupt`.
            self.fail("the conversion stopped for a signal that the caller ignores")

        self.assertEqual(sorted(backend.requests), [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.document().count("\\begin{sourcepage}"), 6)
        self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)

    def test_stop_signals_in_a_thread_that_has_no_signal_handlers(self) -> None:
        # Only the main thread has signal handlers. The conversion must still work.
        inputs = self.photos(2)
        backend = AsyncPageBackend()
        errors: list[BaseException] = []

        def convert() -> None:
            try:
                self.convert(inputs, backend, stop_signals=(signal.SIGTERM, signal.SIGHUP))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=convert)
        thread.start()
        thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(backend.requests), [1, 2])


class RateLimitTests(BookTestCase):
    def limited_backend(self, limited: Mapping[int, int], **options: object) -> AsyncPageBackend:
        """Return a backend whose service refuses the first requests of some pages."""
        left = dict(limited)

        def refuse(page: int) -> None:
            if left.get(page, 0) > 0:
                left[page] -= 1
                raise RateLimitError(f"too many requests at page {page}")

        return AsyncPageBackend(on_request=refuse, **options)

    def test_request_goes_again_after_a_wait(self) -> None:
        backend = self.limited_backend({2: 2})
        with (
            patch("scanwich.pipeline.RATE_LIMIT_WAITS", (0.01, 0.01, 0.01)),
            self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
        ):
            self.convert(self.photos(3), backend, parallel_pages=1)

        self.assertEqual(backend.requests, [1, 2, 2, 2, 3])
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:The service gets too many requests. All pages wait 0 "
                    "seconds before their next request, and then this request goes again"
                )
            ]
            * 2,
        )
        self.assertIn(text_of(2), self.document())

    def test_provider_continues_a_call_that_had_replies_before_the_refusal(self) -> None:
        # A new call of the method starts again at its first request, and each request
        # before the refusal had a reply with a cost.
        class ResumingBackend(AsyncPageBackend):
            resumed = 0

            async def recognize_layout_async(
                self, image_path: Path, *, upright: bool = False
            ) -> PageLayout:
                self.requests.append(1)
                raise RateLimitError("too many requests", resume=self.resume)

            async def resume(self) -> PageLayout:
                self.resumed += 1
                if self.resumed == 1:
                    raise RateLimitError("too many requests again", resume=self.resume)
                return PageLayout(blocks=(Paragraph("The reply after the refusal."),))

        backend = ResumingBackend()
        with (
            patch("scanwich.pipeline.RATE_LIMIT_WAITS", (0.01, 0.01, 0.01)),
            self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
        ):
            self.convert(self.photos(1), backend)

        self.assertEqual(backend.requests, [1])
        self.assertEqual(backend.resumed, 2)
        self.assertEqual(len(captured_logs.output), 2)
        self.assertIn("The reply after the refusal.", self.document())

    def test_all_pages_wait(self) -> None:
        wait = 0.2
        # The service refuses page 1 at once. Page 2 is then in work, and page 3 is not.
        backend = self.limited_backend({1: 1}, delays={2: 0.05})
        started = time.monotonic()
        with (
            patch("scanwich.pipeline.RATE_LIMIT_WAITS", (wait,)),
            self.assertLogs(LOGGER_NAME, level="WARNING"),
        ):
            self.convert(self.photos(3), backend, parallel_pages=2)

        self.assertEqual(sorted(backend.requests), [1, 1, 2, 3])
        # The request of page 3 does not start before the end of the wait.
        self.assertGreaterEqual(backend.started_at[3] - started, wait)

    def test_wait_that_the_service_asks_for(self) -> None:
        self.assertEqual(pipeline.MAX_RATE_LIMIT_WAIT, 300.0)
        self.assertEqual(pipeline.RATE_LIMIT_WAITS, (30.0, 60.0, 120.0))
        # The longer wait is in use, to the maximum.
        waits = {(30.0, None): 30.0, (30.0, 7.0): 30.0, (30.0, 45.0): 45.0, (60.0, 3600.0): 300.0}
        for (wait, retry_after), expected in waits.items():
            with self.subTest(wait=wait, retry_after=retry_after):
                self.assertEqual(pipeline._rate_limit_wait(wait, retry_after), expected)

        class AskingBackend(AsyncPageBackend):
            async def recognize_layout_async(
                self, image_path: Path, *, upright: bool = False
            ) -> PageLayout:
                if not self.requests:
                    self.requests.append(1)
                    raise RateLimitError("too many requests", retry_after=1.0)
                return await super().recognize_layout_async(image_path, upright=upright)

        # The service asks for 1.0 second, and the maximum is 0.2 seconds here. When the
        # maximum is not in use, the test fails after 1 second: it does not wait.
        backend = AskingBackend()
        started = time.monotonic()
        with (
            patch("scanwich.pipeline.MAX_RATE_LIMIT_WAIT", 0.2),
            patch("scanwich.pipeline.RATE_LIMIT_WAITS", (0.01,)),
            self.assertLogs(LOGGER_NAME, level="WARNING"),
        ):
            self.convert(self.photos(1), backend)

        self.assertGreaterEqual(backend.started_at[1] - started, 0.2)
        self.assertLess(backend.started_at[1] - started, 0.8)

    def test_page_fails_when_the_service_refuses_after_the_last_wait(self) -> None:
        inputs = self.photos(2)
        backend = self.limited_backend({2: 100})
        with (
            patch("scanwich.pipeline.RATE_LIMIT_WAITS", (0.01, 0.01, 0.01)),
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(
                PageLayoutError, r"page 2 \(.*photo-02.png\): too many requests at page 2"
            ),
        ):
            self.convert(inputs, backend, parallel_pages=1)
        # One request and one more after each wait.
        self.assertEqual(backend.requests, [1, 2, 2, 2, 2])

    def test_rate_refusal_after_its_waits_counts_as_no_reply(self) -> None:
        # A quota that ends in the middle of a book. Without this, each later page waits
        # 3 times before it fails.
        backend = self.limited_backend(dict.fromkeys(range(3, 13), 100))
        with (
            patch("scanwich.pipeline.RATE_LIMIT_WAITS", (0.01, 0.01)),
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(PageLayoutError, "4 page.s. did not start: 9-12") as raised,
        ):
            self.convert(self.photos(12), backend, parallel_pages=2)
        self.assertIn("got no reply from the service 3 times in sequence", str(raised.exception))
        self.assertEqual(len(self.layout_files()), 2)

    def test_refused_request_for_a_box_is_a_failed_page(self) -> None:
        # A layout with the first box of an illustration must not be the saved layout:
        # no later request corrects that box.
        class RefusedBoxBackend(AsyncPageBackend):
            boxes = 0

            async def locate_figure_async(self, image_path: Path, **hints: object) -> None:
                type(self).boxes += 1
                raise RateLimitError("too many requests for a box")

            def locate_figure(self, image_path: Path, **hints: object) -> None:
                raise AssertionError("a backend with the async methods gets no synchronous call")

        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=RED_BOX)))
        backend = RefusedBoxBackend({1: figure})
        self.assertIsInstance(backend, AsyncFigureLocator)
        with (
            patch("scanwich.pipeline.RATE_LIMIT_WAITS", (0.01,)),
            self.assertLogs(LOGGER_NAME, level="WARNING"),
            self.assertRaisesRegex(PageLayoutError, "too many requests for a box"),
        ):
            self.convert(self.photos(1), backend)
        self.assertEqual(RefusedBoxBackend.boxes, 2)
        self.assertEqual(self.layout_files(), [])
