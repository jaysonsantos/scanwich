"""LaTeX output for a book: many input files, saved layouts, and parallel pages."""

import asyncio
import hashlib
import json
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
from reportlab.pdfgen import canvas

from scanwich import pipeline
from scanwich.layout import Box, Figure, FigureLocation, PageLayout, Paragraph
from scanwich.models import OcrRegion
from scanwich.ocr import (
    AsyncFigureLocator,
    AsyncLayoutBackend,
    FigureLocator,
    LayoutBackend,
    LayoutSettings,
    RateLimitError,
)
from scanwich.pipeline import (
    DEFAULT_PARALLEL_PAGES,
    FIRST_FAILED_PAGES,
    PAGE_QUEUE_SIZE,
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
        delay: float = 0.0,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        self.layouts = dict(layouts or {})
        self.failing = set(failing)
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
                self.convert(inputs, PageBackend(), ocr_output_directory=None)


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
        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            self.convert(inputs, backend, dpi=72)

        self.assertEqual(backend.requests, [])
        self.assertIsInstance(backend, FigureLocator)
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
            "Correct the cause, and run the same command again: it sends requests only for "
            "the pages that have no saved layout.",
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
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:Page 2 failed, and the conversion saves no layouts: "
                    "the layouts of 1 page(s) are lost. With --ocr-output-dir, a new run sends "
                    "requests only for the pages that have no saved layout"
                )
            ],
        )
        self.assertFalse(self.output.exists())

    def test_conversion_starts_no_more_pages_when_its_first_pages_fail(self) -> None:
        # The cause is then probably the service, the key, or an option.
        self.assertEqual(FIRST_FAILED_PAGES, 3)
        inputs = self.photos(7)
        backend = PageBackend(failing=range(1, 8))
        with (
            self.assertLogs(LOGGER_NAME, level="ERROR") as captured_logs,
            self.assertRaises(PageLayoutError) as raised,
        ):
            self.convert(inputs, backend, parallel_pages=1)

        self.assertEqual(backend.requests, [1, 2, 3])
        error = str(raised.exception)
        self.assertIn("7 of 7 page(s) have no layout, so there is no document:", error)
        self.assertIn(f"  page 3 ({inputs[2]}): the service did not read page 3\n", error)
        self.assertIn(
            "  4 page(s) did not start, because the first 3 pages that needed a request "
            "failed: 4-7\n",
            error,
        )
        self.assertIn(
            f"ERROR:{LOGGER_NAME}:The first 3 pages that needed a request failed. The cause "
            "is probably not a page, so the conversion starts no more pages",
            captured_logs.output,
        )

    def test_conversion_continues_when_a_page_got_a_layout_before_the_failures(self) -> None:
        backend = PageBackend(failing=range(2, 8))
        with self.assertLogs(LOGGER_NAME, level="ERROR"), self.assertRaises(PageLayoutError):
            self.convert(self.photos(7), backend, parallel_pages=1)
        self.assertEqual(backend.requests, [1, 2, 3, 4, 5, 6, 7])

        # A page with a saved layout is not a page that got a layout from a request.
        backend = PageBackend(failing=range(2, 8))
        with self.assertLogs(LOGGER_NAME, level="ERROR"), self.assertRaises(PageLayoutError):
            self.convert(self.photos(7), backend, parallel_pages=1)
        self.assertEqual(backend.requests, [2, 3, 4])

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
        # The decode of a photo and the crops are in threads. A task that needs the event
        # loop at short intervals gets it while that work is in progress.
        work_seconds = 0.5
        gaps: list[float] = []

        class HeartbeatBackend(AsyncPageBackend):
            heartbeat: asyncio.Task[None] | None = None

            async def recognize_layout_async(
                self, image_path: Path, *, upright: bool = False
            ) -> PageLayout:
                if self.heartbeat is None:
                    self.heartbeat = asyncio.create_task(self.beat())
                return await super().recognize_layout_async(image_path, upright=upright)

            async def beat(self) -> None:
                last = time.monotonic()
                while True:
                    await asyncio.sleep(0.005)
                    gaps.append(time.monotonic() - last)
                    last = time.monotonic()

            async def aclose(self) -> None:
                assert self.heartbeat is not None
                self.heartbeat.cancel()
                await super().aclose()

        def slow(function: Callable[..., object]) -> Callable[..., object]:
            def work(*arguments: object, **options: object) -> object:
                # The thread keeps the processor busy for the loop only if the call is
                # in the thread of the loop.
                time.sleep(work_seconds)
                return function(*arguments, **options)

            return work

        figure = PageLayout(blocks=(Paragraph("Text."), Figure(box=RED_BOX)))
        backend = HeartbeatBackend(dict.fromkeys(range(1, 5), figure), delay=0.01)
        with (
            patch.object(pipeline, "_make_page_image", slow(pipeline._make_page_image)),
            patch.object(pipeline, "_crop_figures", slow(pipeline._crop_figures)),
        ):
            self.convert(self.photos(4), backend, parallel_pages=2)

        self.assertEqual(len(self.figure_files()), 4)
        # The work of 3 page images and 4 crops was in progress while the task ran.
        self.assertGreater(len(gaps), 50)
        self.assertLess(max(gaps), work_seconds / 2)

    def test_backend_with_only_synchronous_methods_gets_one_call_at_a_time(self) -> None:
        # The pipeline does not know that such a backend permits calls from more than
        # one thread.
        backend = PageBackend(delay=0.01)
        self.convert(self.photos(8), backend, parallel_pages=4)
        self.assertEqual(sorted(backend.requests), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(backend.max_in_flight, 1)
        self.assertEqual(self.document().count("\\begin{sourcepage}"), 8)

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
        class AskingBackend(AsyncPageBackend):
            async def recognize_layout_async(
                self, image_path: Path, *, upright: bool = False
            ) -> PageLayout:
                if not self.requests:
                    self.requests.append(1)
                    raise RateLimitError("too many requests", retry_after=1000.0)
                return await super().recognize_layout_async(image_path, upright=upright)

        backend = AskingBackend()
        started = time.monotonic()
        with (
            patch("scanwich.pipeline.MAX_RATE_LIMIT_WAIT", 0.2),
            patch("scanwich.pipeline.RATE_LIMIT_WAITS", (0.01,)),
            self.assertLogs(LOGGER_NAME, level="WARNING"),
        ):
            self.convert(self.photos(1), backend)

        # The wait is the wait of the service, and not more than the maximum.
        self.assertGreaterEqual(backend.started_at[1] - started, 0.2)
        self.assertLess(backend.started_at[1] - started, 5.0)

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
        self.assertFalse(self.layouts.exists())
