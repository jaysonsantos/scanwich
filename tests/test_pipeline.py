import errno
import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image, ImageChops
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from scanwich.layout import (
    Box,
    Figure,
    FigureLocation,
    Heading,
    PageFurniture,
    PageLayout,
    Paragraph,
)
from scanwich.models import OcrRegion, Point
from scanwich.ocr import FigureLocator, LayoutBackend, OcrBackend, backend_request_options
from scanwich.pipeline import (
    FIGURE_SEARCH_PAGE_REQUESTS,
    FIGURE_SEARCH_REQUESTS,
    PageLayoutError,
    convert_pdf,
    convert_to_latex,
    parse_backend_options,
)

LOGGER_NAME = "scanwich.pipeline"
RED = (255, 0, 0)
WHITE = (255, 255, 255)


class BackendOptionsTests(TestCase):
    def test_parses_json_values_and_plain_strings(self) -> None:
        self.assertEqual(
            parse_backend_options(
                ["gpu=true", 'readtext={"batch_size": 4}', "model_storage_directory=/models"]
            ),
            {
                "gpu": True,
                "readtext": {"batch_size": 4},
                "model_storage_directory": "/models",
            },
        )

    def test_rejects_option_without_equals(self) -> None:
        with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
            parse_backend_options(["gpu"])


class FakeBackend:
    def recognize(self, image_path: Path) -> list[OcrRegion]:
        return [
            OcrRegion(
                text=image_path.stem,
                polygon=(Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)),
            )
        ]


class BackendProtocolTests(TestCase):
    def test_backend_contract_is_provider_neutral(self) -> None:
        result = FakeBackend().recognize(Path("page-000001.png"))
        self.assertEqual(result[0].text, "page-000001")

    def test_backends_declare_their_own_request_options(self) -> None:
        self.assertEqual(backend_request_options("openai-compatible"), frozenset({"model"}))
        with self.assertRaisesRegex(ValueError, "unknown OCR backend"):
            backend_request_options("missing")

    def test_backends_without_a_declaration_take_no_request_options(self) -> None:
        def plugin_factory(*, languages: list[str], options: dict[str, object]) -> FakeBackend:
            return FakeBackend()

        entry_point = SimpleNamespace(
            name="plugin", value="plugin:factory", load=lambda: plugin_factory
        )
        with patch("scanwich.ocr.entry_points", return_value=[entry_point]):
            self.assertEqual(backend_request_options("plugin"), frozenset())


class ConversionTests(TestCase):
    def test_rasterizes_and_reassembles_pdf_with_backend_text(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_pdf = directory / "input.pdf"
            output_pdf = directory / "output.pdf"
            source = canvas.Canvas(str(input_pdf), pagesize=(144, 72))
            source.drawString(10, 50, "This becomes pixels")
            source.save()

            with self.assertLogs("scanwich.pipeline", level="INFO") as captured_logs:
                convert_pdf(input_pdf, output_pdf, backend=FakeBackend(), dpi=72)

            reader = PdfReader(output_pdf)
            self.assertEqual(len(reader.pages), 1)
            self.assertIn("page-000000", reader.pages[0].extract_text())
            progress = "\n".join(captured_logs.output)
            self.assertIn("Rasterized 1 page(s)", progress)
            self.assertIn("Recognizing page 1/1 with FakeBackend", progress)
            self.assertIn("Recognized 1 text region(s) on page 1", progress)
            self.assertIn(f"Wrote searchable PDF to {output_pdf.resolve()}", progress)


class FakeLayoutBackend:
    """Return the layouts in sequence, and record the page image of each request.

    The last layout repeats.
    """

    def __init__(self, *layouts: PageLayout) -> None:
        self._layouts = layouts
        self.image_sizes: list[tuple[int, int]] = []
        self.top_left_pixels: list[object] = []
        self.upright: list[bool] = []

    def recognize(self, image_path: Path) -> list[OcrRegion]:
        return []

    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        with Image.open(image_path) as image:
            self.image_sizes.append(image.size)
            self.top_left_pixels.append(image.getpixel((0, 0)))
        self.upright.append(upright)
        return self._layouts[min(len(self.image_sizes), len(self._layouts)) - 1]


class FailingLayoutBackend(FakeLayoutBackend):
    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        raise RuntimeError("the service is not available")


class LatexConversionTests(TestCase):
    def setUp(self) -> None:
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)

    def photo(self, name: str = "photo.png") -> Path:
        """Return a white image of 400x200 pixels with a red part at its top-right corner."""
        path = self.directory / name
        image = Image.new("RGB", (400, 200), WHITE)
        image.paste(RED, (300, 0, 400, 50))
        image.save(path)
        return path

    def test_backend_contract_for_layouts_is_an_addition(self) -> None:
        self.assertIsInstance(FakeLayoutBackend(), OcrBackend)
        self.assertIsInstance(FakeLayoutBackend(), LayoutBackend)
        self.assertNotIsInstance(FakeBackend(), LayoutBackend)

    def test_writes_one_document_for_all_pages_of_a_pdf(self) -> None:
        input_pdf = self.directory / "input.pdf"
        output_tex = self.directory / "out" / "book.tex"
        json_directory = self.directory / "layout"
        source = canvas.Canvas(str(input_pdf), pagesize=(144, 72))
        source.drawString(10, 50, "First page")
        source.showPage()
        source.setPageSize((72, 144))
        source.drawString(10, 50, "Second")
        source.save()
        first = PageLayout(
            blocks=(
                PageFurniture("page_number", "7"),
                Heading("A chapter"),
                Paragraph("The first page ends in the middle of a sen-"),
            )
        )
        second = PageLayout(blocks=(Paragraph("tence, and stops here.", continued=True),))
        backend = FakeLayoutBackend(first, second)

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            convert_to_latex(
                input_pdf, output_tex, backend=backend, dpi=72, ocr_output_directory=json_directory
            )

        # Each page image has the size of its page at the requested DPI.
        self.assertEqual(backend.image_sizes, [(144, 72), (72, 144)])
        document = output_tex.read_text(encoding="utf-8")
        # Each page of the PDF is one page of the document. The word that the page
        # boundary divides is complete on the first page.
        self.assertIn(
            "% Source page 1 (printed page number: 7)\n"
            "\\sourcepagehead{}{}{7}\n"
            "\\begin{sourcepage}{1}\n\n"
            "\\section*{A chapter}\n\n"
            "The first page ends in the middle of a sentence,\n"
            "% The page boundary of the source divides the last word of this paragraph. The\n"
            "% complete word is here, and the next page starts after it. Compare it with the "
            "source.\n\n"
            "\\end{sourcepage}\n\n"
            "% Source page 2\n"
            "\\begin{sourcepage}{2}\n\n"
            "\\noindent and stops here.\n\n"
            "\\end{sourcepage}\n\n"
            "\\end{document}\n",
            document,
        )
        self.assertEqual(sorted(path.name for path in output_tex.parent.iterdir()), ["book.tex"])
        self.assertEqual(
            [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(json_directory.iterdir())
            ],
            [first.to_json(), second.to_json()],
        )
        self.assertEqual(
            sorted(path.name for path in json_directory.iterdir()),
            ["page-000001.json", "page-000002.json"],
        )
        progress = "\n".join(captured_logs.output)
        self.assertIn("Rasterized 2 page(s)", progress)
        self.assertIn("Recognizing the layout of page 2/2 with FakeLayoutBackend", progress)
        self.assertIn("Recognized 3 block(s) on page 1", progress)
        self.assertIn("Wrote layout JSON for page 2", progress)
        self.assertIn(f"Wrote LaTeX document to {output_tex.resolve()}", progress)
        # The conversion tells where the signal for a page that is too high comes from.
        self.assertIn(
            "Each of the 2 source page(s) is one page of the document. A TeX engine writes a "
            "warning for a source page that it makes smaller to keep it on one page",
            progress,
        )
        self.assertNotIn("illustration", progress)

    def test_flow_gives_a_document_with_its_own_pages(self) -> None:
        output_tex = self.directory / "page.tex"
        first = PageLayout(blocks=(Paragraph("The page ends in a sen-"),))
        second = PageLayout(blocks=(Paragraph("tence.", continued=True),))
        input_pdf = self.directory / "input.pdf"
        source = canvas.Canvas(str(input_pdf), pagesize=(144, 72))
        source.showPage()
        source.showPage()
        source.save()

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            convert_to_latex(
                input_pdf, output_tex, backend=FakeLayoutBackend(first, second), dpi=72, flow=True
            )
        self.assertNotIn("one page of the document", "\n".join(captured_logs.output))

        document = output_tex.read_text(encoding="utf-8")
        # The paragraph is one paragraph, and the document has no box for a source page.
        self.assertIn("The page ends in a sen%\n% Source page 2\n", document)
        self.assertNotIn("sourcepage", document)

    def test_crops_the_illustrations_of_a_photo(self) -> None:
        # The name of the figure directory has only characters that are safe in LaTeX,
        # and a part of the hash of the document name.
        output_tex = self.directory / "my book (1).tex"
        name_hash = hashlib.sha256(b"my book (1).tex").hexdigest()[:8]
        figure_directory_name = f"my-book-1-{name_hash}-figures"
        json_directory = self.directory / "layout"
        layout = PageLayout(
            blocks=(
                Paragraph("Text before the illustration."),
                # The box of the red part of the photo.
                Figure(box=Box(750, 0, 1000, 250), caption="Fig. 1"),
                Figure(box=Box(0, 500, 500, 1000)),
            ),
            content_box=Box(0, 0, 1000, 1000),
        )
        backend = FakeLayoutBackend(layout)

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            convert_to_latex(
                self.photo(), output_tex, backend=backend, ocr_output_directory=json_directory
            )

        self.assertEqual(backend.image_sizes, [(400, 200)])
        figure_directory = self.directory / figure_directory_name
        self.assertEqual(
            sorted(path.name for path in figure_directory.iterdir()),
            ["page-0001-figure-01.jpg", "page-0001-figure-02.jpg"],
        )
        with Image.open(figure_directory / "page-0001-figure-01.jpg") as figure:
            # The crop has a small margin at the sides that are not the edge of the page.
            self.assertEqual((figure.format, figure.size), ("JPEG", (102, 51)))
            red, green, blue = figure.getpixel((60, 20))
            self.assertTrue(red > 240 and green < 20 and blue < 20, (red, green, blue))
        with Image.open(figure_directory / "page-0001-figure-02.jpg") as figure:
            self.assertEqual(figure.size, (202, 101))
        document = output_tex.read_text(encoding="utf-8")
        self.assertIn(
            "\\includegraphics[width=0.25\\linewidth,height=0.75\\textheight,keepaspectratio]"
            f"{{{figure_directory_name}/page-0001-figure-01.jpg}}\n"
            "\\par\\smallskip\nFig. 1\n\\end{center}",
            document,
        )
        self.assertIn(f"{{{figure_directory_name}/page-0001-figure-02.jpg}}", document)
        saved = json.loads((json_directory / "page-000001.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [block.get("image") for block in saved["blocks"]],
            [
                None,
                f"{figure_directory_name}/page-0001-figure-01.jpg",
                f"{figure_directory_name}/page-0001-figure-02.jpg",
            ],
        )
        progress = "\n".join(captured_logs.output)
        self.assertIn("The page image has 400x200 pixels", progress)
        self.assertIn(f"Wrote 2 illustration(s) to {figure_directory.resolve()}", progress)

    def test_turns_a_page_that_is_not_upright_and_recognizes_it_again(self) -> None:
        output_tex = self.directory / "page.tex"
        # The photo shows a page with its top at the right side. The red part is at the
        # top-left corner of that page.
        backend = FakeLayoutBackend(
            PageLayout(page_top="right"),
            PageLayout(blocks=(Paragraph("Upright text."), Figure(box=Box(0, 0, 250, 250)))),
        )

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            convert_to_latex(self.photo(), output_tex, backend=backend)

        self.assertEqual(backend.image_sizes, [(400, 200), (200, 400)])
        self.assertEqual(backend.top_left_pixels, [WHITE, RED])
        # The backend finds the direction in each of the two requests.
        self.assertEqual(backend.upright, [False, False])
        self.assertIn("Upright text.", output_tex.read_text(encoding="utf-8"))
        # The crop is from the page image that the backend got last.
        with Image.open(self.directory / "page-figures" / "page-0001-figure-01.jpg") as figure:
            red, green, blue = figure.getpixel((10, 10))
            self.assertTrue(red > 240 and green < 20 and blue < 20, (red, green, blue))
        self.assertIn(
            "The top of page 1 is at the right of its image; turning the image 270 degrees "
            "clockwise and recognizing the page again",
            "\n".join(captured_logs.output),
        )

    def test_fails_when_a_page_is_not_upright_after_the_turn(self) -> None:
        # A page must not disappear from the document without a failure.
        input_pdf = self.directory / "input.pdf"
        source = canvas.Canvas(str(input_pdf), pagesize=(144, 72))
        source.drawString(10, 50, "First page")
        source.showPage()
        source.drawString(10, 50, "Second page")
        source.save()
        output_tex = self.directory / "book.tex"
        good = PageLayout(blocks=(Paragraph("Text."),))
        for last_reply in (PageLayout(page_top="right"), replace(good, page_top="bottom")):
            with self.subTest(blocks=len(last_reply.blocks)):
                backend = FakeLayoutBackend(good, PageLayout(page_top="right"), last_reply)
                with self.assertRaisesRegex(
                    PageLayoutError,
                    "page 2 has no usable layout: the backend reported the top of the page at "
                    "the right of the image and, after a turn of 270 degrees clockwise, at the "
                    f"{last_reply.page_top}; set --rotate to give the direction of the pages",
                ):
                    convert_to_latex(input_pdf, output_tex, backend=backend, dpi=72)
                # One turn and one more request for the second page.
                self.assertEqual(backend.image_sizes, [(144, 72), (144, 72), (72, 144)])
                self.assertEqual(
                    sorted(path.name for path in self.directory.iterdir()), ["input.pdf"]
                )

    def test_rotate_gives_the_direction_of_each_page(self) -> None:
        backend = FakeLayoutBackend(PageLayout(blocks=(Paragraph("Text."),)))

        convert_to_latex(self.photo(), self.directory / "page.tex", backend=backend, rotate=270)

        self.assertEqual(backend.image_sizes, [(200, 400)])
        self.assertEqual(backend.top_left_pixels, [RED])
        # The backend does not look for the direction, so it reads the page in each case.
        self.assertEqual(backend.upright, [True])

    def test_rotate_keeps_the_blocks_when_the_backend_reports_a_different_direction(self) -> None:
        for rotate in (0, 180):
            with self.subTest(rotate=rotate):
                output_tex = self.directory / "page.tex"
                json_directory = self.directory / f"layout-{rotate}"
                backend = FakeLayoutBackend(
                    PageLayout(blocks=(Paragraph("Text of the page."),), page_top="bottom")
                )
                with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
                    convert_to_latex(
                        self.photo(),
                        output_tex,
                        backend=backend,
                        rotate=rotate,
                        ocr_output_directory=json_directory,
                    )
                self.assertEqual(backend.image_sizes, [(400, 200)])
                self.assertEqual(
                    captured_logs.output,
                    [
                        (
                            f"WARNING:{LOGGER_NAME}:Page 1: --rotate {rotate} gives the direction of "
                            "the page, but the backend reports the top of the page at the bottom of "
                            "the image; using the blocks of the backend"
                        )
                    ],
                )
                self.assertIn("Text of the page.", output_tex.read_text(encoding="utf-8"))
                saved = json.loads((json_directory / "page-000001.json").read_text())
                self.assertEqual(saved["page_top"], "top")

    def test_rotate_fails_for_a_page_that_the_backend_does_not_read(self) -> None:
        # A backend that looks for the direction returns no blocks for a page that is not
        # upright. With `rotate`, that reply must not remove the page from the document.
        output_tex = self.directory / "page.tex"
        for rotate in (0, 90):
            with self.subTest(rotate=rotate):
                backend = FakeLayoutBackend(PageLayout(page_top="left"))
                with self.assertRaisesRegex(
                    PageLayoutError,
                    "page 1 has no usable layout: the backend returned no blocks and reports "
                    "the top of the page at the left of the image, but "
                    f"--rotate {rotate} gives a different direction; check the value of --rotate",
                ):
                    convert_to_latex(self.photo(), output_tex, backend=backend, rotate=rotate)
                self.assertEqual(len(backend.image_sizes), 1)
                self.assertFalse(output_tex.exists())

    def test_document_without_content_is_a_failure(self) -> None:
        output_tex = self.directory / "page.tex"
        for rotate in (None, 0):
            with (
                self.subTest(rotate=rotate),
                self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
                self.assertRaisesRegex(
                    PageLayoutError,
                    "no page of .*photo.png has content in its layout, so there is no document",
                ),
            ):
                convert_to_latex(
                    self.photo(), output_tex, backend=FakeLayoutBackend(PageLayout()), rotate=rotate
                )
            self.assertEqual(
                captured_logs.output,
                [
                    (
                        f"WARNING:{LOGGER_NAME}:Page 1 has no content in its layout; "
                        "compare it with the source"
                    )
                ],
            )
            self.assertFalse(output_tex.exists())

    def test_warns_about_a_page_without_content(self) -> None:
        input_pdf = self.directory / "input.pdf"
        source = canvas.Canvas(str(input_pdf), pagesize=(144, 72))
        source.showPage()
        source.drawString(10, 50, "Second page")
        source.save()
        output_tex = self.directory / "book.tex"
        backend = FakeLayoutBackend(PageLayout(), PageLayout(blocks=(Paragraph("Text."),)))

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            convert_to_latex(input_pdf, output_tex, backend=backend, dpi=72)

        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:Page 1 has no content in its layout; "
                    "compare it with the source"
                )
            ],
        )
        # The page without content is an empty page of the document.
        self.assertIn(
            "% Source page 1\n\\begin{sourcepage}{1}\n\n\\end{sourcepage}\n\n"
            "% Source page 2\n\\begin{sourcepage}{2}\n\nText.\n\n\\end{sourcepage}",
            output_tex.read_text(),
        )

    def test_documents_in_one_directory_do_not_share_their_illustrations(self) -> None:
        backend = FakeLayoutBackend(PageLayout(blocks=(Figure(box=Box(0, 0, 500, 500)),)))
        # Names that differ only in characters that are not safe in LaTeX, and two names
        # that have no safe character.
        stems = [
            "A book (1920)",
            "A-book-1920",
            "A book [1920]",
            "\u041a\u043d\u0438\u0433\u0430",
            "\u0414\u0440\u0443\u0433\u0430\u044f",
        ]
        included: dict[str, str] = {}
        for stem in stems:
            output_tex = self.directory / "books" / f"{stem}.tex"
            convert_to_latex(self.photo(), output_tex, backend=backend, rotate=0)
            document = output_tex.read_text(encoding="utf-8")
            (path,) = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", document)
            self.assertTrue((output_tex.parent / path).is_file())
            included[stem] = path

        self.assertEqual(len(set(included.values())), len(stems))
        # A name that is safe has no hash. The other names have a part of their hash.
        self.assertEqual(included["A-book-1920"], "A-book-1920-figures/page-0001-figure-01.jpg")
        for stem in set(stems) - {"A-book-1920"}:
            name_hash = hashlib.sha256(f"{stem}.tex".encode()).hexdigest()[:8]
            self.assertRegex(
                included[stem],
                rf"^(A-book-1920|document)-{name_hash}-figures/page-0001-figure-01\.jpg$",
            )
        self.assertEqual(
            sorted(path.name for path in (self.directory / "books").iterdir() if path.is_dir()),
            sorted(path.split("/")[0] for path in included.values()),
        )

    def test_replaces_only_the_illustrations_of_an_earlier_conversion(self) -> None:
        output_tex = self.directory / "book.tex"
        figure_directory = self.directory / "book-figures"
        figure_directory.mkdir()
        (figure_directory / "page-0009-figure-01.jpg").write_bytes(b"old")
        # Files of the user. Scanwich gives none of these names to an illustration.
        user_files = [
            "notes.txt",
            "page-0001-figure-01-edited.jpg",
            "page-0001-figure-01.jpg.bak",
            "page-0001-figure-01.png",
            "my-page-0001-figure-01.jpg",
        ]
        for name in user_files:
            (figure_directory / name).write_text("a file of the user")
        (figure_directory / "page-0002-figure-01.jpg").mkdir()
        with_figure = FakeLayoutBackend(PageLayout(blocks=(Figure(box=Box(0, 0, 500, 500)),)))
        without_figure = FakeLayoutBackend(PageLayout(blocks=(Paragraph("Text."),)))

        convert_to_latex(self.photo(), output_tex, backend=with_figure)

        self.assertEqual(
            sorted(path.name for path in figure_directory.iterdir()),
            sorted([*user_files, "page-0001-figure-01.jpg", "page-0002-figure-01.jpg"]),
        )

        # A conversion without illustrations removes those of the earlier conversion.
        convert_to_latex(self.photo(), output_tex, backend=without_figure)
        self.assertEqual(
            sorted(path.name for path in figure_directory.iterdir()),
            sorted([*user_files, "page-0002-figure-01.jpg"]),
        )
        for name in user_files:
            self.assertEqual((figure_directory / name).read_text(), "a file of the user")
        # Scanwich removes the directory only when it is empty.
        shutil.rmtree(figure_directory)
        convert_to_latex(self.photo(), output_tex, backend=with_figure)
        convert_to_latex(self.photo(), output_tex, backend=without_figure)
        self.assertFalse(figure_directory.exists())

    def test_saves_layout_json_that_has_one_half_of_a_character(self) -> None:
        # A lone surrogate comes from a JSON escape without its pair. A file in UTF-8
        # cannot have it.
        half = chr(0xD83D)
        layout = PageLayout(
            blocks=(Paragraph(f"a {half} b"), PageFurniture("page_header", f"h {half}"))
        )
        output_tex = self.directory / "page.tex"
        json_directory = self.directory / "layout"

        convert_to_latex(
            self.photo(),
            output_tex,
            backend=FakeLayoutBackend(layout),
            ocr_output_directory=json_directory,
        )

        saved = (json_directory / "page-000001.json").read_bytes().decode("utf-8")
        self.assertIn('"a \\ud83d b"', saved)
        self.assertEqual(json.loads(saved), layout.to_json())
        self.assertIn("a b", output_tex.read_text(encoding="utf-8"))

    def test_failure_writes_no_document_and_keeps_the_earlier_one(self) -> None:
        output_tex = self.directory / "book.tex"
        output_tex.write_text("the earlier document")
        figure_directory = self.directory / "book-figures"
        figure_directory.mkdir()
        (figure_directory / "page-0001-figure-01.jpg").write_bytes(b"old")

        with self.assertRaisesRegex(RuntimeError, "the service is not available"):
            convert_to_latex(self.photo(), output_tex, backend=FailingLayoutBackend())

        self.assertEqual(output_tex.read_text(), "the earlier document")
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            ["book-figures", "book.tex", "photo.png"],
        )
        self.assertEqual(
            [path.name for path in figure_directory.iterdir()], ["page-0001-figure-01.jpg"]
        )

    @contextmanager
    def full_disk_at_illustration(self, number: int) -> Iterator[None]:
        """Let the copy of a new illustration to the output directory fail: the disk is full.

        `number` is the illustration that does not arrive. The illustrations before it
        arrive.
        """
        arrived = 0

        def guarded(real: object) -> object:
            def copy(source: object, destination: object, *args: object, **options: object):
                nonlocal arrived
                if self.directory in Path(str(destination)).parents:
                    arrived += 1
                    if arrived == number:
                        raise OSError(errno.ENOSPC, "No space left on device", str(destination))
                return real(source, destination, *args, **options)  # type: ignore[operator]

            return copy

        with (
            patch("shutil.move", guarded(shutil.move)),
            patch("shutil.copyfile", guarded(shutil.copyfile)),
        ):
            yield

    def earlier_result(self) -> tuple[Path, Path, bytes, dict[str, bytes]]:
        """Make a document with 2 illustrations and a file of the user next to them."""
        output_tex = self.directory / "book.tex"
        figure_directory = self.directory / "book-figures"
        layout = PageLayout(
            blocks=(
                Paragraph("The earlier text."),
                Figure(box=Box(0, 0, 500, 500)),
                Figure(box=Box(500, 0, 1000, 500)),
            )
        )
        convert_to_latex(self.photo(), output_tex, backend=FakeLayoutBackend(layout))
        (figure_directory / "notes.txt").write_text("a file of the user")
        files = {path.name: path.read_bytes() for path in figure_directory.iterdir()}
        self.assertEqual(
            sorted(files), ["notes.txt", "page-0001-figure-01.jpg", "page-0001-figure-02.jpg"]
        )
        return output_tex, figure_directory, output_tex.read_bytes(), files

    def new_backend(self) -> FakeLayoutBackend:
        """Return a backend for a new conversion: other text and 3 other illustrations."""
        return FakeLayoutBackend(
            PageLayout(
                blocks=(
                    Paragraph("The new text."),
                    Figure(box=Box(750, 0, 1000, 250)),
                    Figure(box=Box(0, 500, 500, 1000)),
                    Figure(box=Box(250, 250, 750, 750)),
                )
            )
        )

    def test_failure_while_the_illustrations_arrive_keeps_the_earlier_result(self) -> None:
        # The earlier document includes its illustrations by name. When a new illustration
        # does not arrive, the earlier document and each earlier illustration must stay.
        output_tex, figure_directory, earlier_tex, earlier_files = self.earlier_result()
        for number in (1, 2, 3):
            with (
                self.subTest(illustration=number),
                self.full_disk_at_illustration(number),
                self.assertRaisesRegex(OSError, "No space left on device"),
            ):
                convert_to_latex(self.photo(), output_tex, backend=self.new_backend())
            self.assertEqual(output_tex.read_bytes(), earlier_tex)
            self.assertEqual(
                {path.name: path.read_bytes() for path in figure_directory.iterdir()},
                earlier_files,
            )
            # No temporary file and no temporary directory stays.
            self.assertEqual(
                sorted(path.name for path in self.directory.iterdir()),
                ["book-figures", "book.tex", "photo.png"],
            )

    def test_failure_of_the_document_step_keeps_the_earlier_result(self) -> None:
        # The new illustrations are in place when the last step replaces the document.
        output_tex, figure_directory, earlier_tex, earlier_files = self.earlier_result()
        real_replace = Path.replace

        def replace_document(source: Path, target: Path) -> Path:
            if source.name.endswith(".scanwich-tmp"):
                raise OSError(errno.EIO, "Input/output error", str(target))
            return real_replace(source, target)

        with (
            patch.object(Path, "replace", replace_document),
            self.assertRaisesRegex(OSError, "Input/output error"),
        ):
            convert_to_latex(self.photo(), output_tex, backend=self.new_backend())

        self.assertEqual(output_tex.read_bytes(), earlier_tex)
        self.assertEqual(
            {path.name: path.read_bytes() for path in figure_directory.iterdir()}, earlier_files
        )
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            ["book-figures", "book.tex", "photo.png"],
        )

    def test_error_record_names_an_earlier_illustration_that_does_not_come_back(self) -> None:
        # The document step fails, and then one earlier illustration does not get its
        # name again. It must stay on the disk, and a log record must name its file.
        output_tex, figure_directory, earlier_tex, earlier_files = self.earlier_result()
        real_replace = Path.replace
        lost = ".page-0001-figure-02.jpg.scanwich-old"

        def replace_with_errors(source: Path, target: Path) -> Path:
            if source.name.endswith(".scanwich-tmp") or source.name == lost:
                raise OSError(errno.EIO, "Input/output error", str(target))
            return real_replace(source, target)

        with (
            patch.object(Path, "replace", replace_with_errors),
            self.assertLogs(LOGGER_NAME, level="ERROR") as captured_logs,
            self.assertRaisesRegex(OSError, "Input/output error"),
        ):
            convert_to_latex(self.photo(), output_tex, backend=self.new_backend())

        self.assertEqual(output_tex.read_bytes(), earlier_tex)
        files = {path.name: path.read_bytes() for path in figure_directory.iterdir()}
        # The other earlier illustration has its name, and no new illustration stays.
        self.assertEqual(sorted(files), [lost, "notes.txt", "page-0001-figure-01.jpg"])
        self.assertEqual(files[lost], earlier_files["page-0001-figure-02.jpg"])
        self.assertEqual(files["page-0001-figure-01.jpg"], earlier_files["page-0001-figure-01.jpg"])
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"ERROR:{LOGGER_NAME}:Could not put back the illustration "
                    f"{figure_directory / 'page-0001-figure-02.jpg'} of the earlier document; it "
                    f"is the file {figure_directory / lost}"
                )
            ],
        )

    def test_failure_of_a_first_conversion_leaves_no_part_of_a_result(self) -> None:
        output_tex = self.directory / "book.tex"
        with (
            self.full_disk_at_illustration(2),
            self.assertRaisesRegex(OSError, "No space left on device"),
        ):
            convert_to_latex(self.photo(), output_tex, backend=self.new_backend())
        self.assertEqual([path.name for path in self.directory.iterdir()], ["photo.png"])

    def test_new_result_has_only_the_new_illustrations(self) -> None:
        output_tex, figure_directory, earlier_tex, earlier_files = self.earlier_result()
        # A conversion that the system stopped can leave its temporary files.
        (figure_directory / ".page-0001-figure-01.jpg.scanwich-new").write_bytes(b"stale")
        (figure_directory / ".page-0007-figure-01.jpg.scanwich-old").write_bytes(b"stale")

        convert_to_latex(self.photo(), output_tex, backend=self.new_backend())

        files = {path.name: path.read_bytes() for path in figure_directory.iterdir()}
        self.assertEqual(
            sorted(files),
            [
                "notes.txt",
                "page-0001-figure-01.jpg",
                "page-0001-figure-02.jpg",
                "page-0001-figure-03.jpg",
            ],
        )
        self.assertEqual(files["notes.txt"], b"a file of the user")
        # Each illustration is the new one: the new crops have other sizes.
        for name in ("page-0001-figure-01.jpg", "page-0001-figure-02.jpg"):
            self.assertNotEqual(files[name], earlier_files[name])
        with Image.open(figure_directory / "page-0001-figure-03.jpg") as figure:
            self.assertEqual(figure.size, (204, 102))
        self.assertIn("The new text.", output_tex.read_text(encoding="utf-8"))
        self.assertNotEqual(output_tex.read_bytes(), earlier_tex)
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            ["book-figures", "book.tex", "photo.png"],
        )

    def test_rejects_arguments_that_cannot_give_a_document(self) -> None:
        photo = self.photo()
        backend = FakeLayoutBackend(PageLayout())
        output_tex = self.directory / "page.tex"
        with self.assertRaisesRegex(FileNotFoundError, "input file does not exist"):
            convert_to_latex(self.directory / "missing.png", output_tex, backend=backend)
        with self.assertRaisesRegex(ValueError, "must be different"):
            convert_to_latex(photo, photo, backend=backend)
        with self.assertRaisesRegex(ValueError, "DPI must be greater than zero"):
            convert_to_latex(photo, output_tex, backend=backend, dpi=0)
        with self.assertRaisesRegex(ValueError, "0, 90, 180, or 270 degrees"):
            convert_to_latex(photo, output_tex, backend=backend, rotate=45)
        with self.assertRaisesRegex(TypeError, "FakeBackend does not implement recognize_layout"):
            convert_to_latex(photo, output_tex, backend=FakeBackend())
        self.assertEqual(backend.image_sizes, [])
        self.assertFalse(output_tex.exists())

    def test_reports_a_file_that_is_not_a_pdf_and_not_an_image(self) -> None:
        text_file = self.directory / "notes.txt"
        text_file.write_text("not a page")
        with self.assertRaisesRegex(RuntimeError, "notes.txt is not a PDF and not an image"):
            convert_to_latex(
                text_file, self.directory / "page.tex", backend=FakeLayoutBackend(PageLayout())
            )


def red_part(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Return the pixel box of the red part of an image, or None."""
    red, green, blue = image.convert("RGB").split()
    mask = ImageChops.multiply(
        red.point(lambda value: 255 if value > 200 else 0),
        ImageChops.multiply(
            green.point(lambda value: 255 if value < 60 else 0),
            blue.point(lambda value: 255 if value < 60 else 0),
        ),
    )
    return mask.getbbox()


class FakeFigureBackend(FakeLayoutBackend):
    """A layout backend that also finds an illustration in a part of a page.

    The illustration is the red part of the image. It continues at each side of the
    image that the red part touches.
    """

    def __init__(self, *layouts: PageLayout) -> None:
        super().__init__(*layouts)
        self.part_sizes: list[tuple[int, int]] = []
        self.estimates: list[Box | None] = []
        self.captions: list[str | None] = []

    def locate_figure(
        self, image_path: Path, *, estimate: Box | None = None, caption: str | None = None
    ) -> FigureLocation | None:
        with Image.open(image_path) as image:
            width, height = image.size
            pixels = red_part(image)
        self.part_sizes.append((width, height))
        self.estimates.append(estimate)
        self.captions.append(caption)
        if pixels is None:
            return None
        left, top, right, bottom = pixels
        sides = {
            "left": left == 0,
            "top": top == 0,
            "right": right == width,
            "bottom": bottom == height,
        }
        return FigureLocation(
            Box(
                left * 1000 / width,
                top * 1000 / height,
                right * 1000 / width,
                bottom * 1000 / height,
            ),
            frozenset(side for side, is_cut in sides.items() if is_cut),
        )


class FigureBoxTests(TestCase):
    """The second request for the box of an illustration.

    The page image has 1000x2000 pixels, so one unit of the grid is 1 pixel across and
    2 pixels down.
    """

    def setUp(self) -> None:
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)

    def page(self, *red_boxes: tuple[int, int, int, int]) -> Path:
        """Return a white page image with red parts. Each box is on the grid of the page."""
        path = self.directory / "page.png"
        image = Image.new("RGB", (1000, 2000), WHITE)
        for left, top, right, bottom in red_boxes:
            image.paste(RED, (left, top * 2, right, bottom * 2))
        image.save(path)
        return path

    def convert(self, backend: FakeLayoutBackend, page: Path) -> Box:
        """Convert the page, and return the box of its illustration in the layout JSON."""
        json_directory = self.directory / "layout"
        convert_to_latex(
            page, self.directory / "page.tex", backend=backend, ocr_output_directory=json_directory
        )
        saved = json.loads((json_directory / "page-000001.json").read_text(encoding="utf-8"))
        (figure,) = [block for block in saved["blocks"] if block["type"] == "figure"]
        return Box(**figure["box"])

    def crop(self) -> Image.Image:
        with Image.open(self.directory / "page-figures" / "page-0001-figure-01.jpg") as image:
            return image.convert("RGB")

    def test_backend_contract_for_the_box_of_an_illustration_is_an_addition(self) -> None:
        self.assertIsInstance(FakeFigureBackend(), LayoutBackend)
        self.assertIsInstance(FakeFigureBackend(), FigureLocator)
        self.assertNotIsInstance(FakeLayoutBackend(), FigureLocator)

    def test_crops_the_box_of_the_second_request(self) -> None:
        # The box of the layout is too low: it cuts the top third of the illustration,
        # and it has the area below the illustration.
        layout = PageLayout(
            blocks=(Paragraph("Text."), Figure(box=Box(300, 450, 600, 600), caption="Fig. 1"))
        )
        backend = FakeFigureBackend(layout)

        with self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs:
            box = self.convert(backend, self.page((300, 400, 600, 550)))

        self.assertEqual(box, Box(300.0, 400.0, 600.0, 550.0))
        # The backend got one part of the page: the box of the layout with a margin of
        # 150 units across (half of the box) and 100 units down.
        self.assertEqual(backend.part_sizes, [(600, 700)])
        self.assertEqual(backend.captions, ["Fig. 1"])
        (estimate,) = backend.estimates
        assert estimate is not None
        # The estimate is the box of the layout on the grid of the part.
        self.assertEqual(
            [
                round(value, 1)
                for value in (estimate.left, estimate.top, estimate.right, estimate.bottom)
            ],
            [250.0, 285.7, 750.0, 714.3],
        )
        # The crop has the full illustration and a margin of 5 units of white paper.
        crop = self.crop()
        self.assertEqual(crop.size, (310, 320))
        self.assertEqual(red_part(crop), (5, 10, 305, 310))
        self.assertIn(
            f"INFO:{LOGGER_NAME}:The box of illustration 1 of page 1 is left 300, top 400, "
            "right 600, bottom 550 after 1 request(s) for it; the page layout gave left 300, "
            "top 450, right 600, bottom 600",
            captured_logs.output,
        )

    def test_asks_again_with_a_larger_part_when_the_part_cuts_the_illustration(self) -> None:
        layout = PageLayout(blocks=(Figure(box=Box(300, 520, 600, 640)),))
        backend = FakeFigureBackend(layout)

        box = self.convert(backend, self.page((300, 400, 600, 550)))

        self.assertEqual(box, Box(300.0, 400.0, 600.0, 550.0))
        # The first part starts at 420 on the grid and the illustration at 400. The
        # second part is 100 units larger at its top.
        self.assertEqual(backend.part_sizes, [(600, 640), (600, 840)])
        # The second request has the box of the first one as its estimate.
        second_estimate = backend.estimates[1]
        assert second_estimate is not None
        self.assertEqual(round(second_estimate.top, 1), 238.1)
        self.assertEqual(red_part(self.crop()), (5, 10, 305, 310))

    def test_asks_again_when_the_box_is_near_the_border_of_the_part(self) -> None:
        # A label of the illustration is above the first part, so the backend cannot
        # see it there. The box of the first request is 15 units from the top border.
        layout = PageLayout(blocks=(Figure(box=Box(300, 485, 600, 640)),))
        backend = FakeFigureBackend(layout)

        box = self.convert(backend, self.page((300, 400, 600, 550), (400, 370, 500, 380)))

        self.assertEqual(backend.part_sizes, [(600, 710), (600, 910)])
        self.assertEqual(box, Box(300.0, 370.0, 600.0, 550.0))

    def test_side_of_the_part_at_the_edge_of_the_page_cannot_move(self) -> None:
        layout = PageLayout(blocks=(Figure(box=Box(0, 410, 200, 560)),))
        backend = FakeFigureBackend(layout)

        box = self.convert(backend, self.page((0, 400, 200, 550)))

        # The illustration touches the left edge of the page, and thus of the part.
        self.assertEqual(backend.part_sizes, [(300, 700)])
        self.assertEqual(box, Box(0.0, 400.0, 200.0, 550.0))

    def test_crop_is_larger_at_a_side_that_stays_cut(self) -> None:
        class CutBackend(FakeFigureBackend):
            def locate_figure(self, image_path: Path, **hints: object) -> FigureLocation:
                super().locate_figure(image_path)
                return FigureLocation(Box(100, 100, 900, 900), frozenset({"bottom"}))

        layout = PageLayout(blocks=(Figure(box=Box(300, 400, 600, 550)),))
        backend = CutBackend(layout)

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            box = self.convert(backend, self.page())

        self.assertEqual(FIGURE_SEARCH_REQUESTS, 3)
        # Each request has a part that is 100 units larger at its bottom.
        self.assertEqual(backend.part_sizes, [(600, 700), (600, 900), (600, 1100)])
        # The last part is from 300 to 850. The box gets the bottom of that part.
        self.assertEqual(box, Box(210.0, 355.0, 690.0, 850.0))
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:After 3 requests, the backend still reports that "
                    "illustration 1 of page 1 continues at the bottom; the crop is larger at such "
                    "a side. Compare the illustration with the source"
                )
            ],
        )

    def test_keeps_the_box_of_the_layout_when_the_second_request_gives_no_box(self) -> None:
        class FailingFigureBackend(FakeFigureBackend):
            def locate_figure(self, image_path: Path, **hints: object) -> FigureLocation:
                raise RuntimeError("the service is not available")

        first_box = Box(300, 450, 600, 600)
        layout = PageLayout(blocks=(Figure(box=first_box),))
        cases = [
            (
                FailingFigureBackend(layout),
                self.page((300, 400, 600, 550)),
                (
                    "The request for the box of illustration 1 of page 1 failed (RuntimeError: "
                    "the service is not available); the crop uses the box that the backend gave "
                    "before. Compare the illustration with the source"
                ),
            ),
            (
                # The part of the page has no red part, so the backend finds nothing.
                FakeFigureBackend(layout),
                self.page(),
                (
                    "The backend found no illustration in the part of the page image for "
                    "illustration 1 of page 1; the crop uses the box that the backend gave "
                    "before. Compare the illustration with the source"
                ),
            ),
        ]
        for backend, page, warning in cases:
            with (
                self.subTest(backend=type(backend).__name__),
                self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
            ):
                box = self.convert(backend, page)
                self.assertEqual(box, first_box)
                self.assertEqual(captured_logs.output, [f"WARNING:{LOGGER_NAME}:{warning}"])
                self.assertEqual(self.crop().size, (310, 320))

    def test_each_error_of_the_second_request_keeps_the_document(self) -> None:
        # The page has its layout. A failure of the request for a box gives the box of
        # the layout, for each type of error of a backend.
        first_box = Box(300, 450, 600, 600)
        layout = PageLayout(blocks=(Paragraph("Text."), Figure(box=first_box)))
        errors = [
            TypeError("a wrong type"),
            ConnectionError("no network"),
            TimeoutError("too slow"),
            KeyError("box"),
            OSError("no space"),
        ]
        for error in errors:

            class RaisingBackend(FakeFigureBackend):
                def locate_figure(
                    self, image_path: Path, *, failure: Exception = error, **hints: object
                ) -> FigureLocation:
                    raise failure

            with (
                self.subTest(error=type(error).__name__),
                self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
            ):
                box = self.convert(RaisingBackend(layout), self.page((300, 400, 600, 550)))
                self.assertEqual(box, first_box)
                self.assertEqual(len(captured_logs.output), 1)
                self.assertIn(
                    f"The request for the box of illustration 1 of page 1 failed "
                    f"({type(error).__name__}: ",
                    captured_logs.output[0],
                )
                self.assertIn("Text.", (self.directory / "page.tex").read_text(encoding="utf-8"))

    def fixed_box_backend(self, layout: PageLayout, reply: FigureLocation) -> FakeFigureBackend:
        class FixedBackend(FakeFigureBackend):
            def locate_figure(self, image_path: Path, **hints: object) -> FigureLocation:
                super().locate_figure(image_path)
                return reply

        return FixedBackend(layout)

    def test_second_box_that_is_in_doubt_does_not_replace_the_first_box(self) -> None:
        # The first box is 300 to 600 across and 400 to 550 down. The part of the page for
        # the second request is 150 to 750 across and 300 to 650 down.
        first_box = Box(300, 400, 600, 550)
        layout = PageLayout(blocks=(Figure(box=first_box),))
        cases = {
            "no part of the first box": (
                # 210 to 270 across, 335 to 363 down: a different place.
                FigureLocation(Box(100, 100, 200, 180)),
                Box(210.0, 335.0, 600.0, 550.0),
                (95, 140, 395, 440),
                (
                    "The second box of illustration 1 of page 1 (left 210, top 335, right 270, "
                    "bottom 363) has no part of the box of the page layout (left 300, top 400, "
                    "right 600, bottom 550); the crop has the two boxes. Compare the "
                    "illustration with the source"
                ),
            ),
            "a small part of the first box": (
                # 420 to 480 across, 440 to 510 down: 9 percent of the area.
                FigureLocation(Box(450, 400, 550, 600)),
                first_box,
                (5, 10, 305, 310),
                (
                    "The second box of illustration 1 of page 1 (left 420, top 440, right 480, "
                    "bottom 510) is much smaller than the box of the page layout (left 300, top "
                    "400, right 600, bottom 550); the crop has the two boxes. Compare the "
                    "illustration with the source"
                ),
            ),
        }
        for name, (reply, final_box, red_pixels, warning) in cases.items():
            backend = self.fixed_box_backend(layout, reply)
            with (
                self.subTest(name=name),
                self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs,
            ):
                box = self.convert(backend, self.page((300, 400, 600, 550)))
                # The crop has the two boxes, so it has the full red part of the first box.
                self.assertEqual(box, final_box)
                self.assertEqual(captured_logs.output, [f"WARNING:{LOGGER_NAME}:{warning}"])
                self.assertEqual(red_part(self.crop()), red_pixels)

    def test_correct_second_box_with_no_part_of_the_first_box_is_in_the_crop(self) -> None:
        # A box of a page layout can be on the caption below a small illustration. The
        # second request then finds the illustration, and the two boxes have no common
        # part. The first box alone would lose the full illustration.
        layout = PageLayout(blocks=(Figure(box=Box(400, 370, 600, 420), caption="Fig. 1"),))
        backend = FakeFigureBackend(layout)

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            box = self.convert(backend, self.page((400, 300, 600, 350)))

        self.assertEqual(backend.part_sizes, [(400, 500)])
        self.assertEqual(box, Box(400.0, 300.0, 600.0, 420.0))
        # The crop has the full illustration: 200 units across and 50 units down.
        self.assertEqual(red_part(self.crop()), (5, 10, 205, 110))
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:The second box of illustration 1 of page 1 (left 400, "
                    "top 300, right 600, bottom 350) has no part of the box of the page layout "
                    "(left 400, top 370, right 600, bottom 420); the crop has the two boxes. "
                    "Compare the illustration with the source"
                )
            ],
        )

    def test_second_box_that_is_much_larger_gets_a_warning(self) -> None:
        # The backend returns the full part of the page, each time.
        first_box = Box(300, 400, 600, 550)
        layout = PageLayout(blocks=(Figure(box=first_box),))
        backend = self.fixed_box_backend(layout, FigureLocation(Box(0, 0, 1000, 1000)))

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            box = self.convert(backend, self.page((300, 400, 600, 550)))

        # Each request has a larger part, because the box is at the border of the part.
        self.assertEqual(len(backend.part_sizes), FIGURE_SEARCH_REQUESTS)
        self.assertEqual(box, Box(0.0, 100.0, 950.0, 850.0))
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:The second box of illustration 1 of page 1 (left 0, "
                    "top 100, right 950, bottom 850) is much larger than the box of the page "
                    "layout (left 300, top 400, right 600, bottom 550); the crop has the two "
                    "boxes. Compare the illustration with the source"
                )
            ],
        )

    def test_second_box_of_a_usual_size_replaces_the_first_box(self) -> None:
        # A first box can be 2 times too large, or have a third of the illustration.
        first_box = Box(300, 400, 600, 550)
        layout = PageLayout(blocks=(Figure(box=first_box),))
        for reply, final_box in (
            # 3.3 times the area of the first box.
            (FigureLocation(Box(60, 100, 940, 900)), Box(186.0, 335.0, 714.0, 615.0)),
            # 0.28 times the area.
            (FigureLocation(Box(350, 300, 650, 500)), Box(360.0, 405.0, 540.0, 475.0)),
        ):
            backend = self.fixed_box_backend(layout, reply)
            with self.subTest(reply=reply), self.assertNoLogs(LOGGER_NAME, level="WARNING"):
                self.assertEqual(self.convert(backend, self.page()), final_box)

    def test_one_page_has_a_limit_of_requests_for_boxes(self) -> None:
        # A model can report each ornament of a page as an illustration.
        figures = tuple(
            Figure(box=Box(100, 40 + 60 * number, 300, 80 + 60 * number)) for number in range(15)
        )
        layout = PageLayout(blocks=(Paragraph("Text."), *figures))
        backend = self.fixed_box_backend(layout, FigureLocation(Box(300, 300, 700, 700)))

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            convert_to_latex(
                self.page(),
                self.directory / "page.tex",
                backend=backend,
                ocr_output_directory=self.directory / "layout",
            )

        self.assertEqual(FIGURE_SEARCH_PAGE_REQUESTS, 12)
        self.assertEqual(len(backend.part_sizes), 12)
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:Page 1 has 15 illustration(s) in its layout, and one "
                    "page has a limit of 12 requests for the boxes of illustrations; 3 "
                    "illustration(s) use the box of the page layout. Compare them with the source"
                )
            ],
        )
        saved = json.loads((self.directory / "layout" / "page-000001.json").read_text())
        boxes = [Box(**block["box"]) for block in saved["blocks"] if block["type"] == "figure"]
        # The last 3 illustrations have the boxes of the page layout.
        self.assertEqual(boxes[12:], [figure.box for figure in figures[12:]])
        self.assertNotEqual(boxes[0], figures[0].box)

    def test_requests_of_one_illustration_count_for_the_limit_of_the_page(self) -> None:
        # Each illustration needs 3 requests here, so 4 illustrations use the 12 requests.
        figures = tuple(
            Figure(box=Box(300, 100 + 150 * number, 600, 200 + 150 * number)) for number in range(5)
        )
        layout = PageLayout(blocks=figures)
        backend = self.fixed_box_backend(
            layout, FigureLocation(Box(300, 300, 700, 700), frozenset({"left"}))
        )

        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            convert_to_latex(self.page(), self.directory / "page.tex", backend=backend)

        self.assertEqual(len(backend.part_sizes), 12)
        self.assertIn("1 illustration(s) use the box of the page layout", captured_logs.output[-1])
