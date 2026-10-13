import hashlib
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from scanwich.layout import Box, Figure, Heading, PageFurniture, PageLayout, Paragraph
from scanwich.models import OcrRegion, Point
from scanwich.ocr import LayoutBackend, OcrBackend, backend_request_options
from scanwich.pipeline import (
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
        self.assertIn(
            "% Source page 1 (printed page number: 7)\n\n"
            "\\section*{A chapter}\n\n"
            "The first page ends in the middle of a sen%\n"
            "% Source page 2\n"
            "% Scanwich joined the word that the page break divides. "
            "Compare it with the source.\n"
            "tence, and stops here.\n\n"
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
        self.assertNotIn("illustration", progress)

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
            "\\par\\smallskip\nFig. 1\n\\end{minipage}\n\\end{center}",
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
        self.assertIn("% Source page 1\n\n% Source page 2\n\nText.", output_tex.read_text())

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
