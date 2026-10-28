from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from scanwich.models import OcrRegion, Point
from scanwich.pdf import (
    DEFAULT_DPI,
    PdfPipelineError,
    RasterizedPage,
    assemble_searchable_pdf,
    rasterize_pdf,
    rasterize_pdf_page,
)


class RasterizePdfTests(TestCase):
    def test_infers_dpi_from_a_full_page_image(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "source.png"
            input_path = directory / "input.pdf"
            Image.new("RGB", (200, 100), "white").save(image_path)
            source = canvas.Canvas(str(input_path), pagesize=(144, 72))
            source.drawImage(str(image_path), 0, 0, width=144, height=72)
            source.save()

            pages = rasterize_pdf(input_path, directory / "pages")

            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0].dpi, 100.0)
            with Image.open(pages[0].image_path) as rendered:
                self.assertEqual(rendered.size, (200, 100))

    def test_rasterizes_one_page_as_the_full_document_does(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "input.pdf"
            source = canvas.Canvas(str(input_path))
            for width in (144, 72, 216):
                source.setPageSize((width, 72))
                source.drawString(10, 50, f"a page {width} points wide")
                source.showPage()
            source.save()
            pages = rasterize_pdf(input_path, directory / "pages", dpi=100)

            one_page = rasterize_pdf_page(input_path, 2, directory / "one.png", dpi=100)

            self.assertEqual(one_page.image_path, directory / "one.png")
            self.assertEqual(
                (one_page.width_points, one_page.height_points, one_page.dpi),
                (pages[2].width_points, pages[2].height_points, pages[2].dpi),
            )
            self.assertEqual((directory / "one.png").read_bytes(), pages[2].image_path.read_bytes())
            # The function makes the image of one page only.
            self.assertEqual(sorted(path.name for path in directory.glob("*.png")), ["one.png"])
            with self.assertRaisesRegex(PdfPipelineError, "has 3 page.s., and no page 4"):
                rasterize_pdf_page(input_path, 3, directory / "none.png")
            self.assertFalse((directory / "none.png").exists())

    def test_uses_default_dpi_without_a_full_page_image(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "input.pdf"
            source = canvas.Canvas(str(input_path), pagesize=(72, 72))
            source.drawString(10, 50, "vector text")
            source.save()

            pages = rasterize_pdf(input_path, directory / "pages")

            self.assertEqual(pages[0].dpi, DEFAULT_DPI)
            with Image.open(pages[0].image_path) as rendered:
                self.assertEqual(rendered.size, (300, 300))

    def test_caps_inferred_dpi_at_default_dpi(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "source.png"
            input_path = directory / "input.pdf"
            Image.new("RGB", (800, 400), "white").save(image_path)
            source = canvas.Canvas(str(input_path), pagesize=(144, 72))
            source.drawImage(str(image_path), 0, 0, width=144, height=72)
            source.save()

            pages = rasterize_pdf(input_path, directory / "pages")

            self.assertEqual(pages[0].dpi, DEFAULT_DPI)
            with Image.open(pages[0].image_path) as rendered:
                self.assertEqual(rendered.size, (600, 300))

    def test_explicit_dpi_overrides_inference(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "source.png"
            input_path = directory / "input.pdf"
            Image.new("RGB", (200, 100), "white").save(image_path)
            source = canvas.Canvas(str(input_path), pagesize=(144, 72))
            source.drawImage(str(image_path), 0, 0, width=144, height=72)
            source.save()

            pages = rasterize_pdf(input_path, directory / "pages", dpi=72)

            self.assertEqual(pages[0].dpi, 72.0)
            with Image.open(pages[0].image_path) as rendered:
                self.assertEqual(rendered.size, (144, 72))

    def test_infers_dpi_from_a_rotated_full_page_image(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "source.png"
            unrotated_path = directory / "unrotated.pdf"
            input_path = directory / "input.pdf"
            Image.new("RGB", (200, 100), "white").save(image_path)
            source = canvas.Canvas(str(unrotated_path), pagesize=(144, 72))
            source.drawImage(str(image_path), 0, 0, width=144, height=72)
            source.save()
            reader = PdfReader(unrotated_path)
            writer = PdfWriter()
            writer.add_page(reader.pages[0].rotate(90))
            with input_path.open("wb") as stream:
                writer.write(stream)

            pages = rasterize_pdf(input_path, directory / "pages")

            self.assertEqual(pages[0].dpi, 100.0)
            with Image.open(pages[0].image_path) as rendered:
                self.assertEqual(rendered.size, (100, 200))


class AssembleSearchablePdfTests(TestCase):
    def test_preserves_image_and_adds_extractable_invisible_text(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "page.png"
            output_path = directory / "result.pdf"
            Image.new("RGB", (600, 300), "white").save(image_path)
            region = OcrRegion(
                text="Wasser água",
                polygon=(
                    Point(50, 50),
                    Point(300, 50),
                    Point(300, 100),
                    Point(50, 100),
                ),
                confidence=0.98,
            )

            page = RasterizedPage(
                image_path=image_path,
                width_points=144,
                height_points=72,
                dpi=300,
            )
            assemble_searchable_pdf([page], [[region]], output_path)

            reader = PdfReader(output_path)
            self.assertEqual(len(reader.pages), 1)
            self.assertIn("Wasser água", reader.pages[0].extract_text())
            self.assertAlmostEqual(float(reader.pages[0].mediabox.width), 144.0)
            self.assertAlmostEqual(float(reader.pages[0].mediabox.height), 72.0)

    def test_warns_about_regions_of_zero_size(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "page.png"
            output_path = directory / "result.pdf"
            Image.new("RGB", (600, 300), "white").save(image_path)
            page = RasterizedPage(
                image_path=image_path, width_points=144, height_points=72, dpi=300
            )
            visible = OcrRegion(
                text="visible",
                polygon=(Point(50, 50), Point(300, 50), Point(300, 100), Point(50, 100)),
            )
            # A backend that clamps a region below the page puts its four points on the edge.
            on_the_edge = OcrRegion(
                text="hidden",
                polygon=(Point(50, 300), Point(300, 300), Point(300, 300), Point(50, 300)),
            )

            with self.assertLogs("scanwich.pdf", level="WARNING") as logs:
                assemble_searchable_pdf(
                    [page, page], [[visible], [visible, on_the_edge]], output_path
                )

            self.assertEqual(
                logs.output,
                [
                    (
                        "WARNING:scanwich.pdf:Page 2: 1 of 2 text region(s) have zero size, "
                        "so their text is not in the PDF"
                    )
                ],
            )
            text = PdfReader(output_path).pages[1].extract_text()
            self.assertIn("visible", text)
            self.assertNotIn("hidden", text)

    def test_extracts_text_drawn_along_a_vertical_baseline(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "page.png"
            output_path = directory / "result.pdf"
            Image.new("RGB", (200, 200), "white").save(image_path)
            # Top-left, top-right, bottom-right, bottom-left of glyphs that read upward.
            region = OcrRegion(
                text="SIDE",
                polygon=(Point(20, 160), Point(20, 40), Point(50, 40), Point(50, 160)),
            )
            page = RasterizedPage(
                image_path=image_path, width_points=200, height_points=200, dpi=72
            )

            assemble_searchable_pdf([page], [[region]], output_path)

            self.assertIn("SIDE", PdfReader(output_path).pages[0].extract_text())
            document = pdfium.PdfDocument(output_path)
            try:
                text_page = document[0].get_textpage()
                centers = []
                for index in range(text_page.count_chars()):
                    if not text_page.get_text_range(index, 1).strip():
                        continue
                    left, bottom, right, top = text_page.get_charbox(index)
                    centers.append(((left + right) / 2, (bottom + top) / 2))
            finally:
                document.close()
            xs = [x for x, _ in centers]
            ys = [y for _, y in centers]
            # The letters of a vertical word stack, so they spread in y and not in x.
            self.assertGreater(max(ys) - min(ys), max(xs) - min(xs))
            self.assertGreater(max(ys) - min(ys), 50)
