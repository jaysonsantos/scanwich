from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from scanwich.models import OcrRegion, Point
from scanwich.pdf import DEFAULT_DPI, RasterizedPage, assemble_searchable_pdf, rasterize_pdf


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
