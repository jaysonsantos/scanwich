from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PIL import Image
from pypdf import PdfReader

from scanwich.models import OcrRegion, Point
from scanwich.pdf import assemble_searchable_pdf


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

            assemble_searchable_pdf([image_path], [[region]], output_path, dpi=300)

            reader = PdfReader(output_path)
            self.assertEqual(len(reader.pages), 1)
            self.assertIn("Wasser água", reader.pages[0].extract_text())
            self.assertAlmostEqual(float(reader.pages[0].mediabox.width), 144.0)
            self.assertAlmostEqual(float(reader.pages[0].mediabox.height), 72.0)
