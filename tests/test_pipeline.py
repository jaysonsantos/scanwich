from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from pypdf import PdfReader
from reportlab.pdfgen import canvas

from scanwich.models import OcrRegion, Point
from scanwich.pipeline import convert_pdf, parse_backend_options


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
            self.assertIn(f"Wrote searchable PDF to {output_pdf}", progress)
