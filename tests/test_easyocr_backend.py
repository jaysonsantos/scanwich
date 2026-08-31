import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from scanwich.backends.easyocr import EasyOcrBackend


class FakeReader:
    def readtext(self, image_path: str, **options: object) -> list[object]:
        return []


class EasyOcrBackendTests(TestCase):
    def test_initializes_reader_lazily_with_progress_logs(self) -> None:
        reader_calls: list[tuple[list[str], dict[str, object]]] = []

        def make_reader(languages: list[str], **options: object) -> FakeReader:
            reader_calls.append((languages, options))
            return FakeReader()

        fake_easyocr = SimpleNamespace(Reader=make_reader)
        backend = EasyOcrBackend(languages=["de", "pt", "en"], options={})
        self.assertEqual(reader_calls, [])

        with (
            patch.dict(sys.modules, {"easyocr": fake_easyocr}),
            self.assertLogs("scanwich.backends.easyocr", level="INFO") as captured_logs,
        ):
            backend.recognize(Path("page.png"))

        self.assertEqual(reader_calls, [(["de", "pt", "en"], {"gpu": False})])
        progress = "\n".join(captured_logs.output)
        self.assertIn("Importing EasyOCR", progress)
        self.assertIn("first use may download models and take a while", progress)
        self.assertIn("EasyOCR ready after", progress)
