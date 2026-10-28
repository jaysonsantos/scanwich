import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from scanwich.backends.easyocr import EasyOcrBackend
from scanwich.models import OcrRegion, Point

RED = (255, 0, 0)
PAGE_SIZE = (80, 30)


def raw_region(
    text: str,
    corners: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ],
    confidence: float,
) -> list[object]:
    return [list(corners), text, confidence]


class RecordingReader:
    """Return scripted detections. The red corner tells a 90-degree turn from a 270-degree turn."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.upright: list[object] = [
            raw_region("ALPHA", ((10, 4), (50, 4), (50, 14), (10, 14)), 0.91)
        ]
        # A horizontal word on the turned page. The backend maps it back to the original page.
        self.turn_90: list[object] = [
            raw_region("BETA", ((4, 10), (16, 10), (16, 14), (4, 14)), 0.88)
        ]
        self.turn_270: list[object] = [
            raw_region("GAMMA", ((4, 10), (16, 10), (16, 14), (4, 14)), 0.8)
        ]
        self.turn_180: list[object] = []

    def readtext(self, image: str, **options: object) -> list[object]:
        with Image.open(image) as opened:
            size = opened.size
            marked = opened.getpixel((opened.width - 1, 0)) == RED
            origin = opened.getpixel((0, 0)) == RED
        self.calls.append({"size": size, "options": options})
        if size == PAGE_SIZE and origin:
            return self.upright
        if size == PAGE_SIZE:
            return self.turn_180
        if marked:
            return self.turn_90
        return self.turn_270


def page_image(path: Path) -> None:
    image = Image.new("RGB", PAGE_SIZE, "white")
    image.putpixel((0, 0), RED)
    image.save(path)


def recognize(reader: RecordingReader, path: Path, **options: object) -> list[OcrRegion]:
    backend = EasyOcrBackend(languages=["en"], options=options)

    def make_reader(languages: list[str], **reader_options: object) -> RecordingReader:
        reader.reader_options = reader_options
        reader.languages = languages
        return reader

    fake_easyocr = SimpleNamespace(Reader=make_reader)
    with patch.dict(sys.modules, {"easyocr": fake_easyocr}):
        return backend.recognize(path)


class EasyOcrBackendTests(TestCase):
    def test_initializes_reader_lazily_with_progress_logs(self) -> None:
        reader_calls: list[tuple[list[str], dict[str, object]]] = []

        def make_reader(languages: list[str], **options: object) -> RecordingReader:
            reader_calls.append((languages, options))
            return RecordingReader()

        fake_easyocr = SimpleNamespace(Reader=make_reader)
        backend = EasyOcrBackend(languages=["de", "pt", "en"], options={})
        self.assertEqual(reader_calls, [])

        with (
            TemporaryDirectory() as temporary_directory,
            patch.dict(sys.modules, {"easyocr": fake_easyocr}),
            self.assertLogs("scanwich.backends.easyocr", level="INFO") as captured_logs,
        ):
            image_path = Path(temporary_directory) / "page.png"
            page_image(image_path)
            backend.recognize(image_path)

        self.assertEqual(reader_calls, [(["de", "pt", "en"], {"gpu": False})])
        progress = "\n".join(captured_logs.output)
        self.assertIn("Importing EasyOCR", progress)
        self.assertIn("first use may download models and take a while", progress)
        self.assertIn("EasyOCR ready after", progress)
        self.assertIn("reads this page 3 times", progress)

    def test_maps_rotated_words_back_onto_the_page(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            page_image(image_path)
            reader = RecordingReader()
            regions = recognize(reader, image_path)

        by_text = {region.text: region for region in regions}
        self.assertEqual(set(by_text), {"ALPHA", "BETA", "GAMMA"})
        self.assertEqual(
            by_text["ALPHA"].polygon,
            (Point(10, 4), Point(50, 4), Point(50, 14), Point(10, 14)),
        )
        # A 90-degree clockwise turn maps (x, y) on the turned page to (y, height - x).
        self.assertEqual(
            by_text["BETA"].polygon,
            (Point(10, 26), Point(10, 14), Point(14, 14), Point(14, 26)),
        )
        # A 270-degree clockwise turn maps (x, y) to (width - y, x).
        self.assertEqual(
            by_text["GAMMA"].polygon,
            (Point(70, 4), Point(70, 16), Point(66, 16), Point(66, 4)),
        )
        turned_options = [call["options"] for call in reader.calls if call["size"] != PAGE_SIZE]
        self.assertEqual(
            turned_options,
            [
                {"detail": 1, "paragraph": False, "rotation_info": [180]},
                {"detail": 1, "paragraph": False, "rotation_info": [180]},
            ],
        )
        self.assertNotIn("rotation_info", reader.calls[0]["options"])

    def test_keeps_an_upright_word_when_a_turned_pass_reads_it_again(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            page_image(image_path)
            reader = RecordingReader()
            # The box is ALPHA after the inverse of a 90-degree turn, and slightly more confident.
            reader.turn_90 = [raw_region("ALPF", ((16, 10), (26, 10), (26, 50), (16, 50)), 0.93)]
            reader.turn_270 = []
            regions = recognize(reader, image_path)

        self.assertEqual([region.text for region in regions], ["ALPHA"])
        self.assertAlmostEqual(regions[0].confidence or 0, 0.91)

    def test_replaces_a_weak_fragment_with_the_rotated_word(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            page_image(image_path)
            reader = RecordingReader()
            reader.upright = [raw_region("B", ((10, 18), (14, 18), (14, 22), (10, 22)), 0.3)]
            reader.turn_270 = []
            regions = recognize(reader, image_path)

        self.assertEqual([region.text for region in regions], ["BETA"])
        self.assertAlmostEqual(regions[0].confidence or 0, 0.88)

    def test_keeps_one_copy_when_both_turns_find_the_same_word(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            page_image(image_path)
            reader = RecordingReader()
            reader.upright = []
            # The 270-degree box maps onto the same spot as BETA, with a lower confidence.
            reader.turn_270 = [raw_region("BETA", ((14, 70), (26, 70), (26, 66), (14, 66)), 0.7)]
            regions = recognize(reader, image_path)

        betas = [region for region in regions if region.text == "BETA"]
        self.assertEqual(len(betas), 1)
        self.assertAlmostEqual(betas[0].confidence or 0, 0.88)

    def test_maps_an_upside_down_pass(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            page_image(image_path)
            reader = RecordingReader()
            reader.turn_180 = [raw_region("TURN", ((10, 4), (50, 4), (50, 14), (10, 14)), 0.77)]
            regions = recognize(reader, image_path, rotations=[180])

        by_text = {region.text: region for region in regions}
        self.assertEqual(set(by_text), {"ALPHA", "TURN"})
        self.assertEqual(
            by_text["TURN"].polygon,
            (Point(70, 26), Point(30, 26), Point(30, 16), Point(70, 16)),
        )
        self.assertEqual(len(reader.calls), 2)

    def test_empty_rotations_read_the_page_once(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            page_image(image_path)
            reader = RecordingReader()
            regions = recognize(reader, image_path, rotations=[])

        self.assertEqual([region.text for region in regions], ["ALPHA"])
        self.assertEqual(len(reader.calls), 1)

    def test_does_not_forward_rotations_to_the_reader(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "page.png"
            page_image(image_path)
            reader = RecordingReader()
            recognize(reader, image_path, rotations=[90], gpu=True, readtext={"batch_size": 4})

        self.assertEqual(reader.reader_options, {"gpu": True})
        self.assertEqual(
            reader.calls[0]["options"],
            {"detail": 1, "paragraph": False, "batch_size": 4},
        )
        self.assertEqual(
            reader.calls[1]["options"],
            {"detail": 1, "paragraph": False, "batch_size": 4, "rotation_info": [180]},
        )

    def test_rejects_a_rotation_that_is_not_a_quarter_turn(self) -> None:
        with self.assertRaisesRegex(ValueError, "90, 180, or 270"):
            EasyOcrBackend(languages=["en"], options={"rotations": [45]})
        with self.assertRaisesRegex(TypeError, "list of 90, 180, and 270"):
            EasyOcrBackend(languages=["en"], options={"rotations": "90"})
