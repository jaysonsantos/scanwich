import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from scanwich.images import (
    UPRIGHT_TURNS,
    crop_box,
    is_pdf,
    read_photo,
    turn_clockwise,
)
from scanwich.layout import PAGE_SIDES, Box

EXIF_ORIENTATION_TAG = 0x0112
# A viewer turns an image with this orientation 90 degrees clockwise.
EXIF_ORIENTATION_TURN_CLOCKWISE = 6
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


class FakeRawPixels(bytes):
    """Bytes with the array interface that the result of rawpy has."""

    def __new__(cls, width: int, height: int, color: tuple[int, int, int]) -> "FakeRawPixels":
        pixels = super().__new__(cls, bytes(color) * (width * height))
        pixels.__array_interface__ = {
            "shape": (height, width, 3),
            "typestr": "|u1",
            "version": 3,
        }
        return pixels


class FakeLibRawError(Exception):
    pass


class FakeRaw:
    def __init__(self, pixels: FakeRawPixels | None, calls: list[dict[str, Any]]) -> None:
        self._pixels = pixels
        self._calls = calls

    def __enter__(self) -> "FakeRaw":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def postprocess(self, **options: Any) -> FakeRawPixels:
        self._calls.append(options)
        if self._pixels is None:
            raise FakeLibRawError("data error")
        return self._pixels


def fake_rawpy(pixels: FakeRawPixels | None, calls: list[dict[str, Any]]) -> SimpleNamespace:
    """Return a module that decodes each file to the pixels, or fails when there are none."""
    return SimpleNamespace(
        imread=lambda path: FakeRaw(pixels, calls),
        LibRawError=FakeLibRawError,
    )


class ReadPhotoTests(TestCase):
    def setUp(self) -> None:
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)

    def test_recognizes_a_pdf_by_its_first_bytes(self) -> None:
        pdf = self.directory / "scan.bin"
        pdf.write_bytes(b"%PDF-1.7\n")
        image = self.directory / "page.pdf"
        Image.new("RGB", (4, 4)).save(image, format="PNG")
        self.assertTrue(is_pdf(pdf))
        self.assertFalse(is_pdf(image))

    def test_turns_a_photo_as_its_exif_orientation_tells(self) -> None:
        path = self.directory / "photo.jpg"
        photo = Image.new("RGB", (40, 20), RED)
        exif = Image.Exif()
        exif[EXIF_ORIENTATION_TAG] = EXIF_ORIENTATION_TURN_CLOCKWISE
        photo.save(path, exif=exif)

        image = read_photo(path)

        self.assertEqual((image.size, image.mode), ((20, 40), "RGB"))
        self.assertNotIn(EXIF_ORIENTATION_TAG, image.getexif())

    def test_replaces_transparent_pixels_with_white(self) -> None:
        path = self.directory / "page.png"
        page = Image.new("RGBA", (2, 1), (*RED, 255))
        page.putpixel((1, 0), (0, 0, 0, 0))
        page.save(path)

        image = read_photo(path)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual([image.getpixel((0, 0)), image.getpixel((1, 0))], [RED, WHITE])

    def test_keeps_the_gray_levels_of_a_16_bit_image(self) -> None:
        path = self.directory / "page.png"
        Image.new("I;16", (2, 2), 0x8000).save(path)

        image = read_photo(path)

        self.assertEqual((image.mode, image.getpixel((0, 0))), ("L", 0x80))

    def test_decodes_a_camera_raw_file_at_full_resolution(self) -> None:
        # Pillow cannot read this file, or reads only its small preview image.
        path = self.directory / "DSC00001.ARW"
        path.write_bytes(b"II*\x00 not an image that Pillow reads")
        calls: list[dict[str, Any]] = []
        rawpy = fake_rawpy(FakeRawPixels(60, 40, GREEN), calls)

        with (
            patch.dict(sys.modules, {"rawpy": rawpy}),
            self.assertLogs("scanwich.images", level="INFO") as captured_logs,
        ):
            image = read_photo(path)

        self.assertEqual((image.size, image.mode), ((60, 40), "RGB"))
        self.assertEqual(image.getpixel((59, 39)), GREEN)
        self.assertEqual(calls, [{"use_camera_wb": True, "output_bps": 8}])
        self.assertIn("Decoding the camera raw file", captured_logs.output[0])

    def test_camera_raw_file_never_goes_to_pillow(self) -> None:
        # Pillow opens some raw formats as a TIFF file and returns the small preview image.
        path = self.directory / "photo.dng"
        Image.new("RGB", (8, 8), RED).save(path, format="TIFF")
        rawpy = fake_rawpy(FakeRawPixels(60, 40, GREEN), [])

        with patch.dict(sys.modules, {"rawpy": rawpy}):
            image = read_photo(path)

        self.assertEqual(image.size, (60, 40))

    def test_camera_raw_file_needs_the_rawpy_package(self) -> None:
        path = self.directory / "photo.nef"
        path.write_bytes(b"raw")
        # `None` makes the import fail, as it does when the package is not installed.
        with (
            patch.dict(sys.modules, {"rawpy": None}),
            self.assertRaisesRegex(RuntimeError, r"requires the 'rawpy' package.*scanwich\[raw\]"),
        ):
            read_photo(path)

    def test_reports_a_raw_file_that_libraw_cannot_decode(self) -> None:
        path = self.directory / "photo.cr2"
        path.write_bytes(b"raw")
        with (
            patch.dict(sys.modules, {"rawpy": fake_rawpy(None, [])}),
            self.assertRaisesRegex(RuntimeError, "LibRaw could not decode .*photo.cr2: data error"),
        ):
            read_photo(path)

    def test_tries_libraw_for_an_unknown_file(self) -> None:
        path = self.directory / "photo.unknown"
        path.write_bytes(b"not an image")
        rawpy = fake_rawpy(FakeRawPixels(6, 4, GREEN), [])
        with patch.dict(sys.modules, {"rawpy": rawpy}):
            self.assertEqual(read_photo(path).size, (6, 4))

        with (
            patch.dict(sys.modules, {"rawpy": fake_rawpy(None, [])}),
            self.assertRaisesRegex(
                RuntimeError,
                "photo.unknown is not a PDF and not an image that Pillow can read; "
                "LibRaw could not decode",
            ),
        ):
            read_photo(path)

    def test_reports_a_damaged_image(self) -> None:
        path = self.directory / "page.png"
        good = self.directory / "good.png"
        Image.new("RGB", (64, 64), RED).save(good)
        path.write_bytes(good.read_bytes()[:60])
        with self.assertRaisesRegex(RuntimeError, "could not read the image .*page.png"):
            read_photo(path)


class TurnAndCropTests(TestCase):
    def marked_image(self) -> Image.Image:
        """Return an image of 4x2 pixels with a red pixel at its top-left corner."""
        image = Image.new("RGB", (4, 2), WHITE)
        image.putpixel((0, 0), RED)
        return image

    def test_turns_an_image_clockwise(self) -> None:
        image = self.marked_image()
        # The size after the turn, and the position of the pixel from the top-left corner.
        cases = {
            0: ((4, 2), (0, 0)),
            90: ((2, 4), (1, 0)),
            180: ((4, 2), (3, 1)),
            270: ((2, 4), (0, 3)),
        }
        for degrees, (size, position) in cases.items():
            with self.subTest(degrees=degrees):
                turned = turn_clockwise(image, degrees)
                self.assertEqual(turned.size, size)
                self.assertEqual(turned.getpixel(position), RED)
        with self.assertRaisesRegex(ValueError, "0, 90, 180, or 270"):
            turn_clockwise(image, 45)

    def test_upright_turn_puts_the_top_of_the_page_at_the_top(self) -> None:
        self.assertEqual(set(UPRIGHT_TURNS), set(PAGE_SIDES))
        # The red pixel is at the top-left corner of an upright page. Each image shows that
        # page with its top at a different side.
        page = self.marked_image()
        views = {
            "top": page,
            "right": turn_clockwise(page, 90),
            "bottom": turn_clockwise(page, 180),
            "left": turn_clockwise(page, 270),
        }
        for page_top, view in views.items():
            with self.subTest(page_top=page_top):
                upright = turn_clockwise(view, UPRIGHT_TURNS[page_top])
                self.assertEqual(upright.size, page.size)
                self.assertEqual(upright.getpixel((0, 0)), RED)

    def test_crops_a_box_of_the_grid(self) -> None:
        image = Image.new("RGB", (200, 100), WHITE)
        image.paste(RED, (50, 20, 150, 60))

        crop = crop_box(image, Box(250, 200, 750, 600))

        self.assertEqual(crop.size, (100, 40))
        self.assertEqual({crop.getpixel((0, 0)), crop.getpixel((99, 39))}, {RED})

    def test_crop_margin_stays_in_the_image(self) -> None:
        image = Image.new("RGB", (200, 100), WHITE)
        self.assertEqual(crop_box(image, Box(250, 200, 750, 600), margin=10).size, (104, 42))
        self.assertEqual(crop_box(image, Box(0, 0, 1000, 1000), margin=10).size, (200, 100))

    def test_crop_of_a_small_image_has_a_pixel(self) -> None:
        image = Image.new("RGB", (2, 2), WHITE)
        for box in (Box(0, 0, 1, 1), Box(999, 999, 1000, 1000), Box(400, 400, 401, 401)):
            with self.subTest(box=box):
                self.assertEqual(crop_box(image, box).size, (1, 1))
