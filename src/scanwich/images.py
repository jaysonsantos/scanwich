"""Read a photo of a page, and turn and crop a page image."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from scanwich.layout import GRID_MAX, PAGE_TOP_UPRIGHT, Box

PDF_MAGIC = b"%PDF-"
# The file suffixes of camera raw formats. Pillow opens some of these files as a TIFF file
# and then returns the small preview image, so Pillow does not get them.
RAW_SUFFIXES = frozenset(
    {
        ".3fr",
        ".arw",
        ".cr2",
        ".cr3",
        ".crw",
        ".dcr",
        ".dng",
        ".erf",
        ".iiq",
        ".kdc",
        ".mef",
        ".mos",
        ".mrw",
        ".nef",
        ".nrw",
        ".orf",
        ".pef",
        ".raf",
        ".raw",
        ".rw2",
        ".rwl",
        ".sr2",
        ".srf",
        ".srw",
        ".x3f",
    }
)
RAW_BITS_PER_SAMPLE = 8
# The Pillow operation that turns an image clockwise by each angle. Pillow names its
# operations by the angle counterclockwise.
CLOCKWISE_TURNS = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}
# The clockwise angle that makes a page upright, for each side of the image where the top
# of the page can be.
UPRIGHT_TURNS = {PAGE_TOP_UPRIGHT: 0, "right": 270, "bottom": 180, "left": 90}
# The modes that a page image file has.
PAGE_IMAGE_MODES = frozenset({"L", "RGB"})
# The Pillow modes with integer values start with this name. A gray image with 16 bits for
# each pixel opens in such a mode.
INTEGER_MODE = "I"
SIXTEEN_BIT_GRAY_STEPS_PER_8_BIT_STEP = 256
ALPHA_BAND = "A"
# Pillow puts the transparent palette entry or the transparent color of a file in this field.
TRANSPARENCY_INFO = "transparency"

logger = logging.getLogger(__name__)


def is_pdf(path: Path) -> bool:
    """Return True for a file that starts as a PDF file starts."""
    with path.open("rb") as file:
        return file.read(len(PDF_MAGIC)) == PDF_MAGIC


def read_photo(path: Path) -> Image.Image:
    """Return the full-resolution image of an image file, upright as its metadata tells.

    The image has the mode `L` or `RGB`. White replaces transparent pixels.
    """
    if path.suffix.lower() in RAW_SUFFIXES:
        return _read_raw(path)
    try:
        with Image.open(path) as image:
            # The result is a copy, so it does not need the open file.
            return _page_mode_image(ImageOps.exif_transpose(image))
    except UnidentifiedImageError:
        # A camera raw file can have a suffix that is not in the list.
        try:
            return _read_raw(path)
        except RuntimeError as error:
            raise RuntimeError(
                f"{path} is not a PDF and not an image that Pillow can read; {error}"
            ) from error
    except OSError as error:
        raise RuntimeError(f"could not read the image {path}: {error}") from error


def _read_raw(path: Path) -> Image.Image:
    """Return the image that LibRaw makes from all the sensor data of a camera raw file."""
    try:
        import rawpy
    except ImportError as error:
        raise RuntimeError(
            "a camera raw file requires the 'rawpy' package: pip install 'scanwich[raw]'"
        ) from error

    logger.info("Decoding the camera raw file %s with LibRaw", path)
    try:
        with rawpy.imread(str(path)) as raw:
            pixels = raw.postprocess(use_camera_wb=True, output_bps=RAW_BITS_PER_SAMPLE)
    except (OSError, rawpy.LibRawError) as error:
        raise RuntimeError(f"LibRaw could not decode {path}: {error}") from error
    return Image.fromarray(pixels)


def _page_mode_image(image: Image.Image) -> Image.Image:
    """Return the image in a mode of a page image file."""
    if image.mode in PAGE_IMAGE_MODES:
        return image
    if image.mode.startswith(INTEGER_MODE):
        # Pillow does not scale these values in a conversion. It limits them to 255.
        to_8_bits = 1 / SIXTEEN_BIT_GRAY_STEPS_PER_8_BIT_STEP
        return image.point(lambda value: value * to_8_bits).convert("L")
    if ALPHA_BAND not in image.getbands() and TRANSPARENCY_INFO not in image.info:
        return image.convert("L" if image.mode == "1" else "RGB")
    rgba = image.convert("RGBA")
    opaque = Image.new("RGB", rgba.size, "white")
    opaque.paste(rgba, mask=rgba)
    return opaque


def turn_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    """Return the image turned clockwise by 0, 90, 180, or 270 degrees."""
    if degrees == 0:
        return image
    if degrees not in CLOCKWISE_TURNS:
        raise ValueError("a page image turns by 0, 90, 180, or 270 degrees")
    return image.transpose(CLOCKWISE_TURNS[degrees])


def crop_box(image: Image.Image, box: Box, *, margin: float = 0.0) -> Image.Image:
    """Return the part of the image that a box of the 0-to-1000 grid has.

    `margin` makes the part larger at each side. It is a distance on the same grid.
    """
    scale_x = image.width / GRID_MAX
    scale_y = image.height / GRID_MAX
    left = max(0, math.floor((box.left - margin) * scale_x))
    top = max(0, math.floor((box.top - margin) * scale_y))
    right = min(image.width, math.ceil((box.right + margin) * scale_x))
    bottom = min(image.height, math.ceil((box.bottom + margin) * scale_y))
    # A box is not empty, but a small image can have less than one pixel for it.
    right = max(right, min(left + 1, image.width))
    bottom = max(bottom, min(top + 1, image.height))
    left = min(left, right - 1)
    top = min(top, bottom - 1)
    return image.crop((left, top, right, bottom))
