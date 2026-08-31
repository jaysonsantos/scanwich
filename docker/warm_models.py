"""Download EasyOCR models and exercise them while building the container image."""

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from scanwich.backends.easyocr import EasyOcrBackend


def main() -> None:
    backend = EasyOcrBackend(
        languages=["de", "pt", "en"],
        options={
            "gpu": False,
            "model_storage_directory": "/opt/easyocr/model",
        },
    )
    with TemporaryDirectory(prefix="scanwich-model-warmup-") as temporary_directory:
        image_path = Path(temporary_directory) / "warmup.png"
        image = Image.new("RGB", (640, 160), "white")
        ImageDraw.Draw(image).text((40, 70), "Scanwich OCR model warmup", fill="black")
        image.save(image_path)
        regions = backend.recognize(image_path)
    print(f"Scanwich EasyOCR warmup complete ({len(regions)} regions)")


if __name__ == "__main__":
    main()
