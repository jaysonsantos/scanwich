from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from scanwich.ocr import available_backends, load_backend
from scanwich.pdf import PdfPipelineError
from scanwich.pipeline import convert_pdf, parse_backend_options

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanwich",
        description="Rebuild a PDF with its page images and a searchable OCR text layer.",
    )
    parser.add_argument("input_pdf", type=Path, nargs="?", help="source PDF")
    parser.add_argument("output_pdf", type=Path, nargs="?", help="searchable PDF to create")
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        default=["de", "pt", "en"],
        metavar="LANG",
        help="OCR language codes (default: de pt en)",
    )
    parser.add_argument(
        "--ocr-backend",
        default="easyocr",
        metavar="NAME",
        help="registered OCR backend (default: easyocr)",
    )
    parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="backend setting; VALUE accepts JSON (repeatable)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="rasterization DPI (default: 300)")
    parser.add_argument(
        "--ocr-output-dir",
        type=Path,
        help="also save normalized per-page OCR JSON files",
    )
    parser.add_argument(
        "--magick-command",
        default="magick",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="list installed OCR backends and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    if args.list_backends:
        for name in available_backends():
            print(name)
        return 0
    if args.input_pdf is None or args.output_pdf is None:
        parser.error("input_pdf and output_pdf are required unless --list-backends is used")

    try:
        options = parse_backend_options(args.backend_option)
        logger.info(
            "Using OCR backend %s with languages: %s",
            args.ocr_backend,
            ", ".join(args.languages),
        )
        backend = load_backend(
            args.ocr_backend,
            languages=args.languages,
            options=options,
        )
        convert_pdf(
            args.input_pdf,
            args.output_pdf,
            backend=backend,
            dpi=args.dpi,
            ocr_output_directory=args.ocr_output_dir,
            magick_command=args.magick_command,
        )
    except (FileNotFoundError, PdfPipelineError, TypeError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    return 0
