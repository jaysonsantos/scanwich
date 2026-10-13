from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from scanwich.ocr import LayoutBackend, available_backends, load_backend
from scanwich.pdf import PdfPipelineError
from scanwich.pipeline import (
    LATEX_SUFFIX,
    PAGE_ROTATIONS,
    convert_pdf,
    convert_to_latex,
    parse_backend_options,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanwich",
        description=(
            "Rebuild a PDF with its page images and a searchable OCR text layer, "
            "or write a PDF or a photo of a page as a LaTeX document."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="source PDF; for LaTeX output also an image or a camera raw file of one page",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="searchable PDF to create, or LaTeX document to create when the name ends in .tex",
    )
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
    parser.add_argument(
        "--dpi",
        type=int,
        help="rasterization DPI (default: infer from a full-page image, otherwise 300)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=PAGE_ROTATIONS,
        metavar="DEGREES",
        help=(
            "LaTeX output: the direction of the pages; turn each page image clockwise by "
            "0, 90, 180, or 270 degrees before recognition (default: the backend finds the "
            "direction of each page)"
        ),
    )
    parser.add_argument(
        "--flow",
        action="store_true",
        help=(
            "LaTeX output: let the text flow across the pages of the document (default: "
            "each source page is one page of the document)"
        ),
    )
    parser.add_argument(
        "--ocr-output-dir",
        type=Path,
        help="also save per-page JSON files: OCR regions, or the page layout for LaTeX output",
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
    if args.input is None or args.output is None:
        parser.error("input and output are required unless --list-backends is used")
    latex_output = args.output.suffix.lower() == LATEX_SUFFIX
    if args.rotate is not None and not latex_output:
        parser.error("--rotate applies to LaTeX output only")
    if args.flow and not latex_output:
        parser.error("--flow applies to LaTeX output only")

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
        if not latex_output:
            convert_pdf(
                args.input,
                args.output,
                backend=backend,
                dpi=args.dpi,
                ocr_output_directory=args.ocr_output_dir,
            )
        elif not isinstance(backend, LayoutBackend):
            raise TypeError(
                f"OCR backend {args.ocr_backend!r} does not recognize the layout of a page; "
                "LaTeX output needs a backend that does, for example "
                "--ocr-backend openai-compatible"
            )
        else:
            convert_to_latex(
                args.input,
                args.output,
                backend=backend,
                dpi=args.dpi,
                rotate=args.rotate,
                flow=args.flow,
                ocr_output_directory=args.ocr_output_dir,
            )
    except (FileNotFoundError, PdfPipelineError, TypeError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    return 0
