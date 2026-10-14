from __future__ import annotations

import argparse
import logging
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from scanwich.images import RAW_SUFFIXES
from scanwich.ocr import LayoutBackend, available_backends, load_backend
from scanwich.pdf import PdfPipelineError
from scanwich.pipeline import (
    DEFAULT_PARALLEL_PAGES,
    LATEX_SUFFIX,
    LOG_PAGE,
    PAGE_ROTATIONS,
    convert_pdf,
    convert_to_latex,
    parse_backend_options,
)

# The exit code of a command that a signal stops is this number and the signal number:
# 130 for Ctrl-C, 143 for SIGTERM, and 129 for SIGHUP.
EXIT_CODE_SIGNAL_BASE = 128
# The language codes when the command has no -l.
DEFAULT_LANGUAGES = ("de", "pt", "en")
# The suffix of a PDF file, for the check of the language codes.
PDF_SUFFIX = ".pdf"
# These signals stop LaTeX output as Ctrl-C does. SIGHUP comes when the terminal closes
# or a connection is lost, which is probable in the hours that a book needs.
STOP_SIGNAL_NAMES = ("SIGTERM", "SIGHUP")

logger = logging.getLogger(__name__)


class _PageFilter(logging.Filter):
    """Give each log record the page that its work is for.

    LaTeX output reads some pages at the same time, so the records of the pages mix.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        page = LOG_PAGE.get()
        record.page = f"[{page}] " if page else ""
        return True


def _is_a_file_name(value: str) -> bool:
    """Return True for a value that is a file, or that has the suffix of a file of the command.

    The suffixes are those of an output and of each input: `.tex`, `.pdf`, the image
    formats of Pillow, and the camera raw formats.
    """
    Image.init()
    suffixes = {LATEX_SUFFIX, PDF_SUFFIX, *RAW_SUFFIXES, *Image.registered_extensions()}
    return (
        Path(value).suffix.lower() in {suffix.lower() for suffix in suffixes}
        or Path(value).is_file()
    )


def _stop_signals() -> tuple[int, ...]:
    """Return the stop signals that this platform has. Windows has no SIGHUP."""
    return tuple(getattr(signal, name) for name in STOP_SIGNAL_NAMES if hasattr(signal, name))


def _parallel_pages(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("the number of parallel pages must be 1 or more")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanwich",
        usage="%(prog)s [options] input [input ...] output",
        description=(
            "Rebuild a PDF with its page images and a searchable OCR text layer, "
            "or write PDFs and photos of pages as one LaTeX document."
        ),
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        metavar="input ... output",
        help=(
            "the source PDF, then the searchable PDF to create. When the last name ends in "
            ".tex, it is the LaTeX document to create, and the names before it are the "
            "sources: PDFs, images, and camera raw files of one page each, in the sequence "
            "of the pages"
        ),
    )
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        metavar="LANG",
        help=f"OCR language codes (default: {' '.join(DEFAULT_LANGUAGES)})",
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
        "--parallel-pages",
        type=_parallel_pages,
        metavar="NUMBER",
        help=(
            "LaTeX output: the number of pages that the backend reads at the same time "
            f"(default: {DEFAULT_PARALLEL_PAGES})"
        ),
    )
    parser.add_argument(
        "--ocr-output-dir",
        type=Path,
        help=(
            "also save per-page JSON files: OCR regions, or the page layout for LaTeX "
            "output. LaTeX output sends no request for a page that has a saved layout there"
        ),
    )
    parser.add_argument(
        "--ignore-saved-layouts",
        action="store_true",
        help=(
            "LaTeX output: send requests for each page, also for a page that has a saved "
            "layout in the directory of --ocr-output-dir"
        ),
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="list installed OCR backends and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = sys.argv[1:] if argv is None else list(argv)
    if sys.version_info < (3, 12) and "--" in arguments:
        # With Python 3.11, `parse_intermixed_args` refuses a name after `--` that starts
        # with `-`. The names after `--` are together, so the usual function is correct.
        args = parser.parse_args(arguments)
    else:
        # An option can be between the names: `in.pdf --dpi 72 out.pdf`. The usual parse
        # function takes the names only when they are together: they are one argument.
        args = parser.parse_intermixed_args(arguments)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(page)s%(message)s",
        stream=sys.stderr,
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_PageFilter())
    if args.list_backends:
        for name in available_backends():
            print(name)
        return 0
    # Only the values of -l get the check: a file named `en` is no reason to refuse a
    # command without -l.
    for language in args.languages or ():
        # `-l` takes each word after it, so `a.pdf b.pdf -l de book.tex` has `b.pdf` as
        # its output. A value for a backend can have a dot or a slash, for example
        # `Latin (med.)` for a backend that puts the value in its prompt.
        if _is_a_file_name(language):
            parser.error(
                f"-l takes language codes, and {language!r} looks like the name of a file. "
                "Put the names of the files before -l, or put -- after the last language code"
            )
    if args.languages is None:
        args.languages = list(DEFAULT_LANGUAGES)
    if len(args.paths) < 2:
        parser.error("input and output are required unless --list-backends is used")
    *inputs, output = args.paths
    latex_output = output.suffix.lower() == LATEX_SUFFIX
    if args.rotate is not None and not latex_output:
        parser.error("--rotate applies to LaTeX output only")
    if args.flow and not latex_output:
        parser.error("--flow applies to LaTeX output only")
    if args.parallel_pages is not None and not latex_output:
        parser.error("--parallel-pages applies to LaTeX output only")
    if args.ignore_saved_layouts and not latex_output:
        parser.error("--ignore-saved-layouts applies to LaTeX output only")
    if args.ignore_saved_layouts and args.ocr_output_dir is None:
        parser.error("--ignore-saved-layouts needs --ocr-output-dir")
    if len(inputs) > 1 and not latex_output:
        parser.error(
            "a searchable PDF has one input file; more than one input file is for LaTeX "
            "output, with an output name that ends in .tex"
        )

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
                inputs[0],
                output,
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
            try:
                convert_to_latex(
                    inputs,
                    output,
                    backend=backend,
                    dpi=args.dpi,
                    rotate=args.rotate,
                    flow=args.flow,
                    ocr_output_directory=args.ocr_output_dir,
                    ignore_saved_layouts=args.ignore_saved_layouts,
                    parallel_pages=args.parallel_pages or DEFAULT_PARALLEL_PAGES,
                    stop_signals=_stop_signals(),
                )
            except KeyboardInterrupt as stop:
                # Ctrl-C, or one of the stop signals.
                signal_number = getattr(stop, "signal_number", signal.SIGINT)
                signal_name = (
                    ""
                    if signal_number == signal.SIGINT
                    else f" (signal {signal.Signals(signal_number).name})"
                )
                if getattr(stop, "document_complete", False):
                    # The stop came in the install of the result, which continued to its
                    # end. The exit code tells a script that a stop came.
                    logger.warning(
                        "The stop%s came after the new document was complete: %s and its "
                        "illustrations are the new result",
                        signal_name,
                        output,
                    )
                    return EXIT_CODE_SIGNAL_BASE + signal_number
                logger.error(
                    "The conversion stopped before its end%s, so there is no new document.%s",
                    signal_name,
                    ""
                    if args.ocr_output_dir is None
                    else (
                        f" The layouts that are complete are in {args.ocr_output_dir}: the "
                        "same command sends requests only for the other pages."
                    ),
                )
                return EXIT_CODE_SIGNAL_BASE + signal_number
    except (FileNotFoundError, PdfPipelineError, TypeError, ValueError, RuntimeError) as error:
        logger.error("%s", error)
        return 1
    return 0
