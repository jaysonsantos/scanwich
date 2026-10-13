import io
import logging
import signal
import time
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from scanwich.cli import main
from scanwich.layout import PageLayout, Paragraph
from scanwich.models import OcrRegion
from scanwich.pipeline import ConversionStopped


class RegionBackend:
    """A backend that returns text regions and no layout."""

    def recognize(self, image_path: Path) -> list[OcrRegion]:
        return []


class LayoutCapableBackend(RegionBackend):
    def __init__(self) -> None:
        self.image_sizes: list[tuple[int, int]] = []

    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        with Image.open(image_path) as image:
            self.image_sizes.append(image.size)
        return PageLayout(blocks=(Paragraph("The text of the page."),))


class UnreadPageBackend(RegionBackend):
    """A backend that reports a page that is not upright, and thus returns no blocks."""

    def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
        return PageLayout(page_top="left")


class CliTests(TestCase):
    def setUp(self) -> None:
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.photo = self.directory / "photo.png"
        Image.new("RGB", (40, 20), "white").save(self.photo)
        # `main` sets the handlers of the root logger.
        root_logger = logging.getLogger()
        handlers, level = root_logger.handlers[:], root_logger.level
        self.addCleanup(root_logger.setLevel, level)
        self.addCleanup(setattr, root_logger, "handlers", handlers)

    def run_main(self, arguments: list[str], backend: object) -> tuple[int, str]:
        """Return the exit code and the log output of one command."""
        log_output = io.StringIO()
        with (
            patch("scanwich.cli.load_backend", return_value=backend) as load_backend,
            redirect_stderr(log_output),
        ):
            exit_code = main(arguments)
        self.load_backend = load_backend
        return exit_code, log_output.getvalue()

    def test_output_name_with_the_tex_suffix_gives_a_latex_document(self) -> None:
        for name in ("page.tex", "PAGE.TEX"):
            with self.subTest(name=name):
                output = self.directory / name
                backend = LayoutCapableBackend()

                exit_code, log_output = self.run_main(
                    [
                        str(self.photo),
                        str(output),
                        "--ocr-backend",
                        "example",
                        "-l",
                        "fr",
                        "--rotate",
                        "90",
                    ],
                    backend,
                )

                self.assertEqual(exit_code, 0, log_output)
                self.assertIn("The text of the page.", output.read_text(encoding="utf-8"))
                # The option turns the page image before the backend gets it.
                self.assertEqual(backend.image_sizes, [(20, 40)])
                self.load_backend.assert_called_once_with("example", languages=["fr"], options={})
                self.assertIn(f"INFO: Wrote LaTeX document to {output.resolve()}", log_output)

    def test_latex_output_needs_a_backend_that_recognizes_layouts(self) -> None:
        output = self.directory / "page.tex"

        exit_code, log_output = self.run_main(
            [str(self.photo), str(output), "--ocr-backend", "example"], RegionBackend()
        )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "ERROR: OCR backend 'example' does not recognize the layout of a page; LaTeX output "
            "needs a backend that does, for example --ocr-backend openai-compatible",
            log_output,
        )
        self.assertFalse(output.exists())

    def test_page_without_a_usable_layout_is_a_failure(self) -> None:
        # A page must not disappear from the document with exit code 0.
        output = self.directory / "page.tex"
        for options in ([], ["--rotate", "0"]):
            with self.subTest(options=options):
                exit_code, log_output = self.run_main(
                    [str(self.photo), str(output), "--ocr-backend", "example", *options],
                    UnreadPageBackend(),
                )
                self.assertEqual(exit_code, 1)
                self.assertIn("ERROR: page 1 has no usable layout: the backend", log_output)
                self.assertFalse(output.exists())

    def test_other_output_names_give_a_searchable_pdf(self) -> None:
        backend = LayoutCapableBackend()
        output = self.directory / "page.pdf"
        with (
            patch("scanwich.cli.convert_pdf") as convert_pdf,
            patch("scanwich.cli.convert_to_latex") as convert_to_latex,
        ):
            exit_code, _ = self.run_main(
                [str(self.photo), str(output), "--dpi", "150", "--ocr-output-dir", "ocr"], backend
            )

        self.assertEqual(exit_code, 0)
        convert_pdf.assert_called_once_with(
            self.photo, output, backend=backend, dpi=150, ocr_output_directory=Path("ocr")
        )
        convert_to_latex.assert_not_called()

    def test_passes_the_options_to_the_latex_conversion(self) -> None:
        backend = LayoutCapableBackend()
        output = self.directory / "page.tex"
        with patch("scanwich.cli.convert_to_latex") as convert_to_latex:
            exit_code, _ = self.run_main(
                [str(self.photo), str(output), "--dpi", "150", "--ocr-output-dir", "layout"],
                backend,
            )

        self.assertEqual(exit_code, 0)
        # Without `--rotate`, the backend finds the direction of each page. Without
        # `--flow`, each source page is one page of the document. A page with a saved
        # layout needs no request, and the backend reads 4 pages at the same time.
        convert_to_latex.assert_called_once_with(
            [self.photo],
            output,
            backend=backend,
            dpi=150,
            rotate=None,
            flow=False,
            ocr_output_directory=Path("layout"),
            ignore_saved_layouts=False,
            parallel_pages=4,
            # These signals stop the conversion as Ctrl-C does.
            stop_signals=(signal.SIGTERM, signal.SIGHUP),
        )
        with patch("scanwich.cli.convert_to_latex") as convert_to_latex:
            exit_code, _ = self.run_main(
                [
                    str(self.photo),
                    str(output),
                    "--flow",
                    "--parallel-pages",
                    "1",
                    "--ocr-output-dir",
                    "layout",
                    "--ignore-saved-layouts",
                ],
                backend,
            )
        self.assertEqual(exit_code, 0)
        options = convert_to_latex.call_args.kwargs
        self.assertTrue(options["flow"])
        self.assertTrue(options["ignore_saved_layouts"])
        self.assertEqual(options["parallel_pages"], 1)

    def test_more_than_one_input_gives_one_latex_document(self) -> None:
        # The pattern of a shell gives the files of a book as many arguments.
        second = self.directory / "photo-2.png"
        Image.new("RGB", (60, 20), "white").save(second)
        output = self.directory / "book.tex"
        backend = LayoutCapableBackend()

        exit_code, log_output = self.run_main(
            [str(self.photo), str(second), str(output), "--parallel-pages", "1"], backend
        )

        self.assertEqual(exit_code, 0, log_output)
        self.assertEqual(backend.image_sizes, [(40, 20), (60, 20)])
        self.assertEqual(output.read_text(encoding="utf-8").count("\\begin{sourcepage}"), 2)
        # A log record of the work for a page names the page.
        self.assertIn(
            "INFO: [page 2/2] Recognizing the layout of page 2/2 with LayoutCapableBackend",
            log_output,
        )
        self.assertIn(f"INFO: [page 1/2] Page 1/2 is {self.photo}\n", log_output)
        self.assertIn(f"INFO: Wrote LaTeX document to {output.resolve()}", log_output)

    def test_option_can_be_between_the_names(self) -> None:
        # With Python 3.12 and later, `in.pdf --dpi 72 out.pdf` was a valid command
        # before the command took more than one input.
        backend = LayoutCapableBackend()
        second = self.directory / "photo-2.png"
        Image.new("RGB", (60, 20), "white").save(second)
        pdf, tex = self.directory / "page.pdf", self.directory / "book.tex"
        photo = str(self.photo)
        commands = {
            "searchable PDF": ([photo, "--dpi", "150", str(pdf)], [self.photo], pdf),
            "searchable PDF with a backend option": (
                [photo, "--backend-option", "gpu=true", str(pdf), "-l", "de", "pt"],
                [self.photo],
                pdf,
            ),
            "one input": ([photo, "--rotate", "0", str(tex)], [self.photo], tex),
            "option after the first input": (
                [photo, "--rotate", "0", str(second), str(tex)],
                [self.photo, second],
                tex,
            ),
            "option before the output": (
                [photo, str(second), "--parallel-pages", "2", str(tex), "-l", "fr"],
                [self.photo, second],
                tex,
            ),
            "language codes that end before the names": (
                ["-l", "fr", "en", "--", photo, str(second), str(tex)],
                [self.photo, second],
                tex,
            ),
        }
        for name, (arguments, inputs, output) in commands.items():
            with (
                self.subTest(command=name),
                patch("scanwich.cli.convert_pdf") as convert_pdf,
                patch("scanwich.cli.convert_to_latex") as convert_to_latex,
            ):
                exit_code, log_output = self.run_main(arguments, backend)
                self.assertEqual(exit_code, 0, log_output)
                if output is pdf:
                    convert_to_latex.assert_not_called()
                    self.assertEqual(convert_pdf.call_args.args, (self.photo, pdf))
                else:
                    convert_pdf.assert_not_called()
                    self.assertEqual(convert_to_latex.call_args.args, (inputs, tex))
        self.assertEqual(self.load_backend.call_args.kwargs["languages"], ["fr", "en"])

    def test_name_of_a_file_is_not_a_language_code(self) -> None:
        # `-l` takes each word after it. Without this check, `a.pdf b.pdf -l de book.tex`
        # writes the searchable PDF of `a.pdf` to `b.pdf`.
        commands = [
            [str(self.photo), "other.pdf", "-l", "de", "book.tex"],
            [str(self.photo), "-l", "fr", "pages/page.png", "book.tex"],
            [str(self.photo), "-l", "fr", "book.pdf"],
        ]
        for arguments in commands:
            usage = io.StringIO()
            with (
                self.subTest(arguments=arguments[1:]),
                patch("scanwich.cli.convert_pdf") as convert_pdf,
                redirect_stderr(usage),
                self.assertRaises(SystemExit) as raised,
            ):
                main(arguments)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("-l takes language codes, and '", usage.getvalue())
            self.assertIn("put -- after the last language code", usage.getvalue())
            convert_pdf.assert_not_called()

    def test_directory_that_the_command_cannot_write_to_is_one_error_line(self) -> None:
        a_file = self.directory / "a-file"
        a_file.write_text("not a directory")
        backend = LayoutCapableBackend()
        commands = {
            "the directory of the document": [str(self.photo), str(a_file / "book.tex")],
            "the layout directory": [
                str(self.photo),
                str(self.directory / "book.tex"),
                "--ocr-output-dir",
                str(a_file / "layouts"),
            ],
        }
        for name, arguments in commands.items():
            with self.subTest(directory=name):
                exit_code, log_output = self.run_main(arguments, backend)
                self.assertEqual(exit_code, 1)
                self.assertIn(f"ERROR: Scanwich cannot write to {name} {a_file}", log_output)
                self.assertNotIn("Traceback", log_output)
        # The command stops before the backend gets a page.
        self.assertEqual(backend.image_sizes, [])

    def test_stop_of_the_user_is_not_a_traceback(self) -> None:
        output = self.directory / "page.tex"
        with patch("scanwich.cli.convert_to_latex", side_effect=KeyboardInterrupt):
            exit_code, log_output = self.run_main(
                [str(self.photo), str(output), "--ocr-output-dir", "layout"],
                LayoutCapableBackend(),
            )
        self.assertEqual(exit_code, 130)
        self.assertIn(
            "ERROR: The conversion stopped before its end, so there is no new document. The "
            "layouts that are complete are in layout: the same command sends requests only "
            "for the other pages.",
            log_output,
        )
        with patch("scanwich.cli.convert_to_latex", side_effect=KeyboardInterrupt):
            exit_code, log_output = self.run_main(
                [str(self.photo), str(output)], LayoutCapableBackend()
            )
        self.assertEqual(exit_code, 130)
        self.assertIn("so there is no new document.\n", log_output)

    def test_signals_of_a_closed_terminal_and_of_the_system_stop_as_ctrl_c_does(self) -> None:
        # A book needs 40 minutes or more. A closed terminal and a lost connection are
        # probable in that time.
        output = self.directory / "page.tex"
        for number, name, exit_code in (
            (signal.SIGTERM, "SIGTERM", 143),
            (signal.SIGHUP, "SIGHUP", 129),
        ):
            with (
                self.subTest(signal=name),
                patch("scanwich.cli.convert_to_latex", side_effect=ConversionStopped(number)),
            ):
                code, log_output = self.run_main(
                    [str(self.photo), str(output), "--ocr-output-dir", "layout"],
                    LayoutCapableBackend(),
                )
                self.assertEqual(code, exit_code)
                self.assertIn(
                    f"ERROR: The conversion stopped before its end (signal {name}), so there "
                    "is no new document. The layouts that are complete are in layout: the same "
                    "command sends requests only for the other pages.",
                    log_output,
                )
                self.assertNotIn("Traceback", log_output)

    def test_real_stop_signal_during_a_conversion(self) -> None:
        class StoppedBackend(LayoutCapableBackend):
            """A backend whose second page gets the signal while its request is in work."""

            def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout:
                layout = super().recognize_layout(image_path, upright=upright)
                if len(self.image_sizes) == 2:
                    signal.raise_signal(signal.SIGTERM)
                    time.sleep(0.05)
                return layout

        photos = [self.photo]
        for number in (2, 3, 4):
            photos.append(self.directory / f"photo-{number}.png")
            Image.new("RGB", (40 + number, 20), "white").save(photos[-1])
        output, layouts = self.directory / "out" / "book.tex", self.directory / "layouts"
        handler = signal.getsignal(signal.SIGTERM)
        backend = StoppedBackend()

        exit_code, log_output = self.run_main(
            [
                *map(str, photos),
                str(output),
                "--ocr-output-dir",
                str(layouts),
                "--parallel-pages",
                "1",
            ],
            backend,
        )

        self.assertEqual(exit_code, 143, log_output)
        self.assertIn("ERROR: The conversion stopped before its end (signal SIGTERM)", log_output)
        self.assertNotIn("Traceback", log_output)
        # The page before the signal is saved, and no later page got a request.
        self.assertEqual(len(backend.image_sizes), 2)
        self.assertEqual(len(list(layouts.glob("*.json"))), 1)
        # No document, and no hidden directory for new illustrations stays.
        self.assertEqual(list(output.parent.iterdir()), [])
        # The handler of the caller is in place again.
        self.assertIs(signal.getsignal(signal.SIGTERM), handler)

    def test_platform_with_no_sighup_has_the_other_stop_signal(self) -> None:
        output = self.directory / "page.tex"
        sighup = signal.SIGHUP
        del signal.SIGHUP
        try:
            with patch("scanwich.cli.convert_to_latex") as convert_to_latex:
                exit_code, _ = self.run_main([str(self.photo), str(output)], LayoutCapableBackend())
        finally:
            signal.SIGHUP = sighup
        self.assertEqual(exit_code, 0)
        self.assertEqual(convert_to_latex.call_args.kwargs["stop_signals"], (signal.SIGTERM,))

    def test_rejects_commands_that_have_no_meaning(self) -> None:
        cases = {
            "--rotate applies to LaTeX output only": [
                str(self.photo),
                "page.pdf",
                "--rotate",
                "90",
            ],
            "argument --rotate: invalid choice": [str(self.photo), "page.tex", "--rotate", "45"],
            "--flow applies to LaTeX output only": [str(self.photo), "page.pdf", "--flow"],
            "input and output are required": [str(self.photo)],
            "a searchable PDF has one input file": [str(self.photo), str(self.photo), "page.pdf"],
            "--parallel-pages applies to LaTeX output only": [
                str(self.photo),
                "page.pdf",
                "--parallel-pages",
                "2",
            ],
            "the number of parallel pages must be 1 or more": [
                str(self.photo),
                "page.tex",
                "--parallel-pages",
                "0",
            ],
            "--ignore-saved-layouts applies to LaTeX output only": [
                str(self.photo),
                "page.pdf",
                "--ocr-output-dir",
                "ocr",
                "--ignore-saved-layouts",
            ],
            "--ignore-saved-layouts needs --ocr-output-dir": [
                str(self.photo),
                "page.tex",
                "--ignore-saved-layouts",
            ],
        }
        for message, arguments in cases.items():
            with self.subTest(message=message):
                usage = io.StringIO()
                with redirect_stderr(usage), self.assertRaises(SystemExit) as raised:
                    main(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, usage.getvalue())
