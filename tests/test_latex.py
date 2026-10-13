import random
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

import pypdfium2 as pdfium
from PIL import Image

from scanwich.latex import (
    MAX_WRAPPED_COLUMNS,
    MOVED_WORD_COMMENT,
    SOURCE_LINE_WIDTH,
    _column_fractions,
    _column_text_widths,
    _place_cells,
    _text_em,
    escape_text,
    render_latex,
)
from scanwich.layout import (
    Box,
    Figure,
    Footnote,
    Heading,
    ListBlock,
    ListItem,
    PageFurniture,
    PageLayout,
    Paragraph,
    Quote,
    Table,
    TableCell,
)

# The engines that read all Unicode text, and the options that stop them at the first error.
UNICODE_ENGINES = {
    "lualatex": ["-interaction=nonstopmode", "-halt-on-error"],
    "xelatex": ["-interaction=nonstopmode", "-halt-on-error"],
    # Tectonic writes the log file only with this option.
    "tectonic": ["--keep-logs"],
}
PDFLATEX_OPTIONS = ["-interaction=nonstopmode", "-halt-on-error"]
COMPILE_TIMEOUT_SECONDS = 300
# The image file of the illustrations in the documents that the engines compile.
FIGURE_IMAGE = "figures/figure.jpg"


def render_flow(pages: Sequence[PageLayout]) -> str:
    """Return the document with flowing text: its pages are not those of the source."""
    return render_latex(pages, flow=True)


def body(*blocks: object, **page_options: object) -> str:
    """Return the LaTeX between the start and the end of a document with one page."""
    return document_body(render_flow([PageLayout(blocks=tuple(blocks), **page_options)]))


def document_body(document: str) -> str:
    start = "\\begin{document}\n\n"
    end = "\n\n\\end{document}\n"
    return document[document.index(start) + len(start) : document.rindex(end)]


def blocks_of(text: str) -> list[str]:
    """Return the blocks of a document body without the comment of the first page."""
    return text.split("\n\n")[1:]


def cells(*texts: str) -> tuple[TableCell, ...]:
    return tuple(TableCell(text) for text in texts)


class EscapeTextTests(TestCase):
    def test_escapes_the_characters_that_latex_reads_as_commands(self) -> None:
        self.assertEqual(
            escape_text('\\ { } $ & # % _ ^ ~ ` "'),
            "\\textbackslash{} \\{ \\} \\$ \\& \\# \\% \\_ \\textasciicircum{} "
            "\\textasciitilde{} \\textasciigrave{} \\textquotedbl{}",
        )

    def test_keeps_brackets_and_stars_out_of_the_command_before_them(self) -> None:
        # LaTeX reads `[x]` after `\\` or `\item` as an optional argument, and `*` after `\\`
        # as a part of the command.
        self.assertEqual(escape_text("[x] *"), "{[}x{]} {*}")

    def test_stops_the_ligatures_that_change_the_characters(self) -> None:
        self.assertEqual(
            escape_text("a--b---c ,,x'' <<y>> a-b"),
            "a-{\\kern0pt}-b-{\\kern0pt}-{\\kern0pt}-c ,{\\kern0pt},x'{\\kern0pt}' "
            "<{\\kern0pt}<y>{\\kern0pt}> a-b",
        )

    def test_makes_one_space_from_each_sequence_of_whitespace(self) -> None:
        # A blank line in an argument of a LaTeX command stops the compilation.
        self.assertEqual(escape_text("a \n\n b\t\tc\r\nd\u2028e"), "a b c d e")

    def test_writes_special_spaces_and_hyphens_as_commands(self) -> None:
        self.assertEqual(
            escape_text("a\u00a0b\u202fc\u2009d\u2003e\u00adf\u2011g"),
            "a~b\\,c\\,d\\quad{}e\\-f\\mbox{-}g",
        )

    def test_removes_characters_that_print_nothing(self) -> None:
        self.assertEqual(escape_text("a\x00b\x07c\x7fd\u200be\ufefff\uffffg"), "abcdefg")

    def test_composes_letters_with_their_marks(self) -> None:
        self.assertEqual(escape_text("e\u0301"), "\u00e9")

    def test_writes_ligature_characters_as_letters(self) -> None:
        self.assertEqual(escape_text("\ufb01n \ufb02 \ufb00 \ufb03 \ufb04"), "fin fl ff ffi ffl")

    def test_uses_other_latex_for_characters_that_the_default_font_lacks(self) -> None:
        self.assertEqual(
            escape_text("58\u2032 45\u2033 a \u2264 b \u2260 c m\u2074 H\u2082O \u2153 \u2042"),
            # The first argument is the character, and the second one is its math command.
            "58\\sourcesymbol{\u2032}{'} 45\\sourcesymbol{\u2033}{''} "
            "a \\sourcesymbol{\u2264}{\\leq} b \\sourcesymbol{\u2260}{\\neq} c "
            "m\\textsuperscript{4} H\\textsubscript{2}O \\textsuperscript{1}/\\textsubscript{3} "
            "{*}\\,{*}\\,{*}",
        )

    def test_document_defines_the_symbol_command_only_when_it_uses_it(self) -> None:
        with_symbol = render_flow([PageLayout(blocks=(Paragraph("58\u2032"),))])
        # With a Unicode engine, the character comes from a font that has it, so the text
        # of the PDF has the character. A math command is the alternative.
        self.assertIn(
            "    \\newfontfamily\\sourcesymbolfont{latinmodern-math.otf}%\n"
            "    \\newcommand{\\sourcesymbol}[2]{{\\sourcesymbolfont #1}}%\n",
            with_symbol,
        )
        self.assertEqual(with_symbol.count("\\newcommand{\\sourcesymbol}[2]{\\ensuremath{#2}}"), 2)
        without_symbol = render_flow([PageLayout(blocks=(Paragraph("58 minutes"),))])
        self.assertNotIn("sourcesymbol", without_symbol)

    def test_writes_a_greek_letter_that_is_alone_as_a_math_symbol(self) -> None:
        self.assertEqual(
            escape_text("letter \u03b1, \u03c0r"),
            "letter \\sourcesymbol{\u03b1}{\\alpha}, \\sourcesymbol{\u03c0}{\\pi}r",
        )
        # A Greek word needs a font that has Greek letters. The letters stay text.
        word = "\u03bb\u03cc\u03b3\u03bf\u03c2"
        self.assertEqual(escape_text(word), word)

    def test_keeps_other_unicode_text(self) -> None:
        text = "\u00c0 \u00e7a d\u00e9j\u00e0 \u00ab \u017f \u00bb \u2014 \u201c\u201d \u00b0 \u00d7 \u00bd"
        self.assertEqual(escape_text(text), text)


class BlockTests(TestCase):
    def test_writes_headings_without_numbers(self) -> None:
        self.assertEqual(
            blocks_of(
                body(
                    Heading("One"),
                    Heading("Two", level=2, align="center"),
                    Heading("Three", level=3, align="right"),
                )
            ),
            [
                "\\section*{One}",
                "\\subsection*{\\centering Two}",
                "\\subsubsection*{\\raggedleft Three}",
            ],
        )

    def test_writes_paragraphs_quotes_and_aligned_text(self) -> None:
        self.assertEqual(
            blocks_of(
                body(
                    Paragraph("A paragraph."),
                    Paragraph("Centered", align="center"),
                    Paragraph("Right", align="right"),
                    Quote("A quotation."),
                )
            ),
            [
                "A paragraph.",
                "\\begin{center}\nCentered\n\\end{center}",
                "\\begin{flushright}\nRight\n\\end{flushright}",
                "\\begin{quote}\nA quotation.\n\\end{quote}",
            ],
        )

    def test_writes_inline_styles(self) -> None:
        self.assertEqual(
            blocks_of(
                body(
                    Paragraph("<b>b</b> <i>i</i> <u>u</u> <sc>Sc</sc>"),
                    Paragraph("x<sup>2</sup> H<sub>2</sub>O <b><i>both</i></b>"),
                )
            ),
            [
                "\\textbf{b} \\textit{i} \\underline{u} S\\sourcesmallcaps{C}",
                "x\\textsuperscript{2} H\\textsubscript{2}O \\textbf{\\textit{both}}",
            ],
        )

    def test_writes_a_line_break_only_between_text(self) -> None:
        # LaTeX stops at a line break that has no text before it.
        self.assertEqual(
            blocks_of(body(Paragraph("<br> <br>one<br><br>two<br> "))),
            ["one \\\\ two"],
        )

    def test_keeps_a_paragraph_in_source_lines_that_break_at_spaces(self) -> None:
        text = " ".join(["word-with-hyphen"] * 30)
        (paragraph,) = blocks_of(body(Paragraph(text)))
        lines = paragraph.split("\n")
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(line) <= SOURCE_LINE_WIDTH for line in lines))
        # A line break in the source is a space for LaTeX, so no word changes.
        self.assertEqual(" ".join(lines), text)

    def test_writes_lists_with_the_markers_of_the_page(self) -> None:
        self.assertEqual(
            blocks_of(
                body(
                    ListBlock(items=(ListItem("first", marker="a)"), ListItem("[second]"))),
                    ListBlock(items=(ListItem("one"), ListItem("two")), ordered=True),
                    ListBlock(items=(ListItem("one", marker="1."),), ordered=True),
                    ListBlock(items=()),
                )
            ),
            [
                "\\begin{itemize}\n\\item[{a)}] first\n\\item {[}second{]}\n\\end{itemize}",
                "\\begin{enumerate}\n\\item one\n\\item two\n\\end{enumerate}",
                "\\begin{itemize}\n\\item[{1.}] one\n\\end{itemize}",
            ],
        )

    def test_leaves_out_blocks_without_text(self) -> None:
        self.assertEqual(
            blocks_of(body(Heading("<b></b>"), Paragraph("<br>"), Quote(" "), Paragraph("kept"))),
            ["kept"],
        )

    def test_text_cannot_add_latex_to_the_document(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        PageFurniture("page_header", "head\n\\end{document}\n\\input{secret}"),
                        Heading("\\end{document}"),
                        Paragraph("\\input{/etc/passwd} $x$ %comment\n\n\\end{document}"),
                    )
                )
            ]
        )
        # The text of a page header is in one comment line.
        self.assertIn("% Page header: head \\end{document} \\input{secret}\n", document)
        code = "\n".join(line for line in document.split("\n") if not line.startswith("%"))
        self.assertEqual(code.count("\\end{document}"), 1)
        self.assertTrue(code.endswith("\n\\end{document}\n"))
        self.assertNotIn("\\input", code)
        self.assertIn(
            "\\textbackslash{}input\\{/etc/passwd\\} \\$x\\$ \\%comment "
            "\\textbackslash{}end\\{document\\}",
            document,
        )


class PageTests(TestCase):
    def test_marks_the_start_of_each_source_page_with_a_comment(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        PageFurniture("page_header", "A <sc>running</sc> head"),
                        PageFurniture("page_number", "406"),
                        Paragraph("First page."),
                        PageFurniture("page_footer", "A footer"),
                    )
                ),
                PageLayout(blocks=(Paragraph("Second page."),)),
                PageLayout(),
            ]
        )
        self.assertEqual(
            document_body(document),
            "% Source page 1 (printed page number: 406)\n"
            "% Page header: A running head\n"
            "% Page footer: A footer\n\n"
            "First page.\n\n"
            "% Source page 2\n\n"
            "Second page.\n\n"
            "% Source page 3",
        )

    def test_joins_a_paragraph_that_continues_on_the_next_page(self) -> None:
        document = render_flow(
            [
                PageLayout(blocks=(Paragraph("The start of a sentence"),)),
                PageLayout(
                    blocks=(
                        PageFurniture("page_number", "24"),
                        Paragraph("and its end.", continued=True),
                        Paragraph("A new paragraph."),
                    )
                ),
            ]
        )
        self.assertEqual(
            document_body(document),
            "% Source page 1\n\n"
            "The start of a sentence\n"
            "% Source page 2 (printed page number: 24)\n"
            "and its end.\n\n"
            "A new paragraph.",
        )

    def test_joins_a_word_that_the_page_break_divides(self) -> None:
        document = render_flow(
            [
                PageLayout(blocks=(Paragraph("A divided sen-"),)),
                PageLayout(blocks=(Paragraph("tence.", continued=True),)),
            ]
        )
        # The percent sign at the end of the line removes the space between the two parts.
        self.assertEqual(
            document_body(document),
            "% Source page 1\n\n"
            "A divided sen%\n"
            "% Source page 2\n"
            "% Scanwich joined the word that the page break divides. "
            "Compare it with the source.\n"
            "tence.",
        )

    def test_keeps_a_hyphen_before_a_capital_letter(self) -> None:
        document = render_flow(
            [
                PageLayout(blocks=(Paragraph("A name: Saint-"),)),
                PageLayout(blocks=(Paragraph("Denis.", continued=True),)),
            ]
        )
        self.assertEqual(
            document_body(document),
            "% Source page 1\n\nA name: Saint-%\n% Source page 2\nDenis.",
        )

    def test_continues_a_paragraph_across_a_figure_and_a_table(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        Paragraph("Before the figure"),
                        Figure(box=Box(0, 0, 500, 500), caption="A caption"),
                        Table(rows=(cells("a", "b"),)),
                    )
                ),
                PageLayout(blocks=(Paragraph("after the figure.", continued=True),)),
            ]
        )
        self.assertIn(
            "Before the figure\n% Source page 2\nafter the figure.\n\n"
            "\\begin{center}\n% Scanwich has no image file",
            document,
        )
        self.assertIn("\\end{center}\n\n\\begin{longtable}", document)

    def test_does_not_continue_across_other_blocks(self) -> None:
        for block in (Heading("H"), Quote("Q"), Paragraph("C", align="center")):
            with self.subTest(block=type(block).__name__):
                document = render_flow(
                    [
                        PageLayout(blocks=(Paragraph("Before"), block)),
                        PageLayout(blocks=(Paragraph("after.", continued=True),)),
                    ]
                )
                self.assertTrue(
                    document_body(document).endswith("\n\n% Source page 2\n\n\\noindent after.")
                )

    def test_first_page_can_start_in_the_middle_of_a_paragraph(self) -> None:
        self.assertEqual(
            blocks_of(body(Paragraph("the end of a sentence.", continued=True))),
            ["\\noindent the end of a sentence."],
        )


class FootnoteTests(TestCase):
    def test_writes_a_footnote_at_its_reference_mark(self) -> None:
        text = body(
            Heading("A heading<fn>a</fn>"),
            Paragraph("Text<fn>1</fn> and more<fn>*</fn>."),
            Footnote("First <i>note</i>.", mark="1"),
            Footnote("Second, 100% & more.", mark="*"),
            Footnote("Note of the heading.", mark="a"),
        )
        self.assertEqual(
            blocks_of(text),
            [
                "\\section*{A heading\\footnote{Note of the heading.}}",
                "Text\\footnote{First \\textit{note}.} and more\\footnote{Second, 100\\% \\& more.}.",
            ],
        )

    def test_keeps_a_reference_mark_that_has_no_footnote(self) -> None:
        self.assertEqual(
            blocks_of(body(Paragraph("Text<fn>3</fn> and<fn>*</fn>."))),
            ["Text\\textsuperscript{3} and\\textsuperscript{{*}}."],
        )

    def test_keeps_a_footnote_that_has_no_reference_mark(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        Paragraph("Text<fn>1</fn>."),
                        Footnote("Linked.", mark="1"),
                        Footnote("Second with the same mark.", mark="1"),
                        Footnote("No reference.", mark="\u2020"),
                        Footnote("No mark."),
                    )
                )
            ]
        )
        self.assertIn("\\newcommand{\\sourcefootnotetext}[2]", document)
        self.assertEqual(
            blocks_of(document_body(document)),
            [
                "Text\\footnote{Linked.}.",
                "\\sourcefootnotetext{1}{Second with the same mark.}",
                "\\sourcefootnotetext{\u2020}{No reference.}",
                "\\sourcefootnotetext{}{No mark.}",
            ],
        )

    def test_footnote_marks_are_those_of_one_page(self) -> None:
        document = render_flow(
            [
                PageLayout(blocks=(Paragraph("One<fn>1</fn>."), Footnote("Page one.", mark="1"))),
                PageLayout(blocks=(Paragraph("Two<fn>1</fn>."), Footnote("Page two.", mark="1"))),
            ]
        )
        self.assertIn("One\\footnote{Page one.}.", document)
        self.assertIn("Two\\footnote{Page two.}.", document)

    def test_table_keeps_the_marks_and_its_footnotes_come_after_it(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        Table(rows=(cells("a<fn>*</fn>", "b<fn>x</fn>"),), caption="C<fn>1</fn>"),
                        Footnote("Note of a cell.", mark="*"),
                        Footnote("Note of the caption.", mark="1"),
                    )
                )
            ]
        )
        table, caption_footnote, cell_footnote = blocks_of(document_body(document))
        # LaTeX loses a footnote that starts in a table, so the table keeps the marks.
        self.assertIn("{\\centering\nC\\textsuperscript{1}\n}}}", table)
        self.assertIn("a\\textsuperscript{{*}} & b\\textsuperscript{x} \\\\", table)
        self.assertNotIn("\\footnote{", table)
        self.assertEqual(caption_footnote, "\\sourcefootnotetext{1}{Note of the caption.}")
        self.assertEqual(cell_footnote, "\\sourcefootnotetext{{*}}{Note of a cell.}")

    def test_caption_of_a_figure_keeps_the_mark(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        Figure(box=Box(0, 0, 500, 500), caption="Fig. 1<fn>1</fn>", image="f.jpg"),
                        Footnote("Note of the caption.", mark="1"),
                    )
                )
            ]
        )
        # LaTeX puts a footnote that starts in a minipage at the end of the minipage.
        self.assertIn("Fig. 1\\textsuperscript{1}\n\\end{minipage}", document)
        self.assertIn("\\end{center}\n\n\\sourcefootnotetext{1}{Note of the caption.}", document)

    def test_document_without_source_footnotes_has_no_command_for_them(self) -> None:
        document = render_flow(
            [PageLayout(blocks=(Paragraph("T<fn>1</fn>"), Footnote("N", mark="1")))]
        )
        self.assertNotIn("sourcefootnotetext", document)


class FigureTests(TestCase):
    def test_includes_the_image_at_its_part_of_the_printed_width(self) -> None:
        figure = Figure(
            box=Box(300, 400, 700, 600), caption="Fig. 1. <i>A caption</i>", image="b-figures/f.jpg"
        )
        document = render_flow([PageLayout(blocks=(figure,), content_box=Box(200, 100, 1000, 900))])
        self.assertIn("\\usepackage{graphicx}\n", document)
        self.assertEqual(
            document_body(document),
            "% Source page 1\n\n"
            "\\begin{center}\n"
            # LaTeX does not divide a minipage, so the caption stays on the page of the
            # illustration.
            "\\begin{minipage}{\\linewidth}\n"
            "\\centering\n"
            "\\includegraphics[width=0.50\\linewidth,height=0.75\\textheight,"
            "keepaspectratio]{b-figures/f.jpg}\n"
            "\\par\\smallskip\n"
            "Fig. 1. \\textit{A caption}\n"
            "\\end{minipage}\n"
            "\\end{center}",
        )

    def test_figure_width_stays_in_the_line(self) -> None:
        for box, content_box, width in (
            (Box(0, 0, 400, 100), None, "0.40"),
            (Box(0, 0, 900, 100), Box(100, 0, 600, 1000), "1.00"),
            (Box(0, 0, 10, 100), None, "0.15"),
        ):
            with self.subTest(width=width):
                document = render_flow(
                    [PageLayout(blocks=(Figure(box=box, image="f.jpg"),), content_box=content_box)]
                )
                self.assertIn(f"\\includegraphics[width={width}\\linewidth,", document)

    def test_figure_without_an_image_file_keeps_its_caption(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        Figure(box=Box(0, 0, 500, 500)),
                        Figure(box=Box(0, 0, 500, 500), caption="A caption"),
                    )
                )
            ]
        )
        self.assertNotIn("graphicx", document)
        self.assertNotIn("includegraphics", document)
        comment = "% Scanwich has no image file for the illustration that is here in the source."
        self.assertEqual(
            document_body(document),
            f"% Source page 1\n\n{comment}\n\n"
            f"\\begin{{center}}\n{comment}\nA caption\n\\end{{center}}",
        )

    def test_rejects_an_image_path_that_latex_cannot_read_safely(self) -> None:
        for path in ("a b.jpg", "a}.jpg", "a%.jpg", "\\input.jpg", "b\u00fcch.jpg", ""):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "figure image"):
                render_flow([PageLayout(blocks=(Figure(box=Box(0, 0, 9, 9), image=path),))])


class TableTests(TestCase):
    def table_lines(self, table: Table) -> list[str]:
        (block,) = blocks_of(body(table))
        return block.split("\n")

    def column(self, alignment: str, width: str, rules: int = 0) -> str:
        """Return the specification of a column that wraps its text."""
        rule_width = {0: "", 1: "-\\arrayrulewidth", 2: "-2\\arrayrulewidth"}[rules]
        return (
            f">{{\\{alignment}\\arraybackslash}}"
            f"p{{\\dimexpr {width}\\linewidth-2\\tabcolsep{rule_width}\\relax}}"
        )

    def test_writes_a_table_with_a_caption_and_a_header_that_latex_repeats(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        Table(
                            rows=(cells("Name", "Value"), cells("[a]", "1,5"), cells("*", "")),
                            caption="<sc>Table I.</sc> A caption<br>in two lines",
                            header_rows=1,
                            columns=("left", "right"),
                        ),
                    )
                )
            ]
        )
        self.assertIn("\\usepackage{array}\n\\usepackage{longtable}\n", document)
        self.assertNotIn("multirow", document)
        (block,) = blocks_of(document_body(document))
        # A column is as wide as its longest text and the padding of a cell.
        widths = [f"{(_text_em(text) + 1.1) / 32:.4f}" for text in ("Name", "Value")]
        spec = self.column("raggedright", widths[0]) + self.column("raggedleft", widths[1])
        header = ["\\hline", "Name & Value \\\\", "\\hline"]
        self.assertEqual(
            block.split("\n"),
            [
                f"\\begin{{longtable}}{{{spec}}}",
                # The caption is a row of the table, so it stays on the page of its rows.
                "\\multicolumn{2}{c}{\\makebox[0pt]{\\parbox[b]{\\linewidth}{\\centering",
                "T\\sourcesmallcaps{ABLE} I. A caption \\\\ in two lines",
                "}}}\\\\[\\medskipamount]",
                *header,
                "\\endfirsthead",
                *header,
                "\\endhead",
                # A bracket or a star at the start of a row is not an argument of `\\`.
                "{[}a{]} & 1,5 \\\\",
                "{*} & \\\\",
                "\\hline",
                "\\end{longtable}",
            ],
        )

    def test_table_without_a_header_can_have_a_caption(self) -> None:
        lines = self.table_lines(Table(rows=(cells("a"), cells("b")), caption="A caption"))
        self.assertEqual(
            lines[1:],
            [
                "\\multicolumn{1}{c}{\\makebox[0pt]{\\parbox[b]{\\linewidth}{\\centering",
                "A caption",
                "}}}\\\\[\\medskipamount]",
                "\\hline",
                "\\endfirsthead",
                "\\hline",
                "\\endhead",
                "a \\\\",
                "b \\\\",
                "\\hline",
                "\\end{longtable}",
            ],
        )

    def test_writes_the_rules_that_the_table_prints(self) -> None:
        rows = (cells("a", "b"), cells("c", "d"), cells("e", "f"))
        width = "0.0656"
        plain = self.column("raggedright", width)
        one_rule = self.column("raggedright", width, rules=1)
        two_rules = self.column("raggedright", width, rules=2)
        cases = {
            "none": (plain + plain, ["a & b \\\\", "\\endhead", "c & d \\\\", "e & f \\\\"]),
            "horizontal": (
                plain + plain,
                [
                    "\\hline",
                    "a & b \\\\",
                    "\\hline",
                    "\\endhead",
                    "c & d \\\\",
                    "e & f \\\\",
                    "\\hline",
                ],
            ),
            "columns": (
                # The width of a column has the width of its rule.
                one_rule + "|" + plain,
                [
                    "\\hline",
                    "a & b \\\\",
                    "\\hline",
                    "\\endhead",
                    "c & d \\\\",
                    "e & f \\\\",
                    "\\hline",
                ],
            ),
            "grid": (
                "|" + two_rules + "|" + one_rule + "|",
                [
                    "\\hline",
                    "a & b \\\\",
                    "\\hline",
                    "\\endhead",
                    "c & d \\\\",
                    "\\hline",
                    "e & f \\\\",
                    "\\hline",
                ],
            ),
        }
        for rules, (spec, lines) in cases.items():
            with self.subTest(rules=rules):
                self.assertEqual(
                    self.table_lines(Table(rows=rows, rules=rules, header_rows=1)),
                    [f"\\begin{{longtable}}{{{spec}}}", *lines, "\\end{longtable}"],
                )

    def test_table_without_header_rows_has_no_repeated_part(self) -> None:
        for header_rows in (0, 2):
            with self.subTest(header_rows=header_rows):
                lines = self.table_lines(
                    Table(rows=(cells("a"), cells("b")), header_rows=header_rows)
                )
                self.assertEqual(
                    lines[1:], ["\\hline", "a \\\\", "b \\\\", "\\hline", "\\end{longtable}"]
                )

    def test_writes_cells_that_span_columns_and_rows(self) -> None:
        document = render_flow(
            [
                PageLayout(
                    blocks=(
                        Table(
                            rows=(
                                (TableCell("tall", rowspan=2), TableCell("wide", colspan=2)),
                                cells("b", "c"),
                                cells("d", "e", "f"),
                                # A short row gets empty cells.
                                cells("g"),
                            ),
                            rules="grid",
                        ),
                    )
                )
            ]
        )
        # A small TeX installation does not have the package, so the document has a
        # definition for that case.
        self.assertIn(
            "\\IfFileExists{multirow.sty}{\\usepackage{multirow}}{}\n"
            "\\providecommand{\\multirow}[3]{#3}\n",
            document,
        )
        (block,) = blocks_of(document_body(document))
        wide = f"\\multicolumn{{2}}{{{self.column('centering', '0.1313', rules=1)}|}}{{wide}}"
        self.assertEqual(
            block.split("\n")[1:],
            [
                "\\hline",
                f"\\multirow{{2}}{{=}}{{tall}} & {wide} \\\\",
                # The rule has a gap where the cell continues.
                "\\cline{2-3}",
                "& b & c \\\\",
                "\\hline",
                "d & e & f \\\\",
                "\\hline",
                "g &  & \\\\",
                "\\hline",
                "\\end{longtable}",
            ],
        )

    def test_makes_a_span_smaller_when_it_does_not_fit(self) -> None:
        column_count, owners = _place_cells(
            (
                (TableCell("a", rowspan=9), TableCell("b"), TableCell("c", rowspan=2)),
                # The second cell cannot use the position of the cell from the row before.
                (TableCell("d", colspan=5),),
                (TableCell("e", colspan=1000),),
            )
        )
        placed = {cell.cell.text: (cell.colspan, cell.rowspan) for cell in owners.values()}
        # The table has three rows, and the writer accepts a span of 40 columns.
        self.assertEqual(placed, {"a": (1, 3), "b": (1, 1), "c": (1, 2), "d": (1, 1), "e": (40, 1)})
        self.assertEqual(column_count, 41)

    def test_line_break_in_a_cell_does_not_end_the_row(self) -> None:
        lines = self.table_lines(Table(rows=(cells("a<br>b", "c"),), rules="none"))
        self.assertEqual(lines[1], "a \\newline{} b & c \\\\")

    def test_table_with_many_columns_gets_columns_as_wide_as_their_text(self) -> None:
        column_count = MAX_WRAPPED_COLUMNS + 1
        lines = self.table_lines(
            Table(
                rows=(
                    (TableCell("tall", rowspan=2), *cells(*map(str, range(1, column_count)))),
                    (),
                ),
                columns=("right", "center"),
                rules="columns",
            )
        )
        self.assertEqual(
            lines[:2],
            [
                "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}",
                "\\begin{longtable}{r|c|" + "l|" * (column_count - 3) + "l}",
            ],
        )
        self.assertTrue(lines[3].startswith("\\multirow{2}{*}{tall} & 1 & 2 &"))
        self.assertEqual(lines[-2:], ["\\end{longtable}", "\\endgroup"])

    def test_estimates_the_width_of_text_from_its_characters(self) -> None:
        # A digit has half an em. Narrow letters, capital letters, and signs differ from it.
        for text, width in (("", 0.0), ("0123", 2.0), ("il. ", 1.44), ("ab", 1.06), ("AB", 1.5)):
            with self.subTest(text=text):
                self.assertAlmostEqual(_text_em(text), width)
        self.assertGreater(_text_em("MW"), _text_em("mw"))
        self.assertGreater(_text_em("mw"), _text_em("no"))
        self.assertAlmostEqual(_text_em("1 \u00d7 2 = 3"), 3 * 0.5 + 4 * 0.36 + 2 * 0.8)

    def test_measures_the_longest_line_and_the_longest_word_of_each_column(self) -> None:
        column_count, owners = _place_cells(
            (
                cells("12 <b>3456</b> 78<br>9", "0"),
                (TableCell("0000 a cell that spans the columns does not count", colspan=2),),
                cells("", "0" * 99),
            )
        )
        line_widths, word_widths = _column_text_widths(column_count, owners)
        # The first column: eight digits and two spaces in its longest line, and four
        # digits in its longest word. The widths have no markup.
        self.assertAlmostEqual(line_widths[0], 8 * 0.5 + 2 * 0.36)
        self.assertAlmostEqual(word_widths[0], 4 * 0.5)
        # Longer text wraps, so the width of a column has a limit.
        self.assertEqual((line_widths[1], word_widths[1]), (15.0, 15.0))
        # A column without text has a small width.
        _, owners = _place_cells((cells("", "."),))
        self.assertEqual(_column_text_widths(2, owners), ([1.0, 1.0], [1.0, 1.0]))

    def test_table_that_fits_gets_columns_as_wide_as_their_text(self) -> None:
        fractions, compact = _column_fractions([1.0, 5.0], [1.0, 2.0])
        self.assertFalse(compact)
        self.assertEqual(fractions, [(1.0 + 1.1) / 32, (5.0 + 1.1) / 32])

    def test_table_gets_compact_padding_before_its_text_wraps(self) -> None:
        # With the usual padding, these five columns are wider than the line.
        line_widths = [3.0, 7.5, 8.5, 3.0, 7.0]
        fractions, compact = _column_fractions(line_widths, [3.0, 4.0, 4.0, 3.0, 4.0])
        self.assertTrue(compact)
        self.assertEqual(fractions, [(width + 0.55) / 32 for width in line_widths])
        self.assertLessEqual(sum(fractions), 1.0)

    def test_wide_table_fills_the_line_and_keeps_room_for_the_longest_words(self) -> None:
        line_widths = [15.0, 15.0, 3.0]
        word_widths = [4.0, 6.0, 3.0]
        fractions, compact = _column_fractions(line_widths, word_widths)
        self.assertTrue(compact)
        self.assertAlmostEqual(sum(fractions), 1.0)
        text_widths = [fraction * 32 - 0.55 for fraction in fractions]
        for text_width, word_width in zip(text_widths, word_widths, strict=True):
            self.assertGreaterEqual(text_width, word_width - 1e-9)
        # A column that has no text to wrap keeps the width of its text.
        self.assertAlmostEqual(text_widths[2], 3.0)

    def test_table_that_cannot_keep_its_words_shares_the_line_by_word_width(self) -> None:
        word_widths = [15.0] * (MAX_WRAPPED_COLUMNS - 1) + [1.0]
        fractions, compact = _column_fractions(word_widths, word_widths)
        self.assertTrue(compact)
        self.assertAlmostEqual(sum(fractions), 1.0)
        # Each column keeps room for text: its width is more than its padding.
        self.assertGreater(min(fractions) * 32, 0.55)
        self.assertEqual(fractions.index(min(fractions)), len(word_widths) - 1)

    def test_leaves_out_a_table_without_cells(self) -> None:
        document = render_flow([PageLayout(blocks=(Table(rows=((),)),))])
        self.assertNotIn("longtable", document)


class SmallCapitalsTests(TestCase):
    """The document does not print a lowercase letter where the source has a capital form."""

    def test_small_capitals_are_capital_letters_at_a_smaller_size(self) -> None:
        cases = {
            # The page prints these letters as small capitals, so the text has capitals.
            "<sc>BOOK THE FIRST</sc>": "\\sourcesmallcaps{BOOK THE FIRST}",
            # A large capital letter is before the tag.
            "B<sc>OOK THE</sc> F<sc>IRST</sc>": (
                "B\\sourcesmallcaps{OOK THE} F\\sourcesmallcaps{IRST}"
            ),
            # A text with lowercase letters uses them for the small capitals. Its capital
            # letters are large capitals.
            "<sc>Book the First</sc>": "B\\sourcesmallcaps{OOK THE} F\\sourcesmallcaps{IRST}",
            "<sc>book</sc>": "\\sourcesmallcaps{BOOK}",
            # Digits and punctuation keep their size.
            "<sc>FROM 12 TO 48, END</sc>": (
                "\\sourcesmallcaps{FROM} 12 \\sourcesmallcaps{TO} 48, \\sourcesmallcaps{END}"
            ),
            "<sc>caf\u00e9 \u00e9t\u00e9</sc>": "\\sourcesmallcaps{CAF\u00c9 \u00c9T\u00c9}",
            "<sc>a & b</sc>": "\\sourcesmallcaps{A} \\& \\sourcesmallcaps{B}",
            # Without the tag, the capital letters stay capital letters.
            "BOOK THE FIRST": "BOOK THE FIRST",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(blocks_of(body(Paragraph(text))), [expected])

    def test_small_capitals_stay_capitals_in_bold_and_italic_text(self) -> None:
        # The default font has no bold and no italic small capitals. A font command for
        # small capitals then prints lowercase letters.
        self.assertEqual(
            blocks_of(
                body(
                    Heading("C<sc>HAPTER</sc> II"),
                    Paragraph("<i><sc>ROME</sc></i> <b><sc>x</sc></b>"),
                )
            ),
            [
                "\\section*{C\\sourcesmallcaps{HAPTER} II}",
                "\\textit{\\sourcesmallcaps{ROME}} \\textbf{\\sourcesmallcaps{X}}",
            ],
        )

    def test_document_defines_the_command_only_when_it_uses_it(self) -> None:
        definition = "\\DeclareRobustCommand{\\sourcesmallcaps}[1]{{%\n"
        for flow in (False, True):
            with self.subTest(flow=flow):
                with_command = render_latex(
                    [PageLayout(blocks=(Paragraph("<sc>A</sc>"),))], flow=flow
                )
                self.assertEqual(with_command.count(definition), 1)
                self.assertLess(
                    with_command.index(definition), with_command.index("\\begin{document}")
                )
                without = render_latex([PageLayout(blocks=(Paragraph("A"),))], flow=flow)
                self.assertNotIn("sourcesmallcaps", without)


def page_bodies(*pages: PageLayout) -> list[list[str]]:
    """Return the blocks of each source page of a document that keeps the pages of the source.

    The first block of a page has its comment, the commands for its running head and its
    page number, and the start of its box.
    """
    text = document_body(render_latex(list(pages)))
    bodies = []
    for part in text.split("\n\n\\end{sourcepage}")[:-1]:
        bodies.append(part.strip("\n").split("\n\n"))
    return bodies


class SourcePageTests(TestCase):
    """The document that has one page for each page of the source."""

    def test_each_source_page_is_in_a_box_of_its_own(self) -> None:
        document = render_latex(
            [
                PageLayout(blocks=(Heading("A chapter"), Paragraph("First."))),
                PageLayout(),
                PageLayout(blocks=(Paragraph("Third."),)),
            ]
        )
        self.assertEqual(
            document_body(document),
            "% Source page 1\n\\begin{sourcepage}{1}\n\n\\section*{A chapter}\n\nFirst.\n\n"
            "\\end{sourcepage}\n\n"
            # A source page without content is an empty page of the document.
            "% Source page 2\n\\begin{sourcepage}{2}\n\n\\end{sourcepage}\n\n"
            "% Source page 3\n\\begin{sourcepage}{3}\n\nThird.\n\n\\end{sourcepage}",
        )
        for part in (
            "\\usepackage{graphicx}\n",
            "\\newenvironment{sourcepage}[1]{%\n",
            "\\pagestyle{sourcepage}\n",
            # A box that is too high becomes smaller, and the engine reports the page.
            "\\PackageWarningNoLine{scanwich}{Source page \\sourcepagenumber\\space is at\n",
            # The smaller box is 1pt less high than the text area. A box of the full
            # height can be too high by a rounding error.
            "\\resizebox*{!}{\\dimexpr\\textheight-1pt\\relax}{\\usebox{\\sourcepagebox}}",
        ):
            with self.subTest(part=part):
                self.assertEqual(document.count(part), 1)
        self.assertLess(
            document.index("\\pagestyle{sourcepage}"), document.index("\\begin{document}")
        )
        # The document with flowing text has none of this.
        self.assertNotIn(
            "sourcepage", render_latex([PageLayout(blocks=(Paragraph("A"),))], flow=True)
        )

    def test_document_without_pages_has_one_empty_page(self) -> None:
        self.assertTrue(document_body(render_latex([])).endswith("\\mbox{}"))
        self.assertNotIn("\\mbox{}", render_latex([PageLayout()]))

    def test_prints_the_running_head_and_the_page_number_at_their_places(self) -> None:
        cases: list[tuple[tuple[object, ...], list[str]]] = [
            (
                (
                    PageFurniture("page_header", "<sc>A RUNNING HEAD</sc>", "top-center"),
                    PageFurniture("page_number", "23", "top-right"),
                    Paragraph("Text."),
                ),
                ["\\sourcepagehead{}{\\sourcesmallcaps{A RUNNING HEAD}}{23}"],
            ),
            (
                (
                    PageFurniture("page_number", "24", "top-left"),
                    PageFurniture("page_header", "The chapter", "top-center"),
                    Paragraph("Text."),
                    PageFurniture("page_footer", "Sheet 3", "bottom-right"),
                    PageFurniture("page_footer", "Printed in 1900", "bottom-right"),
                ),
                [
                    "\\sourcepagehead{24}{The chapter}{}",
                    "\\sourcepagefoot{}{}{Sheet 3\\quad{}Printed in 1900}",
                ],
            ),
            # A block without a place gets the usual place of its type. A page number
            # is at the top right when it comes before the content, else at the bottom.
            (
                (
                    PageFurniture("page_header", "Head"),
                    PageFurniture("page_number", "5"),
                    Paragraph("Text."),
                    PageFurniture("page_footer", "Foot"),
                ),
                ["\\sourcepagehead{}{Head}{5}", "\\sourcepagefoot{}{Foot}{}"],
            ),
            ((Paragraph("Text."), PageFurniture("page_number", "6")), ["\\sourcepagefoot{}{6}{}"]),
            # A page number at the top with no side is at the outer side of its page: an
            # even number is on a left page.
            (
                (PageFurniture("page_number", "24"), Paragraph("Text.")),
                ["\\sourcepagehead{24}{}{}"],
            ),
            (
                (PageFurniture("page_number", "xii"), Paragraph("Text.")),
                ["\\sourcepagehead{}{}{xii}"],
            ),
            (
                (Paragraph("Text."), PageFurniture("page_number", "30", "top")),
                ["\\sourcepagehead{30}{}{}"],
            ),
            # A place with only one part gets the usual other part of its type.
            (
                (
                    PageFurniture("page_header", "Head", "right"),
                    PageFurniture("page_footer", "Foot", "left"),
                    PageFurniture("page_number", "8", "bottom"),
                    PageFurniture("page_number", "9", "center"),
                    Paragraph("Text."),
                ),
                ["\\sourcepagehead{}{9}{Head}", "\\sourcepagefoot{Foot}{8}{}"],
            ),
            # The text cannot add a command or end the line.
            (
                (PageFurniture("page_header", "50 % } \\end{document}<br>next", "top-left"),),
                [
                    "\\sourcepagehead{50 \\% \\} \\textbackslash{}end\\{document\\} \\quad{} next}{}{}"
                ],
            ),
            ((Paragraph("Text."),), []),
        ]
        for number, (blocks, commands) in enumerate(cases, start=1):
            with self.subTest(case=number):
                (page,) = page_bodies(PageLayout(blocks=tuple(blocks)))
                printed = ["23", "24", "5", "6", "24", "xii", "30", "8, 9"][number - 1 : number]
                comment = "% Source page 1" + (
                    f" (printed page number: {printed[0]})" if printed else ""
                )
                self.assertEqual(
                    page[0].split("\n"), [comment, *commands, "\\begin{sourcepage}{1}"]
                )

    def test_paragraph_that_continues_stays_on_its_two_pages(self) -> None:
        pages = page_bodies(
            PageLayout(blocks=(Paragraph("The first part of a sentence"),)),
            PageLayout(blocks=(Paragraph("continues on the next page.", continued=True),)),
        )
        self.assertEqual(pages[0][1:], ["The first part of a sentence"])
        # The second part has no indent, because it starts no paragraph.
        self.assertEqual(pages[1][1:], ["\\noindent continues on the next page."])

    def test_word_that_the_page_boundary_divides_is_complete_on_the_first_page(self) -> None:
        # A search finds the word, and a copy of the text has it in one piece.
        pages = page_bodies(
            PageLayout(blocks=(Paragraph("The end of the page divides a docu-"), Figure(box=BOX))),
            PageLayout(blocks=(Paragraph("ment, and the text continues.", continued=True),)),
        )
        self.assertEqual(
            pages[0][1], "The end of the page divides a document,\n" + MOVED_WORD_COMMENT
        )
        self.assertEqual(pages[1][1:], ["\\noindent and the text continues."])

        # The word can be all of the text of the second page.
        pages = page_bodies(
            PageLayout(blocks=(Paragraph("A docu-"),)),
            PageLayout(blocks=(Paragraph("ment", continued=True), Paragraph("Next."))),
        )
        self.assertEqual(pages[0][1], "A document\n" + MOVED_WORD_COMMENT)
        self.assertEqual(pages[1][1:], ["Next."])

    def test_keeps_the_division_that_it_cannot_join_safely(self) -> None:
        cases = {
            # A hyphen before a capital letter is a part of the word.
            "a capital letter": ("the Franco-", "Prussian war.", "\\noindent Prussian war."),
            # The writer moves only plain text.
            "markup": ("a docu-", "<i>ment</i> here.", "\\noindent \\textit{ment} here."),
            "no hyphen": ("a document", "that continues.", "\\noindent that continues."),
            "a dash": ("a pause \u2014", "then more.", "\\noindent then more."),
        }
        for name, (first, second, second_latex) in cases.items():
            with self.subTest(name=name):
                pages = page_bodies(
                    PageLayout(blocks=(Paragraph(first),)),
                    PageLayout(blocks=(Paragraph(second, continued=True),)),
                )
                self.assertEqual(pages[0][1:], [escape_text(first)])
                self.assertEqual(pages[1][1:], [second_latex])

    def test_footnotes_keep_the_marks_of_the_source(self) -> None:
        document = render_latex(
            [
                PageLayout(
                    blocks=(
                        Paragraph("A claim.<fn>1</fn> More text.<fn>*</fn>"),
                        Table(rows=(cells("a<fn>a</fn>"),)),
                        Footnote("The first note.", mark="1"),
                        Footnote("The starred note.", mark="*"),
                        Footnote("The note of the table.", mark="a"),
                        Footnote("A note without a mark in the text.", mark="7"),
                    )
                ),
                # The marks start again on each page of the source.
                PageLayout(
                    blocks=(Paragraph("Text.<fn>1</fn>"), Footnote("Second page.", mark="1"))
                ),
            ]
        )
        first, second = (page[1:] for page in page_bodies_of(document))
        self.assertEqual(
            first[0],
            "A claim.\\textsuperscript{1}\\sourcefootnotetext{1}{The first note.} More\n"
            "text.\\textsuperscript{{*}}\\sourcefootnotetext{{*}}{The starred note.}",
        )
        self.assertIn("a\\textsuperscript{a} \\\\", first[1])
        self.assertEqual(
            first[2:],
            [
                "\\sourcefootnotetext{a}{The note of the table.}",
                "\\sourcefootnotetext{7}{A note without a mark in the text.}",
            ],
        )
        self.assertEqual(second, ["Text.\\textsuperscript{1}\\sourcefootnotetext{1}{Second page.}"])
        # In the box of a page, LaTeX takes the mark from this command.
        self.assertIn(
            "\\begingroup\\renewcommand{\\thempfn}{#1}\\footnotetext{#2}\\endgroup}", document
        )
        self.assertNotIn("\\footnote{", document)

    def test_table_is_in_the_box_of_its_page(self) -> None:
        document = render_latex(
            [
                PageLayout(
                    blocks=(
                        Table(
                            rows=(cells("Name", "Value"), cells("a", "1,5")),
                            caption="T<sc>ABLE</sc> I<br>A caption",
                            header_rows=1,
                            columns=("left", "right"),
                            rules="columns",
                            double_rule=True,
                        ),
                    )
                )
            ]
        )
        ((_, block),) = page_bodies_of(document)
        lines = block.split("\n")
        self.assertEqual(
            [lines[0], lines[1], lines[2], lines[3][:17], *lines[4:]],
            [
                "\\begin{center}",
                "T\\sourcesmallcaps{ABLE} I \\\\ A caption",
                "\\par\\medskip",
                "\\begin{tabular}{>",
                # The rule above the table and the rule below it are double rules.
                "\\hline\\hline",
                "Name & Value \\\\",
                "\\hline",
                "a & 1,5 \\\\",
                "\\hline\\hline",
                "\\end{tabular}",
                "\\end{center}",
            ],
        )
        # A column is as wide as its text and its padding, on a line of 38 em.
        self.assertIn(f"p{{\\dimexpr {(_text_em('Name') + 1.1) / 38:.4f}\\linewidth", lines[3])
        self.assertIn("\\usepackage{array}\n", document)
        self.assertNotIn("longtable", document)

    def test_double_rule_is_also_in_the_document_with_flowing_text(self) -> None:
        table = Table(rows=(cells("a"), cells("b")), header_rows=1, double_rule=True)
        lines = blocks_of(body(table))[0].split("\n")
        self.assertEqual(
            lines[1:],
            [
                "\\hline\\hline",
                "a \\\\",
                "\\hline",
                "\\endhead",
                "b \\\\",
                "\\hline\\hline",
                "\\end{longtable}",
            ],
        )
        # A table without rules has no double rule.
        no_rules = Table(rows=(cells("a"),), rules="none", double_rule=True)
        self.assertNotIn("hline", body(no_rules))

    def test_illustration_and_its_caption_need_no_box_of_their_own(self) -> None:
        # A box in the box of the page would take the footnotes of the page.
        ((_, block),) = page_bodies(
            PageLayout(blocks=(Figure(box=BOX, caption="Fig. 1<br>A wheel", image="f.jpg"),))
        )
        self.assertEqual(
            block.split("\n"),
            [
                "\\begin{center}",
                "\\includegraphics[width=0.50\\linewidth,height=0.75\\textheight,keepaspectratio]{f.jpg}",
                "\\par\\smallskip",
                "Fig. 1 \\\\ A wheel",
                "\\end{center}",
            ],
        )


def page_bodies_of(document: str) -> list[list[str]]:
    """Return the blocks of each source page of a document."""
    parts = document_body(document).split("\n\n\\end{sourcepage}")[:-1]
    return [part.strip("\n").split("\n\n") for part in parts]


class PreambleTests(TestCase):
    def test_document_of_text_needs_no_package_for_other_content(self) -> None:
        document = render_flow([PageLayout(blocks=(Paragraph("Text."),))])
        preamble = document[: document.index("\\begin{document}")]
        self.assertIn("\\documentclass[11pt,a4paper]{article}\n", preamble)
        self.assertIn("\\usepackage{fontspec}\n", preamble)
        self.assertIn("\\usepackage[utf8]{inputenc}\n", preamble)
        self.assertEqual(
            [line for line in preamble.split("\n") if line.startswith("\\usepackage")],
            ["\\usepackage{iftex}"],
        )


class EmptyDocumentTests(TestCase):
    def test_document_without_content_has_one_empty_page(self) -> None:
        # LaTeX makes no PDF for a document that has no page.
        no_content = "% The source has no content. The empty box makes one empty page.\n\\mbox{}"
        for pages in ([], [PageLayout(), PageLayout()], [PageLayout(blocks=(Figure(box=BOX),))]):
            with self.subTest(pages=len(pages)):
                self.assertTrue(document_body(render_flow(pages)).endswith(no_content))

    def test_document_with_content_has_no_empty_page(self) -> None:
        for block in (Paragraph("Text."), Footnote("A footnote."), Figure(box=BOX, caption="C")):
            with self.subTest(block=type(block).__name__):
                self.assertNotIn(
                    "\\mbox{}", render_flow([PageLayout(), PageLayout(blocks=(block,))])
                )


BOX = Box(0, 0, 500, 500)
# Texts with two line breaks and no text that LaTeX prints between them. LaTeX stops at the
# second line break of such a text in a centered block.
LINE_BREAK_TEXTS = {
    "nothing": "first line<br><br>second line",
    "space": "first line<br> <br>second line",
    "line feed": "first line<br>\n<br>second line",
    "bold space": "first line<br><b> </b><br>second line",
    "thin space": "first line<br>\u2009<br>second line",
    "em space": "first line<br>\u2003<br>second line",
    "no-break space": "first line<br>\u00a0<br>second line",
    "zero-width space": "first line<br>\u200b<br>second line",
    "soft hyphen": "first line<br>\u00ad<br>second line",
    "three breaks": "<br> first line <br> <br> <br> second line <br> ",
}
# One or more line breaks of LaTeX with nothing between them that LaTeX prints.
REPEATED_LINE_BREAK = re.compile(
    r"(\\\\|\\newline\{\})(\s|~|\\,|\\-|\\quad\{\}|\\enspace\{\})*(\\\\(?!\[)|\\newline\{\})"
)


def blocks_with_text(text: str, image: str | None) -> tuple[object, ...]:
    """Return one block of each type and alignment that has the text."""
    return (
        Paragraph(text),
        Paragraph(text, align="center"),
        Paragraph(text, align="right"),
        Heading(text),
        Heading(text, level=2, align="center"),
        Heading(text, level=3, align="right"),
        Quote(text),
        ListBlock(items=(ListItem(text, marker=text), ListItem(text))),
        ListBlock(items=(ListItem(text),), ordered=True),
        Figure(box=BOX, caption=text, image=image),
        Figure(box=BOX, caption=text),
        Table(rows=(cells(text, "b"), cells("c", text)), caption=text, header_rows=1),
        Table(rows=((TableCell(text, colspan=2),), cells(text, text)), rules="grid"),
        Paragraph("A reference<fn>1</fn>."),
        Footnote(text, mark="1"),
        Footnote(text, mark="\u2020"),
    )


class LineBreakTests(TestCase):
    def test_writes_one_line_break_between_two_texts(self) -> None:
        for name, text in LINE_BREAK_TEXTS.items():
            with self.subTest(text=name):
                document = document_body(
                    render_flow([PageLayout(blocks=blocks_with_text(text, "f.jpg"))])
                )
                self.assertNotRegex(document, REPEATED_LINE_BREAK)
                # Each text of a block has its two lines, with one line break between them.
                # A table with a caption has its header row two times.
                lines = re.findall(
                    r"first line(?:\\,|\\-|\\quad\{\}|~| )* (\\\\|\\newline\{\}) second line",
                    document,
                )
                self.assertEqual(len(lines), 22, document)
                self.assertEqual(lines.count("\\newline{}"), 6)

    def test_line_break_needs_text_before_it_and_after_it(self) -> None:
        for text in ("<br>", " <br> ", "<br>text", "text<br>", "\u2009<br>text<br>\u00a0"):
            with self.subTest(text=text):
                document = document_body(
                    render_flow([PageLayout(blocks=blocks_with_text(text, "f.jpg"))])
                )
                self.assertNotIn("\\\\ ", document)
                self.assertNotIn("\\newline", document)


def difficult_pages(image: str) -> list[PageLayout]:
    """Return pages that use each block type and the characters that are hard for LaTeX."""
    hard_text = (
        "50% of $5 & #3 {a} [b] *c* x_y ^z ~t \\\\ `tick` \"q\" -- --- ,, << >> '' "
        "a\u00a0b so\u00adft \u2011 thin\u202fsp \ufb01n 58\u2032 \u2264 \u03b1 m\u2074 \u2153"
    )
    return [
        PageLayout(
            content_box=Box(100, 80, 900, 950),
            blocks=(
                PageFurniture("page_header", "Head % with } special \\end{document}"),
                PageFurniture("page_number", "406"),
                Heading("[Chapter] <sc>One</sc><fn>1</fn><br>second line", align="center"),
                Heading("*Section*", level=2),
                Heading("Subsection", level=3),
                Paragraph(
                    f"<br>Start. {hard_text} <b>bold <i>both</i></b> <u>u</u> x<sup>2</sup>.<fn>*</fn><br>"
                ),
                Paragraph("[bracket] and *star", align="center"),
                Paragraph("Right<br>two lines", align="right"),
                Quote("\u00ab A quotation \u00bb \u2014 \u201cquotes\u201d \u2026"),
                ListBlock(
                    items=(
                        ListItem("first [x]", marker="a)"),
                        ListItem("second", marker="[b]"),
                        ListItem("[no marker]"),
                        ListItem("", marker="*"),
                    )
                ),
                ListBlock(items=(ListItem("one"), ListItem("[two]")), ordered=True),
                Table(
                    rows=(
                        cells("Part", "Mass", "Price", "Note"),
                        cells("Bolt", "4,75 kg", "3 \u00d7 8 = 24", "58\u2032"),
                        cells("[2]", "*", "\u00bb", "a<br>b<fn>2</fn>"),
                        (
                            TableCell("wide", colspan=2),
                            TableCell("tall", rowspan=2),
                            TableCell("x"),
                        ),
                        cells("p", "q", "r"),
                    ),
                    caption="<sc>Table I.</sc> A caption<br>second line",
                    header_rows=1,
                    columns=("left", "right", "center", "right"),
                    rules="columns",
                ),
                Table(
                    rows=(
                        (TableCell("A", rowspan=2), TableCell("B", colspan=2)),
                        cells("b1", "b2"),
                        cells("1", "2", "3"),
                        (TableCell("4"), TableCell("tall", rowspan=3), TableCell("6")),
                        cells("7", "9"),
                        cells("10"),
                    ),
                    header_rows=2,
                    rules="grid",
                ),
                Table(rows=(cells("only", "row"),), rules="none"),
                Table(rows=(cells(*map(str, range(15))), cells(*["wide"] * 15))),
                Table(rows=(cells("Long text column " * 6, "n", "A second long column " * 3),)),
                Figure(
                    box=Box(200, 500, 700, 800),
                    caption="Fig. 1. <i>A figure</i> & more",
                    image=image,
                ),
                Figure(box=Box(0, 0, 10, 10)),
                Paragraph("The last paragraph ends with a divided wo-"),
                Footnote("Footnote of the heading.", mark="1"),
                Footnote("Footnote with <i>italic</i> and 100% & more.<br>Second line.", mark="*"),
                Footnote("Footnote of the table.", mark="2"),
                Footnote("A footnote without a reference.", mark="\u2020"),
                Footnote("A footnote without a mark."),
            ),
        ),
        PageLayout(
            blocks=(
                Paragraph("rd that continues.<fn>1</fn>", continued=True),
                Footnote("Footnote on page two.", mark="1"),
            )
        ),
        PageLayout(),
    ]


# The parts of the generated texts: the inline markup, and the characters that are special
# for LaTeX or that the writer changes.
TEXT_PARTS = (
    "<br>",
    "<br>",
    " ",
    " ",
    "word",
    "x",
    "<b>",
    "</b>",
    "<i>",
    "</i>",
    "<u>",
    "</u>",
    "<sc>",
    "</sc>",
    "<sup>",
    "</sup>",
    "<sub>",
    "</sub>",
    "<fn>1</fn>",
    "<fn>*</fn>",
    "\\",
    "{",
    "}",
    "$",
    "&",
    "#",
    "%",
    "_",
    "^",
    "~",
    "`",
    '"',
    "'",
    "[",
    "]",
    "*",
    "-",
    ",",
    "<",
    ">",
    "|",
    "=",
    "\u00a0",
    "\u00ad",
    "\u2009",
    "\u2003",
    "\u200b",
    "\u2011",
    "\u2032",
    "\u2260",
    "\u03b1",
    "\u2074",
    "\u2153",
    "\u00e9",
    "e\u0301",
    "\n",
    "\t",
)
GENERATED_TEXT_COUNT = 40
GENERATED_TEXT_PARTS = 12


def generated_texts() -> list[str]:
    """Return texts that put the parts of `TEXT_PARTS` in sequences that no test names."""
    # The fixed seed gives the same texts in each run.
    choice = random.Random(20261013).choice
    return [
        "".join(choice(TEXT_PARTS) for _ in range(GENERATED_TEXT_PARTS))
        for _ in range(GENERATED_TEXT_COUNT)
    ]


class CompilationTests(TestCase):
    """Compile documents with the TeX engines that are installed.

    The other tests compare the LaTeX text. Only an engine can show that the text compiles.
    The Unicode engines need the font Latin Modern Math for the test of the PDF text.
    """

    def compile(self, engine: str, options: list[str], document: str) -> str:
        """Compile a document, and return the text of its PDF with single spaces."""
        texts, _ = self.compile_pages(engine, options, document)
        return " ".join(" ".join(texts).split())

    def compile_pages(
        self, engine: str, options: list[str], document: str
    ) -> tuple[list[str], str]:
        """Compile a document. Return the text of each page of its PDF, and the engine output.

        The text of a page has single spaces. The engine output has the terminal text and
        the log file, with the line breaks of the engine removed.
        """
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "figures").mkdir()
            Image.new("RGB", (300, 200), "gray").save(directory / FIGURE_IMAGE)
            (directory / "document.tex").write_text(document, encoding="utf-8")
            result = subprocess.run(
                [engine, *options, "document.tex"],
                cwd=directory,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=COMPILE_TIMEOUT_SECONDS,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            pdf_path = directory / "document.pdf"
            self.assertTrue(pdf_path.is_file(), f"{engine} wrote no PDF")
            pdf = pdfium.PdfDocument(pdf_path)
            try:
                texts = []
                for index in range(len(pdf)):
                    page = pdf[index]
                    text_page = page.get_textpage()
                    texts.append(" ".join(text_page.get_text_range().split()))
                    text_page.close()
                    page.close()
            finally:
                pdf.close()
            log_path = directory / "document.log"
            log = log_path.read_text(errors="replace") if log_path.is_file() else ""
        # An engine breaks the lines of its messages at a fixed width.
        return texts, (result.stdout + result.stderr + log).replace("\n", "")

    def documents(self, *, unicode_text: bool) -> dict[str, str]:
        """Return the documents that each engine must compile."""
        pages = difficult_pages(FIGURE_IMAGE)
        if unicode_text:
            # pdfLaTeX does not know these characters. The other engines print them or
            # report them, and continue.
            pages.append(
                PageLayout(
                    blocks=(
                        Paragraph("\u03bb\u03cc\u03b3\u03bf\u03c2 \u0416 \u261e \u4e2d \u017f"),
                    )
                )
            )
        line_breaks = [
            PageLayout(blocks=blocks_with_text(text, FIGURE_IMAGE))
            for text in LINE_BREAK_TEXTS.values()
        ]
        generated = [
            PageLayout(blocks=blocks_with_text(text, FIGURE_IMAGE)) for text in generated_texts()
        ]
        documents = {
            "each block type": pages,
            "line breaks": line_breaks,
            "generated texts": generated,
            "no pages": [],
            "empty pages": [PageLayout(), PageLayout()],
            "figure without an image file": [PageLayout(blocks=(Figure(box=BOX),))],
        }
        # Each document compiles with the pages of the source and with flowing text.
        return {
            f"{name}, {layout}": render_latex(document_pages, flow=layout == "flowing text")
            for name, document_pages in documents.items()
            for layout in ("source pages", "flowing text")
        }

    def unicode_engines(self) -> list[str]:
        engines = [engine for engine in UNICODE_ENGINES if shutil.which(engine)]
        if not engines:
            self.skipTest("no LuaLaTeX, XeLaTeX, or Tectonic is installed")
        return engines

    def test_documents_compile_with_the_unicode_engines(self) -> None:
        for engine in self.unicode_engines():
            for name, document in self.documents(unicode_text=True).items():
                with self.subTest(engine=engine, document=name):
                    self.compile(engine, UNICODE_ENGINES[engine], document)

    @skipUnless(shutil.which("pdflatex"), "pdfLaTeX is not installed")
    def test_documents_compile_with_pdflatex(self) -> None:
        for name, document in self.documents(unicode_text=False).items():
            with self.subTest(document=name):
                self.compile("pdflatex", PDFLATEX_OPTIONS, document)

    def all_engines(self) -> dict[str, list[str]]:
        engines = {engine: UNICODE_ENGINES[engine] for engine in self.unicode_engines()}
        if shutil.which("pdflatex"):
            engines["pdflatex"] = PDFLATEX_OPTIONS
        return engines

    def test_each_source_page_is_one_page_of_the_pdf(self) -> None:
        filler = "The invented text of this page has no meaning and it only fills the lines. "
        dense_table = Table(
            caption="T<sc>ABLE</sc> 4. V<sc>ALUES FOR EACH STEP</sc>",
            header_rows=1,
            columns=("center", "center", "center"),
            rules="columns",
            double_rule=True,
            rows=(
                cells("Step", "Value", "Note<fn>a</fn>"),
                *(
                    cells(str(step), f"0,{step * 137 % 1000:03d}", "\u00bb")
                    for step in range(1, 61)
                ),
            ),
        )
        pages = [
            PageLayout(
                blocks=(
                    PageFurniture("page_header", "<sc>AN INVENTED BOOK</sc>", "top-center"),
                    PageFurniture("page_number", "12", "top-right"),
                    Heading("C<sc>HAPTER</sc> II", align="center"),
                    Paragraph("firstmarker " + filler * 6 + "A mark.<fn>1</fn> " + filler * 5),
                    Paragraph(filler * 3 + "The page ends in a docu-"),
                    Footnote("The footnote of the first page.", mark="1"),
                )
            ),
            # A dense page: a long table, an illustration with a caption, footnotes, and
            # more text than one page holds at the usual size.
            PageLayout(
                blocks=(
                    PageFurniture("page_number", "13", "top-left"),
                    Paragraph(
                        "mentation that continues. densemarker " + filler * 8, continued=True
                    ),
                    dense_table,
                    Footnote("The footnote of the table.", mark="a"),
                    Figure(box=BOX, caption="Fig. 7<br>An invented drawing", image=FIGURE_IMAGE),
                    Paragraph(filler * 12 + "Second mark.<fn>2</fn>"),
                    Footnote("The second footnote.", mark="2"),
                    PageFurniture("page_footer", "densefoot", "bottom-center"),
                )
            ),
            PageLayout(blocks=(Paragraph("sparsemarker"), PageFurniture("page_number", "14"))),
            PageLayout(),
            PageLayout(blocks=(Paragraph("lastmarker"),)),
        ]
        document = render_latex(pages)
        for engine, options in self.all_engines().items():
            with self.subTest(engine=engine):
                texts, output = self.compile_pages(engine, options, document)
                # Five pages of the source are five pages of the PDF, in sequence.
                self.assertEqual(len(texts), len(pages), output[-2000:])
                for index, marker in enumerate(
                    ("firstmarker", "densemarker", "sparsemarker", None, "lastmarker")
                ):
                    for other_index, text in enumerate(texts):
                        if marker is not None:
                            self.assertEqual(marker in text, other_index == index, (marker, text))
                self.assertEqual(texts[3], "")
                # The running head and the printed page number are on the page.
                self.assertTrue(texts[0].startswith("AN INVENTED BOOK 12 CHAPTER II"), texts[0])
                self.assertTrue(texts[1].startswith("13 "), texts[1][:40])
                self.assertTrue(texts[1].endswith("densefoot"), texts[1][-80:])
                self.assertTrue(texts[2].endswith("14"), texts[2])
                # The divided word is complete on the first page.
                self.assertIn("The page ends in a documentation", texts[0])
                self.assertIn("that continues. densemarker", texts[1])
                # Each footnote is on the page of its mark, after the text of the page.
                self.assertTrue(
                    texts[0].endswith("1The footnote of the first page."), texts[0][-80:]
                )
                self.assertLess(
                    texts[1].index("60 0,220"), texts[1].index("The footnote of the table.")
                )
                self.assertLess(
                    texts[1].index("An invented drawing"),
                    texts[1].index("The footnote of the table."),
                )
                self.assertIn("2The second footnote.", texts[1])
                # The engine reports the page that is smaller, and no other page.
                reports = re.findall(
                    r"Source page (\d+) is at\s+(\d+)\s+percent of its size", output
                )
                self.assertEqual({number for number, _ in reports}, {"2"}, output[-2000:])
                self.assertTrue(all(30 <= int(percent) < 100 for _, percent in reports), reports)
                self.assertNotIn("Overfull \\vbox", output)

    def test_dense_page_gets_no_empty_page_before_it(self) -> None:
        # The box of a dense page becomes smaller. At the full height of the text area, it
        # can be too high by a rounding error, and the engine then puts an empty page
        # before it. This occurs for a page that ends with a block that has no depth: an
        # illustration with no caption, and with pdfLaTeX a line that has no descender.
        usual = PageLayout(blocks=(Paragraph("A usual page."),))
        endings = {
            "an illustration with no caption": (
                (Figure(box=BOX, image=FIGURE_IMAGE),),
                (60, 68, 108),
            ),
            "a line with no descender": (
                (Paragraph("THE END OF THE CHAPTER 1234", align="center"),),
                (60, 76),
            ),
        }
        for ending, (blocks, row_counts) in endings.items():
            for row_count in row_counts:
                rows = tuple(cells(f"r{row}", "value") for row in range(row_count))
                dense = PageLayout(blocks=(Paragraph("Dense page."), Table(rows=rows), *blocks))
                document = render_latex([usual, dense, usual])
                for engine, options in self.all_engines().items():
                    with self.subTest(ending=ending, rows=row_count, engine=engine):
                        texts, output = self.compile_pages(engine, options, document)
                        self.assertEqual(len(texts), 3, output[-1500:])
                        self.assertEqual(texts[0], "A usual page.")
                        self.assertTrue(texts[1].startswith("Dense page. r0 value"), texts[1][:60])
                        self.assertEqual(texts[2], "A usual page.")
                        self.assertIn("Source page 2 is at", output)

    def test_small_capitals_are_capital_letters_in_the_pdf(self) -> None:
        # The default font has no bold and no italic small capitals. The text of the PDF
        # must have the capital forms of the source also there.
        pages = [
            PageLayout(
                blocks=(
                    Heading("C<sc>HAPTER</sc> II"),
                    Paragraph(
                        "<sc>Book the First</sc>, <i><sc>ROME</sc></i> and <b><sc>paris</sc></b>."
                    ),
                )
            )
        ]
        for flow in (False, True):
            document = render_latex(pages, flow=flow)
            for engine, options in self.all_engines().items():
                with self.subTest(engine=engine, flow=flow):
                    (text,) = self.compile_pages(engine, options, document)[0]
                    # The document with flowing text has a page number of LaTeX.
                    self.assertEqual(
                        text.removesuffix(" 1"), "CHAPTER II BOOK THE FIRST, ROME and PARIS."
                    )

    def test_pdf_text_has_the_characters_of_the_source(self) -> None:
        # A search in the PDF and a copy from it must give the characters of the source.
        # The default text font does not have these characters.
        text = "Position 58\u2032 45\u2033 a\u2264b\u2260c \u222b\u2211\u220f \u2202\u03b1 end."
        document = render_flow([PageLayout(blocks=(Paragraph(text), Table(rows=(cells(text),))))])
        for engine in self.unicode_engines():
            with self.subTest(engine=engine):
                pdf_text = self.compile(engine, UNICODE_ENGINES[engine], document)
                self.assertEqual(pdf_text.count(text), 2, pdf_text)
