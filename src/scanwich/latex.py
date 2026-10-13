"""Write page layouts as one LaTeX document.

The writer makes all the LaTeX. The text of a layout is data: the writer escapes each
character, so the text cannot add a command. The writer also leaves out what stops
LuaLaTeX, XeLaTeX, and Tectonic, for example a line break that has no line to end.
`tests.test_latex.CompilationTests` compiles text that is hard for LaTeX with the engines
that are installed. pdfLaTeX stops at a character that it does not know.
"""

from __future__ import annotations

import re
import textwrap
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from scanwich.layout import (
    ALIGN_CENTER,
    ALIGN_LEFT,
    ALIGN_RIGHT,
    GRID_MAX,
    PAGE_FOOTER,
    PAGE_HEADER,
    PAGE_NUMBER,
    RULES_COLUMNS,
    RULES_GRID,
    RULES_NONE,
    SAFE_IMAGE_PATH,
    STYLE_BOLD,
    STYLE_ITALIC,
    STYLE_SMALL_CAPS,
    STYLE_SUBSCRIPT,
    STYLE_SUPERSCRIPT,
    STYLE_UNDERLINE,
    Figure,
    Footnote,
    FootnoteReference,
    Heading,
    LineBreak,
    ListBlock,
    PageFurniture,
    PageLayout,
    Paragraph,
    Quote,
    Table,
    TableCell,
    parse_inline,
    plain_text,
)

# The source lines of a paragraph are not longer than this, so a person can edit the file.
SOURCE_LINE_WIDTH = 96
HEADING_COMMANDS = {1: "section", 2: "subsection", 3: "subsubsection"}
# The command of each style. The first command is the outer one.
STYLE_COMMANDS = (
    (STYLE_SUPERSCRIPT, "textsuperscript"),
    (STYLE_SUBSCRIPT, "textsubscript"),
    (STYLE_UNDERLINE, "underline"),
    (STYLE_BOLD, "textbf"),
    (STYLE_ITALIC, "textit"),
    (STYLE_SMALL_CAPS, "textsc"),
)
ALIGNMENT_ENVIRONMENTS = {ALIGN_CENTER: "center", ALIGN_RIGHT: "flushright"}
ALIGNMENT_DECLARATIONS = {
    ALIGN_LEFT: "raggedright",
    ALIGN_CENTER: "centering",
    ALIGN_RIGHT: "raggedleft",
}
NATURAL_COLUMNS = {ALIGN_LEFT: "l", ALIGN_CENTER: "c", ALIGN_RIGHT: "r"}

THIN_SPACE = "\\,"
EN_SPACE = "\\enspace{}"
EM_SPACE = "\\quad{}"
# Each character that LaTeX does not print as text, and the LaTeX that prints it.
ESCAPES = {
    "\\": "\\textbackslash{}",
    "{": "\\{",
    "}": "\\}",
    "$": "\\$",
    "&": "\\&",
    "#": "\\#",
    "%": "\\%",
    "_": "\\_",
    "^": "\\textasciicircum{}",
    "~": "\\textasciitilde{}",
    "`": "\\textasciigrave{}",
    # Without this command, some engines print a closing curly quote.
    '"': "\\textquotedbl{}",
    # LaTeX reads these characters as a part of the command before them: an optional
    # argument after `\item`, and a star after `\\`. In a group they are text.
    "[": "{[}",
    "]": "{]}",
    "*": "{*}",
    "\u00a0": "~",
    "\u00ad": "\\-",
    "\u2011": "\\mbox{-}",
    "\u2000": EN_SPACE,
    "\u2002": EN_SPACE,
    "\u2004": EN_SPACE,
    "\u2005": EN_SPACE,
    "\u2007": EN_SPACE,
    "\u2001": EM_SPACE,
    "\u2003": EM_SPACE,
    "\u2006": THIN_SPACE,
    "\u2008": THIN_SPACE,
    "\u2009": THIN_SPACE,
    "\u200a": THIN_SPACE,
    "\u202f": THIN_SPACE,
    # A font can lack the ligature characters, and a search does not find them.
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "\u017ft",
    "\ufb06": "st",
    # A zero-width space and a byte order mark print nothing.
    "\u200b": "",
    "\ufeff": "",
    # The default font of LaTeX does not have the characters below. LaTeX prints nothing
    # for a character that the font does not have, so the writer uses other LaTeX for them.
    "\N{FIGURE DASH}": "\N{EN DASH}",
    "\N{HORIZONTAL BAR}": "\N{EM DASH}",
    "\N{DOUBLE OBLIQUE HYPHEN}": "-",
    "\N{LATIN SMALL LETTER R ROTUNDA}": "r",
    "\N{DITTO MARK}": "\\textquotedbl{}",
    "\N{LOW ASTERISK}": "{*}",
    "\N{TWO ASTERISKS ALIGNED VERTICALLY}": "{*}{*}",
    "\N{ASTERISM}": "{*}\\,{*}\\,{*}",
}


def _symbol(character: str, math_command: str) -> str:
    """Return the LaTeX that prints a character of a math font, see `SYMBOL_COMMAND`."""
    return f"\\sourcesymbol{{{character}}}{{{math_command}}}"


# The characters that only the math fonts of LaTeX have, and their math commands. The
# document prints them with `\sourcesymbol`, see `SYMBOL_COMMAND`.
MATH_SYMBOLS = {
    "\N{PRIME}": "'",
    "\N{DOUBLE PRIME}": "''",
    "\N{TRIPLE PRIME}": "'''",
    "\N{SCRIPT SMALL L}": "\\ell",
    "\N{PARTIAL DIFFERENTIAL}": "\\partial",
    "\N{INCREMENT}": "\\Delta",
    "\N{NABLA}": "\\nabla",
    "\N{ELEMENT OF}": "\\in",
    "\N{NOT AN ELEMENT OF}": "\\notin",
    "\N{N-ARY PRODUCT}": "\\prod",
    "\N{N-ARY SUMMATION}": "\\sum",
    "\N{INTEGRAL}": "\\int",
    "\N{RING OPERATOR}": "\\circ",
    "\N{BULLET OPERATOR}": "\\bullet",
    "\N{DOT OPERATOR}": "\\cdot",
    "\N{PROPORTIONAL TO}": "\\propto",
    "\N{ANGLE}": "\\angle",
    "\N{PARALLEL TO}": "\\parallel",
    "\N{UP TACK}": "\\perp",
    "\N{INTERSECTION}": "\\cap",
    "\N{UNION}": "\\cup",
    "\N{SUBSET OF}": "\\subset",
    "\N{SUPERSET OF}": "\\supset",
    "\N{TILDE OPERATOR}": "\\sim",
    "\N{ASYMPTOTICALLY EQUAL TO}": "\\simeq",
    "\N{APPROXIMATELY EQUAL TO}": "\\cong",
    "\N{ALMOST EQUAL TO}": "\\approx",
    "\N{NOT EQUAL TO}": "\\neq",
    "\N{IDENTICAL TO}": "\\equiv",
    "\N{LESS-THAN OR EQUAL TO}": "\\leq",
    "\N{GREATER-THAN OR EQUAL TO}": "\\geq",
    "\N{MUCH LESS-THAN}": "\\ll",
    "\N{MUCH GREATER-THAN}": "\\gg",
    "\N{LEFT RIGHT ARROW}": "\\leftrightarrow",
    "\N{LEFTWARDS DOUBLE ARROW}": "\\Leftarrow",
    "\N{RIGHTWARDS DOUBLE ARROW}": "\\Rightarrow",
    "\N{LEFT RIGHT DOUBLE ARROW}": "\\Leftrightarrow",
}
ESCAPES.update({symbol: _symbol(symbol, command) for symbol, command in MATH_SYMBOLS.items()})
# The digits and signs that Unicode has as superscript and subscript characters.
SUPERSCRIPT_CHARACTERS = dict(
    zip(
        "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b",
        "0123456789+-",
        strict=True,
    )
)
SUBSCRIPT_CHARACTERS = dict(
    zip(
        "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b",
        "0123456789+-",
        strict=True,
    )
)
ESCAPES.update(
    {symbol: f"\\textsuperscript{{{value}}}" for symbol, value in SUPERSCRIPT_CHARACTERS.items()}
)
ESCAPES.update(
    {symbol: f"\\textsubscript{{{value}}}" for symbol, value in SUBSCRIPT_CHARACTERS.items()}
)
# The fraction characters that the default font does not have.
FRACTION_CHARACTERS = {
    "\N{VULGAR FRACTION ONE THIRD}": (1, 3),
    "\N{VULGAR FRACTION TWO THIRDS}": (2, 3),
    "\N{VULGAR FRACTION ONE FIFTH}": (1, 5),
    "\N{VULGAR FRACTION TWO FIFTHS}": (2, 5),
    "\N{VULGAR FRACTION THREE FIFTHS}": (3, 5),
    "\N{VULGAR FRACTION FOUR FIFTHS}": (4, 5),
    "\N{VULGAR FRACTION ONE SIXTH}": (1, 6),
    "\N{VULGAR FRACTION FIVE SIXTHS}": (5, 6),
    "\N{VULGAR FRACTION ONE EIGHTH}": (1, 8),
    "\N{VULGAR FRACTION THREE EIGHTHS}": (3, 8),
    "\N{VULGAR FRACTION FIVE EIGHTHS}": (5, 8),
    "\N{VULGAR FRACTION SEVEN EIGHTHS}": (7, 8),
}
ESCAPES.update(
    {
        symbol: f"\\textsuperscript{{{numerator}}}/\\textsubscript{{{denominator}}}"
        for symbol, (numerator, denominator) in FRACTION_CHARACTERS.items()
    }
)
# The default font has no small Greek letters. A Greek letter that is alone is mostly a
# symbol, so it gets the letter of a math font. Greek words need a font that has them.
GREEK_SYMBOLS = {
    "\N{GREEK SMALL LETTER ALPHA}": "\\alpha",
    "\N{GREEK SMALL LETTER BETA}": "\\beta",
    "\N{GREEK SMALL LETTER GAMMA}": "\\gamma",
    "\N{GREEK SMALL LETTER DELTA}": "\\delta",
    "\N{GREEK SMALL LETTER EPSILON}": "\\varepsilon",
    "\N{GREEK SMALL LETTER ZETA}": "\\zeta",
    "\N{GREEK SMALL LETTER ETA}": "\\eta",
    "\N{GREEK SMALL LETTER THETA}": "\\theta",
    "\N{GREEK SMALL LETTER IOTA}": "\\iota",
    "\N{GREEK SMALL LETTER KAPPA}": "\\kappa",
    "\N{GREEK SMALL LETTER LAMDA}": "\\lambda",
    "\N{GREEK SMALL LETTER MU}": "\\mu",
    "\N{GREEK SMALL LETTER NU}": "\\nu",
    "\N{GREEK SMALL LETTER XI}": "\\xi",
    "\N{GREEK SMALL LETTER OMICRON}": "o",
    "\N{GREEK SMALL LETTER PI}": "\\pi",
    "\N{GREEK SMALL LETTER RHO}": "\\rho",
    "\N{GREEK SMALL LETTER FINAL SIGMA}": "\\varsigma",
    "\N{GREEK SMALL LETTER SIGMA}": "\\sigma",
    "\N{GREEK SMALL LETTER TAU}": "\\tau",
    "\N{GREEK SMALL LETTER UPSILON}": "\\upsilon",
    "\N{GREEK SMALL LETTER PHI}": "\\varphi",
    "\N{GREEK SMALL LETTER CHI}": "\\chi",
    "\N{GREEK SMALL LETTER PSI}": "\\psi",
    "\N{GREEK SMALL LETTER OMEGA}": "\\omega",
    "\N{GREEK THETA SYMBOL}": "\\vartheta",
    "\N{GREEK PHI SYMBOL}": "\\phi",
    "\N{GREEK PI SYMBOL}": "\\varpi",
}
# The Unicode blocks of Greek letters: Greek and Coptic, and Greek Extended.
GREEK_BLOCKS = (("\u0370", "\u03ff"), ("\u1f00", "\u1fff"))
# Two of these characters in sequence make one different character in a TeX font, for
# example a dash from two hyphens.
LIGATURE_CHARACTERS = frozenset("-,<>'")
# An empty group does not stop a ligature in LuaTeX. A kern stops it in all engines.
LIGATURE_BREAK = "{\\kern0pt}"
# The first letter of the Unicode category of the characters that print: letters, marks,
# numbers, punctuation, and symbols. Spaces and format characters do not print.
PRINTING_CATEGORIES = frozenset("LMNPS")
# Control characters, surrogates, and code points that have no character.
REMOVED_CATEGORIES = frozenset({"Cc", "Cs", "Cn"})
SOURCE_WHITESPACE = re.compile("[ \t\n\r\f\v\x1c-\x1f\x85\u2028\u2029]+")
REPEATED_SPACE = re.compile(" {2,}")
# A word that a page break divides ends in a letter and a hyphen.
DIVIDED_WORD = re.compile(r"\w-$")
DIVIDED_WORD_COMMENT = (
    "% Scanwich joined the word that the page break divides. Compare it with the source."
)

MIN_FIGURE_WIDTH = 0.15
# An illustration is not higher than this part of the text area, so it fits on one page
# with its caption.
MAX_FIGURE_HEIGHT = 0.75
NO_IMAGE_COMMENT = "% Scanwich has no image file for the illustration that is here in the source."
# The estimates that give the width of a table column from its characters. The widths
# are those of the default font, with a small margin: in a column that is too narrow,
# LaTeX wraps the text or divides a word.
NARROW_CHARACTERS = frozenset(" .,:;'!()[]fijlrt\N{RIGHT SINGLE QUOTATION MARK}")
NARROW_CHARACTER_EM = 0.36
WIDE_CHARACTERS = frozenset("mw=+<>%&@\N{MULTIPLICATION SIGN}\N{DIVISION SIGN}\N{MINUS SIGN}")
WIDE_CHARACTER_EM = 0.8
WIDEST_CHARACTERS = frozenset("MW\N{EM DASH}")
WIDEST_CHARACTER_EM = 1.0
CAPITAL_LETTER_EM = 0.75
DIGIT_EM = 0.5
OTHER_CHARACTER_EM = 0.53
# The width of the line of the document class.
LINE_EM = 32.0
# The space that LaTeX puts at the two sides of the text of a cell, and the smaller space
# of a table that does not fit in the line without it.
COLUMN_PADDING_EM = 1.1
COMPACT_COLUMN_PADDING_EM = 0.55
COMPACT_COLUMN_PADDING = "\\setlength{\\tabcolsep}{3pt}"
# The limits for the width of the text of a column. Longer text wraps.
MIN_CELL_EM = 1.0
MAX_CELL_EM = 15.0
CELL_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
# A table with more columns than this gets columns as wide as their text. Its cells
# do not wrap.
MAX_WRAPPED_COLUMNS = 12
MAX_COLUMN_SPAN = 40

CONTEXT_TEXT = "text"
CONTEXT_CELL = "cell"
CONTEXT_CAPTION = "caption"
# LaTeX loses or moves a footnote that starts in a table or in the box of a caption. Thus
# the text of these parts keeps the reference marks of the source.
MARK_CONTEXTS = frozenset({CONTEXT_CELL, CONTEXT_CAPTION})
LINE_BREAKS = {
    CONTEXT_TEXT: "\\\\",
    CONTEXT_CAPTION: "\\\\",
    # In a table cell, `\\` ends the row.
    CONTEXT_CELL: "\\newline{}",
}

PREAMBLE_START = """\
% Scanwich made this file from page images. Compare the text with the source.
% Compile it with LuaLaTeX, XeLaTeX, or Tectonic. These engines read all Unicode text.
% pdfLaTeX stops at a character that it does not know.
\\documentclass[11pt,a4paper]{article}
\\usepackage{iftex}
\\ifPDFTeX
  \\usepackage[T1]{fontenc}
  \\usepackage[utf8]{inputenc}
  \\IfFileExists{lmodern.sty}{\\usepackage{lmodern}}{}
\\else
  \\usepackage{fontspec}
  % The default font has Latin text. For other scripts, set a font that has them:
  % \\setmainfont{...}
\\fi
% The engine reports each character that the font does not have.
\\tracinglostchars=2
"""
TABLE_PACKAGES = """\
\\usepackage{array}
\\usepackage{longtable}
"""
SYMBOL_COMMAND = """\
% The text font does not have some characters, for example a prime mark. With LuaLaTeX,
% XeLaTeX, and Tectonic, such a character comes from the font Latin Modern Math. The text
% of the PDF then has that character, so a search finds it and a copy is correct. A math
% command prints the character without that font and with pdfLaTeX. The text of the PDF
% can then have a different character.
\\ifPDFTeX
  \\newcommand{\\sourcesymbol}[2]{\\ensuremath{#2}}
\\else
  \\IfFontExistsTF{latinmodern-math.otf}{%
    \\newfontfamily\\sourcesymbolfont{latinmodern-math.otf}%
    \\newcommand{\\sourcesymbol}[2]{{\\sourcesymbolfont #1}}%
  }{%
    \\newcommand{\\sourcesymbol}[2]{\\ensuremath{#2}}%
  }
\\fi
"""
# A document without content has this text. LaTeX makes no PDF for a document that has no
# page, and Tectonic then fails.
NO_CONTENT = "% The source has no content. The empty box makes one empty page.\n\\mbox{}"
MULTIROW_PACKAGE = """\
% A small TeX installation does not have the multirow package. Without the package, the
% text of a cell that spans rows is at the top of the cell and not at its center.
\\IfFileExists{multirow.sty}{\\usepackage{multirow}}{}
\\providecommand{\\multirow}[3]{#3}
"""
SOURCE_FOOTNOTE_COMMAND = """\
% A footnote that keeps the reference mark of the source: a footnote of a table, and a
% footnote that has no reference mark in the text of its page.
\\newcommand{\\sourcefootnotetext}[2]{%
  \\begingroup\\renewcommand{\\thefootnote}{#1}\\footnotetext{#2}\\endgroup}
"""


def render_latex(pages: Sequence[PageLayout]) -> str:
    """Return one LaTeX document that has the blocks of all pages in sequence."""
    writer = _DocumentWriter()
    for page_number, page in enumerate(pages, start=1):
        writer.add_page(page_number, page)
    return writer.document()


def escape_text(text: str) -> str:
    """Return LaTeX that prints the text. Each sequence of whitespace becomes one space."""
    text = SOURCE_WHITESPACE.sub(" ", unicodedata.normalize("NFC", text))
    result: list[str] = []
    previous = ""
    for index, character in enumerate(text):
        if character == previous and character in LIGATURE_CHARACTERS:
            result.append(LIGATURE_BREAK)
        escaped = ESCAPES.get(character)
        if (
            character in GREEK_SYMBOLS
            and not _is_greek(previous)
            and not _is_greek(text[index + 1 : index + 2])
        ):
            escaped = _symbol(character, GREEK_SYMBOLS[character])
        previous = character
        if escaped is not None:
            result.append(escaped)
        elif unicodedata.category(character) not in REMOVED_CATEGORIES:
            result.append(character)
    return "".join(result)


def _prints(text: str) -> bool:
    """Return True for a text that has a character that LaTeX prints."""
    return any(unicodedata.category(character)[0] in PRINTING_CATEGORIES for character in text)


def _is_greek(character: str) -> bool:
    return any(first <= character <= last for first, last in GREEK_BLOCKS)


class _PageFootnotes:
    """The footnotes of one page, and the footnotes that wait for the end of a table."""

    def __init__(self, page: PageLayout) -> None:
        self._by_mark: dict[str, Footnote] = {}
        for block in page.blocks:
            if isinstance(block, Footnote) and block.mark:
                self._by_mark.setdefault(block.mark, block)
        self._placed: set[int] = set()
        self.waiting: list[Footnote] = []

    def take(self, mark: str) -> Footnote | None:
        """Return the footnote of a reference mark, one time for each footnote."""
        footnote = self._by_mark.pop(mark, None)
        if footnote is not None:
            self._placed.add(id(footnote))
        return footnote

    def place(self, footnote: Footnote) -> bool:
        """Return False for a footnote that a reference mark took."""
        if id(footnote) in self._placed:
            return False
        if self._by_mark.get(footnote.mark) is footnote:
            del self._by_mark[footnote.mark]
        return True


@dataclass(frozen=True)
class _PlacedCell:
    cell: TableCell
    row: int
    column: int
    colspan: int
    rowspan: int


class _DocumentWriter:
    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._has_figures = False
        self._has_tables = False
        self._has_row_spans = False
        self._has_source_footnotes = False
        # The index of the last usual paragraph, while a paragraph of the next page can
        # continue it. The text of that paragraph without markup comes with it.
        self._open_paragraph: int | None = None
        self._open_paragraph_text = ""

    def document(self) -> str:
        preamble = [PREAMBLE_START]
        if self._has_figures:
            preamble.append("\\usepackage{graphicx}\n")
        if self._has_tables:
            preamble.append(TABLE_PACKAGES)
        if self._has_row_spans:
            preamble.append(MULTIROW_PACKAGE)
        # The headings of the source have their own numbers.
        preamble.append("\\setcounter{secnumdepth}{0}\n")
        if self._has_source_footnotes:
            preamble.append(SOURCE_FOOTNOTE_COMMAND)
        chunks = list(self._chunks)
        if all(line.startswith("%") for chunk in chunks for line in chunk.split("\n")):
            chunks.append(NO_CONTENT)
        body = "\n\n".join(chunks)
        if "\\sourcesymbol{" in body:
            preamble.append(SYMBOL_COMMAND)
        return "".join(preamble) + "\n\\begin{document}\n\n" + body + "\n\n\\end{document}\n"

    def add_page(self, page_number: int, page: PageLayout) -> None:
        footnotes = _PageFootnotes(page)
        comment = _page_comment(page_number, page)
        blocks = [block for block in page.blocks if not isinstance(block, PageFurniture)]
        first_index = next(
            (index for index, block in enumerate(blocks) if not isinstance(block, Footnote)),
            None,
        )
        first = None if first_index is None else blocks[first_index]
        if (
            isinstance(first, Paragraph)
            and first.continued
            and first.align is None
            and self._open_paragraph is not None
        ):
            del blocks[first_index]
            self._continue_paragraph(first, comment, footnotes)
        else:
            self._chunks.append("\n".join(comment))
        for block in blocks:
            self._add_block(block, page, footnotes)

    def _add_block(self, block: object, page: PageLayout, footnotes: _PageFootnotes) -> None:
        if isinstance(block, Paragraph):
            self._add_paragraph(block, footnotes)
        elif isinstance(block, Footnote):
            self._add_footnote_texts([block] if footnotes.place(block) else [])
        elif isinstance(block, Figure):
            self._add_figure(block, page, footnotes)
        elif isinstance(block, Table):
            self._add_table(block, footnotes)
        else:
            # A paragraph of the next page cannot continue across this block.
            self._open_paragraph = None
            if isinstance(block, Heading):
                self._add_heading(block, footnotes)
            elif isinstance(block, Quote):
                text = self._inline(block.text, footnotes, CONTEXT_TEXT)
                self._add_environment("quote", text)
            elif isinstance(block, ListBlock):
                self._add_list(block, footnotes)
        self._add_footnote_texts(footnotes.waiting)
        footnotes.waiting = []

    def _add_paragraph(self, block: Paragraph, footnotes: _PageFootnotes) -> None:
        text = self._inline(block.text, footnotes, CONTEXT_TEXT)
        if block.align is not None:
            self._open_paragraph = None
            self._add_environment(ALIGNMENT_ENVIRONMENTS[block.align], text)
            return
        if not text:
            return
        # The start of this paragraph is on a page that the document does not have.
        start = "\\noindent " if block.continued else ""
        self._open_paragraph = len(self._chunks)
        self._open_paragraph_text = plain_text(block.text)
        self._chunks.append(_wrap(start + text))

    def _continue_paragraph(
        self, block: Paragraph, comment: list[str], footnotes: _PageFootnotes
    ) -> None:
        """Add a paragraph to the open paragraph of the page before."""
        assert self._open_paragraph is not None
        text = _wrap(self._inline(block.text, footnotes, CONTEXT_TEXT))
        before = self._chunks[self._open_paragraph]
        continuation = plain_text(block.text).lstrip()
        if before.endswith("-") and DIVIDED_WORD.search(self._open_paragraph_text.rstrip()):
            # The percent sign at the end of the line joins the two parts without a space.
            if continuation[:1].islower():
                before = before[:-1] + "%"
                comment = [*comment, DIVIDED_WORD_COMMENT]
            else:
                # A hyphen before a capital letter or a digit is a part of the word.
                before += "%"
        self._chunks[self._open_paragraph] = "\n".join([before, *comment, text])
        self._open_paragraph_text = plain_text(block.text)

    def _add_environment(self, environment: str, text: str) -> None:
        if text:
            self._chunks.append(f"\\begin{{{environment}}}\n{_wrap(text)}\n\\end{{{environment}}}")

    def _add_heading(self, block: Heading, footnotes: _PageFootnotes) -> None:
        text = self._inline(block.text, footnotes, CONTEXT_TEXT)
        if not text:
            return
        command = HEADING_COMMANDS[min(max(block.level, 1), len(HEADING_COMMANDS))]
        declaration = f"\\{ALIGNMENT_DECLARATIONS[block.align]} " if block.align is not None else ""
        self._chunks.append(_wrap(f"\\{command}*{{{declaration}{text}}}"))

    def _add_list(self, block: ListBlock, footnotes: _PageFootnotes) -> None:
        if not block.items:
            return
        has_markers = any(item.marker for item in block.items)
        environment = "enumerate" if block.ordered and not has_markers else "itemize"
        lines = [f"\\begin{{{environment}}}"]
        for item in block.items:
            # The braces keep a bracket of the marker out of the optional argument.
            label = f"[{{{self._inline(item.marker, None, CONTEXT_TEXT)}}}]" if item.marker else ""
            text = self._inline(item.text, footnotes, CONTEXT_TEXT)
            lines.append(_wrap(f"\\item{label} {text}".rstrip()))
        lines.append(f"\\end{{{environment}}}")
        self._chunks.append("\n".join(lines))

    def _add_figure(self, block: Figure, page: PageLayout, footnotes: _PageFootnotes) -> None:
        caption = _wrap(self._inline(block.caption or "", footnotes, CONTEXT_CAPTION))
        if block.image is None:
            lines = [NO_IMAGE_COMMENT, *([caption] if caption else [])]
        else:
            if not SAFE_IMAGE_PATH.fullmatch(block.image):
                raise ValueError(
                    "the path of a figure image can have only ASCII letters, digits, "
                    f"and the characters . _ / -: {block.image!r}"
                )
            self._has_figures = True
            page_width = page.content_box.width if page.content_box else GRID_MAX
            width = min(1.0, max(MIN_FIGURE_WIDTH, block.box.width / page_width))
            options = (
                f"width={width:.2f}\\linewidth,height={MAX_FIGURE_HEIGHT}\\textheight,"
                "keepaspectratio"
            )
            lines = [f"\\includegraphics[{options}]{{{block.image}}}"]
            if caption:
                # LaTeX does not divide a minipage, so the illustration and all lines of
                # its caption stay on one page.
                lines = [
                    "\\begin{minipage}{\\linewidth}",
                    "\\centering",
                    *lines,
                    "\\par\\smallskip",
                    caption,
                    "\\end{minipage}",
                ]
        if len(lines) == 1 and block.image is None:
            self._chunks.append(NO_IMAGE_COMMENT)
        else:
            self._chunks.append("\n".join(["\\begin{center}", *lines, "\\end{center}"]))

    def _add_footnote_texts(self, footnotes: Sequence[Footnote]) -> None:
        for footnote in footnotes:
            self._has_source_footnotes = True
            mark = escape_text(footnote.mark)
            text = self._inline(footnote.text, None, CONTEXT_TEXT)
            self._chunks.append(_wrap(f"\\sourcefootnotetext{{{mark}}}{{{text}}}"))

    def _add_table(self, table: Table, footnotes: _PageFootnotes) -> None:
        column_count, owners = _place_cells(table.rows)
        if not column_count:
            return
        self._has_tables = True
        wraps = column_count <= MAX_WRAPPED_COLUMNS
        alignments = [
            table.columns[column] if column < len(table.columns) else ALIGN_LEFT
            for column in range(column_count)
        ]
        fractions: list[float] = []
        compact = False
        if wraps:
            fractions, compact = _column_fractions(*_column_text_widths(column_count, owners))

        def column_spec(column: int, span: int, alignment: str) -> str:
            """Return the column specification of a cell, with the rules at its sides."""
            last = column + span == column_count
            left = "|" if table.rules == RULES_GRID and column == 0 else ""
            right = (
                "|"
                if table.rules == RULES_GRID or (table.rules == RULES_COLUMNS and not last)
                else ""
            )
            if not wraps:
                return left + NATURAL_COLUMNS[alignment] + right
            # The width of the column has its text, its padding, and its rules.
            width = sum(fractions[column : column + span])
            rules = {0: "", 1: "-\\arrayrulewidth", 2: "-2\\arrayrulewidth"}[len(left + right)]
            return (
                f"{left}>{{\\{ALIGNMENT_DECLARATIONS[alignment]}\\arraybackslash}}"
                f"p{{\\dimexpr {width:.4f}\\linewidth-2\\tabcolsep{rules}\\relax}}{right}"
            )

        def row_line(row: int) -> str:
            cells: list[str] = []
            column = 0
            while column < column_count:
                placed = owners.get((row, column))
                if placed is None:
                    cells.append("")
                    column += 1
                    continue
                text = ""
                if placed.row == row:
                    text = self._inline(placed.cell.text, footnotes, CONTEXT_CELL)
                    if placed.rowspan > 1 and text:
                        self._has_row_spans = True
                        # `=` gives the text the width of the column.
                        text = (
                            f"\\multirow{{{placed.rowspan}}}{{{'=' if wraps else '*'}}}{{{text}}}"
                        )
                if placed.colspan > 1:
                    spec = column_spec(column, placed.colspan, ALIGN_CENTER)
                    text = f"\\multicolumn{{{placed.colspan}}}{{{spec}}}{{{text}}}"
                cells.append(text)
                column += placed.colspan
            return " & ".join(cells).strip() + " \\\\"

        def rule_below(row: int) -> str:
            """Return the rule below a row. The rule has a gap where a cell continues."""
            open_columns = [
                column
                for column in range(column_count)
                if owners.get((row, column)) is None
                or owners.get((row, column)) is not owners.get((row + 1, column))
            ]
            if len(open_columns) == column_count:
                return "\\hline"
            # The first column and the last column of each part of the rule.
            parts: list[list[int]] = []
            for column in open_columns:
                if parts and parts[-1][1] == column - 1:
                    parts[-1][1] = column
                else:
                    parts.append([column, column])
            return "".join(f"\\cline{{{first + 1}-{last + 1}}}" for first, last in parts)

        caption = self._inline(table.caption or "", footnotes, CONTEXT_CAPTION)
        has_rules = table.rules != RULES_NONE
        row_count = len(table.rows)
        # LaTeX repeats the header rows when the table continues on a new page.
        header_rows = table.header_rows if table.header_rows < row_count else 0
        head = ["\\hline"] if has_rules else []
        body: list[str] = []
        for row in range(row_count):
            lines = head if row < header_rows else body
            lines.append(row_line(row))
            is_last = row == row_count - 1
            if has_rules and (is_last or row == table.header_rows - 1 or table.rules == RULES_GRID):
                lines.append("\\hline" if is_last else rule_below(row))

        spec = "".join(column_spec(column, 1, alignments[column]) for column in range(column_count))
        lines = [f"\\begin{{longtable}}{{{spec}}}"]
        if caption:
            # The caption is a row of the table, so it stays on the page of the first rows.
            # Only the first page of the table has it. The box of zero width lets the
            # caption be wider than the table.
            caption_start = (
                f"\\multicolumn{{{column_count}}}{{c}}{{\\makebox[0pt]{{"
                "\\parbox[b]{\\linewidth}{\\centering"
            )
            lines += [
                caption_start,
                _wrap(caption),
                "}}}\\\\[\\medskipamount]",
                *head,
                "\\endfirsthead",
            ]
        lines += head
        if caption or header_rows:
            lines.append("\\endhead")
        lines += [*body, "\\end{longtable}"]
        if not wraps:
            # A table with this many columns needs small text to fit in the line.
            lines = [f"\\begingroup\\footnotesize{COMPACT_COLUMN_PADDING}", *lines, "\\endgroup"]
        elif compact:
            lines = [f"\\begingroup{COMPACT_COLUMN_PADDING}", *lines, "\\endgroup"]
        self._chunks.append("\n".join(lines))

    def _inline(self, text: str, footnotes: _PageFootnotes | None, context: str) -> str:
        """Return the LaTeX of a text that uses the inline markup.

        A line break is only between two texts that LaTeX prints. LaTeX stops at a line
        break that has no line to end: at the start of a block, and after a different
        line break in a centered block. Whitespace does not make a line.
        """
        pieces: list[str] = []
        has_printed_text = False
        # A line break waits for the next text that LaTeX prints.
        line_break_waits = False
        for part in parse_inline(text):
            if isinstance(part, LineBreak):
                line_break_waits = has_printed_text
                continue
            if isinstance(part, FootnoteReference):
                piece = self._footnote_reference(part.mark, footnotes, context)
                prints = True
            else:
                piece = escape_text(part.text)
                prints = _prints(part.text)
                if prints:
                    for style, command in reversed(STYLE_COMMANDS):
                        if style in part.styles:
                            piece = f"\\{command}{{{piece}}}"
            if prints:
                if line_break_waits:
                    pieces.append(f" {LINE_BREAKS[context]} ")
                    line_break_waits = False
                has_printed_text = True
            pieces.append(piece)
        return REPEATED_SPACE.sub(" ", "".join(pieces)).strip()

    def _footnote_reference(self, mark: str, footnotes: _PageFootnotes | None, context: str) -> str:
        """Return the LaTeX for a reference mark.

        In text, the footnote of the mark becomes a LaTeX footnote, and LaTeX gives it a
        new number. The pages of the LaTeX document are not those of the source, so the
        marks of the source can occur two times on one page. A table and a caption keep
        the mark, and the footnote comes after the block.
        """
        footnote = footnotes.take(mark) if footnotes is not None else None
        if footnote is not None and footnotes is not None and context in MARK_CONTEXTS:
            footnotes.waiting.append(footnote)
        elif footnote is not None:
            return f"\\footnote{{{self._inline(footnote.text, None, CONTEXT_TEXT)}}}"
        return f"\\textsuperscript{{{escape_text(mark)}}}"


def _wrap(text: str) -> str:
    """Return the text in source lines of a limited length. A line breaks only at a space."""
    return "\n".join(
        textwrap.wrap(
            text,
            width=SOURCE_LINE_WIDTH,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _comment_text(text: str) -> str:
    """Return a text for one comment line: no markup, no line break, no control character."""
    plain = unicodedata.normalize("NFC", plain_text(text))
    visible = "".join(
        " " if unicodedata.category(character) in REMOVED_CATEGORIES else character
        for character in plain
    )
    return " ".join(visible.split())


def _page_comment(page_number: int, page: PageLayout) -> list[str]:
    """Return the comment lines that show where a source page starts."""
    furniture = [block for block in page.blocks if isinstance(block, PageFurniture)]
    printed_numbers = [
        _comment_text(block.text) for block in furniture if block.kind == PAGE_NUMBER
    ]
    line = f"% Source page {page_number}"
    if printed_numbers:
        line += f" (printed page number: {', '.join(printed_numbers)})"
    lines = [line]
    labels = {PAGE_HEADER: "Page header", PAGE_FOOTER: "Page footer"}
    lines.extend(
        f"% {labels[block.kind]}: {_comment_text(block.text)}"
        for block in furniture
        if block.kind in labels
    )
    return lines


def _place_cells(
    rows: Sequence[Sequence[TableCell]],
) -> tuple[int, dict[tuple[int, int], _PlacedCell]]:
    """Return the number of columns, and the cell that has each position of the table.

    A cell with a span has more than one position. A span becomes smaller when it does
    not fit: at the end of the table, or at a position that a different cell has.
    """
    owners: dict[tuple[int, int], _PlacedCell] = {}
    column_count = 0
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row:
            while (row_index, column) in owners:
                column += 1
            colspan = 1
            while (
                colspan < min(cell.colspan, MAX_COLUMN_SPAN)
                and (row_index, column + colspan) not in owners
            ):
                colspan += 1
            rowspan = 1
            while (
                rowspan < cell.rowspan
                and row_index + rowspan < len(rows)
                and all(
                    (row_index + rowspan, column + offset) not in owners
                    for offset in range(colspan)
                )
            ):
                rowspan += 1
            placed = _PlacedCell(cell, row_index, column, colspan, rowspan)
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    owners[(row_index + row_offset, column + column_offset)] = placed
            column += colspan
            column_count = max(column_count, column)
    return column_count, owners


def _text_em(text: str) -> float:
    """Return an estimate of the width of a text in em."""
    width = 0.0
    for character in text:
        if character in NARROW_CHARACTERS:
            width += NARROW_CHARACTER_EM
        elif character in WIDEST_CHARACTERS:
            width += WIDEST_CHARACTER_EM
        elif character in WIDE_CHARACTERS:
            width += WIDE_CHARACTER_EM
        elif character.isdigit():
            width += DIGIT_EM
        elif character.isupper():
            width += CAPITAL_LETTER_EM
        else:
            width += OTHER_CHARACTER_EM
    return width


def _column_text_widths(
    column_count: int, owners: dict[tuple[int, int], _PlacedCell]
) -> tuple[list[float], list[float]]:
    """Return two widths in em for the text of each column: its longest line and longest word."""
    line_widths = [MIN_CELL_EM] * column_count
    word_widths = [MIN_CELL_EM] * column_count
    for placed in set(owners.values()):
        if placed.colspan != 1:
            continue
        for raw_line in CELL_LINE_BREAK.split(placed.cell.text):
            # The markup is not text, so the widths are those of the text without it.
            line = plain_text(raw_line).strip()
            longest_word = max(map(_text_em, line.split()), default=0.0)
            line_widths[placed.column] = max(
                line_widths[placed.column], min(_text_em(line), MAX_CELL_EM)
            )
            word_widths[placed.column] = max(
                word_widths[placed.column], min(longest_word, MAX_CELL_EM)
            )
    return line_widths, word_widths


def _column_fractions(
    line_widths: Sequence[float], word_widths: Sequence[float]
) -> tuple[list[float], bool]:
    """Return the width of each column as a part of the line, and True for compact padding.

    The width of a column has its text and its padding. A table that fits in the line gets
    columns as wide as their longest lines, with compact padding when that makes it fit.
    A wider table fills the line and its text wraps. Each column then keeps room for its
    longest word when that is possible, because LaTeX divides a word that has no room.
    """
    count = len(line_widths)
    for padding in (COLUMN_PADDING_EM, COMPACT_COLUMN_PADDING_EM):
        if sum(line_widths) + count * padding <= LINE_EM:
            fractions = [(width + padding) / LINE_EM for width in line_widths]
            return fractions, padding == COMPACT_COLUMN_PADDING_EM
    available = LINE_EM - count * COMPACT_COLUMN_PADDING_EM
    if sum(word_widths) <= available:
        # The columns share the space that the longest words do not need.
        spare = available - sum(word_widths)
        slack = [line - word for line, word in zip(line_widths, word_widths, strict=True)]
        text_widths = [
            word + spare * column_slack / sum(slack)
            for word, column_slack in zip(word_widths, slack, strict=True)
        ]
    else:
        text_widths = [available * word / sum(word_widths) for word in word_widths]
    return [(width + COMPACT_COLUMN_PADDING_EM) / LINE_EM for width in text_widths], True
