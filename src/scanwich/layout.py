"""Backend-neutral page layout types: the structure of one page in reading order.

The JSON form of these types is the contract between a layout backend and the document
writers. A backend returns it, `--ocr-output-dir` saves it, and `scanwich.latex` reads it.

Text fields use a small inline markup. `parse_inline` reads it:

- `<b>`, `<i>`, `<u>`, `<sc>` (small capitals), `<sup>`, and `<sub>` set a style.
- `<br>` is a line break that the page makes on purpose.
- `<fn>mark</fn>` is the place of a footnote reference mark.

Other text that has `<` or `&` is literal text.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# A box gives its coordinates on a grid of this size for the width and for the height of
# the page image. Thus a box does not change when the page image gets a different resolution.
GRID_MAX = 1000.0
# A box can be this much outside the grid, as a result of rounding. A box that is more
# outside is not a box of the grid. A model that uses the pixels of an image makes such
# a box.
GRID_TOLERANCE = 0.02 * GRID_MAX
# The sides of a box, in the sequence of the array form of a box.
BOX_SIDES = ("left", "top", "right", "bottom")
# The sides of the page image. `PAGE_TOP_UPRIGHT` is the side for a page that is upright.
PAGE_TOP_UPRIGHT = "top"
PAGE_SIDES = (PAGE_TOP_UPRIGHT, "right", "bottom", "left")
MAX_HEADING_LEVEL = 3
ALIGN_CENTER = "center"
ALIGN_RIGHT = "right"
ALIGN_LEFT = "left"
# The alignment names that a layout can have, and the name that the layout types use.
ALIGNMENTS = {
    "center": ALIGN_CENTER,
    "centre": ALIGN_CENTER,
    "centered": ALIGN_CENTER,
    "middle": ALIGN_CENTER,
    "right": ALIGN_RIGHT,
    "end": ALIGN_RIGHT,
    "left": ALIGN_LEFT,
    "start": ALIGN_LEFT,
    # The usual alignment of a paragraph. For a table column, it is the left alignment.
    "justify": ALIGN_LEFT,
    "justified": ALIGN_LEFT,
}
RULES_NONE = "none"
RULES_HORIZONTAL = "horizontal"
RULES_COLUMNS = "columns"
RULES_GRID = "grid"
TABLE_RULES = (RULES_NONE, RULES_HORIZONTAL, RULES_COLUMNS, RULES_GRID)
# The names that a layout can have for the rules of a table.
RULE_NAMES = {
    **{rules: rules for rules in TABLE_RULES},
    "all": RULES_GRID,
    "vertical": RULES_COLUMNS,
}
PAGE_HEADER = "page_header"
PAGE_FOOTER = "page_footer"
PAGE_NUMBER = "page_number"
PAGE_FURNITURE_KINDS = (PAGE_HEADER, PAGE_FOOTER, PAGE_NUMBER)
# The fields that the layout reads from each JSON object. Text in a different field is
# content that the layout cannot keep, so it is an error.
PAGE_FIELDS = ("page_top", "content_box", "blocks")
BLOCK_FIELDS = {
    "heading": ("type", "level", "text", "align"),
    "paragraph": ("type", "text", "align", "continued"),
    "quote": ("type", "text"),
    "list": ("type", "ordered", "items"),
    "table": ("type", "caption", "header_rows", "columns", "rules", "rows", "header"),
    "figure": ("type", "box", "caption", "image"),
    "footnote": ("type", "mark", "text"),
    PAGE_HEADER: ("type", "text"),
    PAGE_FOOTER: ("type", "text"),
    PAGE_NUMBER: ("type", "text"),
}
BLOCK_TYPES = frozenset(BLOCK_FIELDS)
LIST_ITEM_FIELDS = ("marker", "text")
CELL_FIELDS = ("text", "colspan", "rowspan")
FLAG_VALUES = {"true": True, "false": False}
COLUMN_LETTERS = {"l": ALIGN_LEFT, "c": ALIGN_CENTER, "r": ALIGN_RIGHT}
# A JSON escape such as `\f` or `\b` makes one of these control characters. Page text
# does not have them. A model makes them when it writes a backslash without an escape,
# for example in a LaTeX command.
BROKEN_ESCAPE_CHARACTER = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# The path of the image file of a figure can have only these characters, because the
# LaTeX writer does not escape it.
SAFE_IMAGE_PATH = re.compile(r"[A-Za-z0-9._/-]+")
# The hint of an error names this many fields, and this many characters of each name.
MAX_HINT_FIELDS = 8
MAX_HINT_NAME_CHARACTERS = 40
# One half of a character outside the Basic Multilingual Plane. A file in UTF-8 cannot
# have it.
SURROGATE = re.compile("[\ud800-\udfff]")

STYLE_BOLD = "b"
STYLE_ITALIC = "i"
STYLE_UNDERLINE = "u"
STYLE_SMALL_CAPS = "sc"
STYLE_SUPERSCRIPT = "sup"
STYLE_SUBSCRIPT = "sub"
# The tag names of the inline markup, and the style that each one sets.
STYLE_TAGS = {
    "b": STYLE_BOLD,
    "strong": STYLE_BOLD,
    "i": STYLE_ITALIC,
    "em": STYLE_ITALIC,
    "u": STYLE_UNDERLINE,
    "sc": STYLE_SMALL_CAPS,
    "sup": STYLE_SUPERSCRIPT,
    "sub": STYLE_SUBSCRIPT,
}
FOOTNOTE_TAG = "fn"
INLINE_TOKEN = re.compile(
    r"<(?P<close>/?)(?P<name>" + "|".join(sorted(STYLE_TAGS, key=len, reverse=True)) + r")>"
    r"|<fn>(?P<mark>.*?)</fn>"
    r"|(?P<break><br\s*/?>)",
    re.IGNORECASE | re.DOTALL,
)
# A model that writes tags can also write these entities for literal characters.
ENTITY = re.compile(r"&(amp|lt|gt|quot|apos|nbsp|#39);")
ENTITY_CHARACTERS = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "#39": "'",
    "nbsp": "\u00a0",
}

logger = logging.getLogger(__name__)


class LayoutFormatError(ValueError):
    """Layout data does not have the documented shape.

    The message has no text from the data, so a log record can have it. `hint` has more
    help for the author of the data. It can have field names from the data, so it is
    only for that author, for example the model that gets a correction request.
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


@dataclass(frozen=True)
class Box:
    """A rectangle on the 0-to-1000 grid of the page image. (0, 0) is the top-left corner."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def to_json(self) -> dict[str, float]:
        # Names for the sides leave no doubt about their sequence.
        return {"left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom}


@dataclass(frozen=True)
class Heading:
    text: str
    # 1 is a chapter or the main title, 2 is a section, and 3 is a subsection.
    level: int = 1
    align: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _without_defaults(
            {"type": "heading", "level": self.level, "text": self.text, "align": self.align}
        )


@dataclass(frozen=True)
class Paragraph:
    text: str
    # `None` is a usual paragraph. The other values are for a paragraph that the page centers
    # or sets flush right.
    align: str | None = None
    # The paragraph continues the last paragraph of the page before.
    continued: bool = False

    def to_json(self) -> dict[str, Any]:
        return _without_defaults(
            {
                "type": "paragraph",
                "text": self.text,
                "align": self.align,
                "continued": self.continued or None,
            }
        )


@dataclass(frozen=True)
class Quote:
    """A block quotation."""

    text: str

    def to_json(self) -> dict[str, Any]:
        return {"type": "quote", "text": self.text}


@dataclass(frozen=True)
class ListItem:
    text: str
    # The marker that the page prints, for example "a)". `None` is the default marker.
    marker: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _without_defaults({"marker": self.marker, "text": self.text})


@dataclass(frozen=True)
class ListBlock:
    items: tuple[ListItem, ...]
    ordered: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "list",
            "ordered": self.ordered,
            "items": [item.to_json() for item in self.items],
        }


@dataclass(frozen=True)
class TableCell:
    text: str
    colspan: int = 1
    rowspan: int = 1

    def to_json(self) -> str | dict[str, Any]:
        if self.colspan == 1 and self.rowspan == 1:
            return self.text
        result: dict[str, Any] = {"text": self.text}
        if self.colspan != 1:
            result["colspan"] = self.colspan
        if self.rowspan != 1:
            result["rowspan"] = self.rowspan
        return result


@dataclass(frozen=True)
class Table:
    rows: tuple[tuple[TableCell, ...], ...]
    caption: str | None = None
    # The first rows are the header of the table.
    header_rows: int = 0
    # The alignment of each column. A column without a value is left-aligned.
    columns: tuple[str, ...] = ()
    rules: str = RULES_HORIZONTAL

    def to_json(self) -> dict[str, Any]:
        return _without_defaults(
            {
                "type": "table",
                "caption": self.caption,
                "header_rows": self.header_rows,
                "columns": list(self.columns) or None,
                "rules": self.rules,
                "rows": [[cell.to_json() for cell in row] for row in self.rows],
            }
        )


@dataclass(frozen=True)
class Figure:
    """An illustration. The pipeline crops `box` from the page image and sets `image`."""

    box: Box
    caption: str | None = None
    # The path of the cropped image, relative to the document that includes it.
    image: str | None = None

    def to_json(self) -> dict[str, Any]:
        return _without_defaults(
            {
                "type": "figure",
                "box": self.box.to_json(),
                "caption": self.caption,
                "image": self.image,
            }
        )


@dataclass(frozen=True)
class FigureLocation:
    """The place of one illustration in an image that shows a part of a page.

    `box` is on the 0-to-1000 grid of that image. `cut_sides` has the sides of the image
    where the illustration continues outside the image.
    """

    box: Box
    cut_sides: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Footnote:
    text: str
    # The reference mark that the page prints, for example "1" or "*".
    mark: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"type": "footnote", "mark": self.mark, "text": self.text}


@dataclass(frozen=True)
class PageFurniture:
    """Text that belongs to the page and not to the document: a running head or a page number."""

    kind: str
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"type": self.kind, "text": self.text}


Block = Heading | Paragraph | Quote | ListBlock | Table | Figure | Footnote | PageFurniture


@dataclass(frozen=True)
class PageLayout:
    """The blocks of one page in reading order."""

    blocks: tuple[Block, ...] = ()
    # The side of the page image where the top of the page is. A backend that gets a page
    # that is not upright can return this side and no blocks. The pipeline then turns the
    # page image and asks again.
    page_top: str = PAGE_TOP_UPRIGHT
    # The printed area of the page: all its text and illustrations, without the margins.
    content_box: Box | None = None

    def to_json(self) -> dict[str, Any]:
        return _without_defaults(
            {
                "page_top": self.page_top,
                "content_box": self.content_box.to_json() if self.content_box else None,
                "blocks": [block.to_json() for block in self.blocks],
            }
        )


@dataclass(frozen=True)
class Span:
    """Text with one set of styles."""

    text: str
    styles: frozenset[str] = frozenset()


@dataclass(frozen=True)
class LineBreak:
    pass


@dataclass(frozen=True)
class FootnoteReference:
    mark: str


Inline = Span | LineBreak | FootnoteReference


def parse_inline(text: str) -> list[Inline]:
    """Return the parts of a text that uses the inline markup.

    The function accepts all text. A closing tag without its opening tag has no effect, and
    a style that has no closing tag continues to the end of the text.
    """
    parts: list[Inline] = []
    # A style can be open more than one time, for example `<i>a <i>b</i> c</i>`.
    open_styles: dict[str, int] = {}

    def add_text(value: str) -> None:
        value = ENTITY.sub(lambda match: ENTITY_CHARACTERS[match.group(1)], value)
        if not value:
            return
        styles = frozenset(open_styles)
        if parts and isinstance(parts[-1], Span) and parts[-1].styles == styles:
            parts[-1] = Span(parts[-1].text + value, styles)
        else:
            parts.append(Span(value, styles))

    position = 0
    for token in INLINE_TOKEN.finditer(text):
        add_text(text[position : token.start()])
        position = token.end()
        if token.group("break"):
            parts.append(LineBreak())
        elif token.group("mark") is not None:
            mark = plain_text(token.group("mark")).strip()
            if mark:
                parts.append(FootnoteReference(mark))
        else:
            style = STYLE_TAGS[token.group("name").lower()]
            if not token.group("close"):
                open_styles[style] = open_styles.get(style, 0) + 1
            elif style in open_styles:
                open_styles[style] -= 1
                if not open_styles[style]:
                    del open_styles[style]
    add_text(text[position:])
    return parts


def plain_text(text: str) -> str:
    """Return a text without its inline markup. A line break becomes a space."""
    return "".join(
        part.text if isinstance(part, Span) else " "
        for part in parse_inline(text)
        if not isinstance(part, FootnoteReference)
    )


@dataclass
class _Reading:
    """The state of the check of one JSON value."""

    # The value is the reply of a model. The image file of a figure comes from the
    # pipeline, so a reply has no `image` field.
    reply: bool = False
    # Each object that has text in a field that the layout does not read: its place, the
    # fields that the layout reads, and the names of the other fields.
    other_content: list[tuple[str, Sequence[str], tuple[str, ...]]] = field(default_factory=list)

    def note_other_content(
        self, raw_object: Mapping[Any, Any], fields: Sequence[str], where: str
    ) -> None:
        """Record an object that has text in a field that the layout does not read.

        Such text is content of the page that the layout cannot keep. A field without
        text, for example a number or a flag, is a hint that the layout ignores.
        """
        names = tuple(
            str(name)
            for name, value in raw_object.items()
            if name not in fields and _has_text(value)
        )
        if names:
            self.other_content.append((where, fields, names))

    def other_content_hint(self) -> str:
        """Return the text that names the fields, for the author of the value."""
        places: dict[str, list[str]] = {}
        for where, _, names in self.other_content:
            for name in names:
                places.setdefault(name, []).append(where)
        parts = []
        for name, wheres in list(places.items())[:MAX_HINT_FIELDS]:
            quoted = json.dumps(name[:MAX_HINT_NAME_CHARACTERS], ensure_ascii=False)
            if len(wheres) == 1:
                parts.append(f"{quoted} in 1 object ({wheres[0]})")
            else:
                parts.append(f"{quoted} in {len(wheres)} objects (the first one is {wheres[0]})")
        if len(places) > MAX_HINT_FIELDS:
            parts.append(f"and {len(places) - MAX_HINT_FIELDS} more fields")
        return (
            "The layout does not read these fields: " + "; ".join(parts) + ". "
            "This rule is for each object of the reply. Remove these fields from all "
            "objects, or put their text in a field that the layout reads."
        )


def page_layout_from_json(payload: Any, *, reply: bool = False) -> PageLayout:
    """Return the page layout that a JSON value describes.

    Text of the value does not disappear: the function keeps it or raises
    `LayoutFormatError`. A backend can then ask its model for a correct value. The function
    accepts a value that has one clear meaning, for example an integer where the page
    prints digits. It ignores a field that has no text and that the layout does not read.

    `reply` is True for the reply of a model. A figure of a reply has no `image` field.

    An error message has no text from the value, so a log record can have the message.
    The `hint` of the error can have field names of the value.
    """
    reading = _Reading(reply=reply)
    try:
        layout = _parse_page(payload, reading)
    except LayoutFormatError as error:
        if reading.other_content and not error.hint:
            # The author of the value gets all the defects that the check found.
            error.hint = reading.other_content_hint()
        raise
    if reading.other_content:
        where, fields, _ = reading.other_content[0]
        raise LayoutFormatError(
            f"{where} has text in a field that the layout does not read; "
            "its fields are: " + ", ".join(fields),
            hint=reading.other_content_hint(),
        )
    return layout


def figure_location_from_json(payload: Any) -> FigureLocation | None:
    """Check the JSON form of the place of an illustration, and return the place.

    The form is {"box": {"left": L, "top": T, "right": R, "bottom": B}, "cut": ["top"]}.
    A `box` of null gives None: the image shows no illustration. The function raises
    `LayoutFormatError` for a value that it cannot use.
    """
    if not isinstance(payload, Mapping) or "box" not in payload:
        raise LayoutFormatError("the reply must be a JSON object with a box")
    if payload["box"] is None:
        return None
    box = _parse_box(payload["box"], "box", _Reading())
    raw_sides = payload.get("cut")
    if raw_sides is None:
        raw_sides = []
    elif isinstance(raw_sides, str):
        raw_sides = [raw_sides]
    if not isinstance(raw_sides, list):
        raw_sides = [None]
    sides = [side.strip().lower() if isinstance(side, str) else "" for side in raw_sides]
    if not all(side in BOX_SIDES for side in sides):
        raise LayoutFormatError(
            "cut must be an array that has only the sides " + ", ".join(BOX_SIDES)
        )
    return FigureLocation(box=box, cut_sides=frozenset(sides))


def _parse_page(payload: Any, reading: _Reading) -> PageLayout:
    if isinstance(payload, list):
        # A value that is only the array of the blocks.
        payload = {"blocks": payload}
    if not isinstance(payload, Mapping):
        raise LayoutFormatError("the page layout must be a JSON object")
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list):
        raise LayoutFormatError("the page layout must have a blocks array")
    reading.note_other_content(payload, PAGE_FIELDS, "the page layout")
    blocks: list[Block] = []
    for block_index, raw_block in enumerate(raw_blocks, start=1):
        block = _parse_block(raw_block, f"block {block_index}", reading)
        if block is not None:
            blocks.append(block)
    return PageLayout(
        blocks=tuple(blocks),
        page_top=_parse_page_top(payload.get("page_top")),
        content_box=_parse_content_box(payload.get("content_box"), reading),
    )


def _parse_page_top(value: Any) -> str:
    if value is None:
        return PAGE_TOP_UPRIGHT
    side = value.strip().lower() if isinstance(value, str) else None
    if side not in PAGE_SIDES:
        raise LayoutFormatError("page_top must be one of: " + ", ".join(PAGE_SIDES))
    return side


def _parse_content_box(value: Any, reading: _Reading) -> Box | None:
    """Return the box of the printed area, or `None` for numbers that are not a usable box.

    The box is only a hint for the width of an illustration, so numbers that are not
    usable are not a reason to reject the layout of a page. Text is not a hint.
    """
    if value is None:
        return None
    try:
        return _parse_box(value, "content_box", reading)
    except LayoutFormatError as error:
        if _has_text(value):
            raise
        logger.info("Ignoring the content_box of a page layout: %s", error)
        return None


def _parse_block(raw_block: Any, where: str, reading: _Reading) -> Block | None:
    """Return one block, or `None` for a block that has no content."""
    if not isinstance(raw_block, Mapping):
        raise LayoutFormatError(f"{where} must be an object")
    block_type = raw_block.get("type")
    if not isinstance(block_type, str):
        raise LayoutFormatError(f"{where} must have a type")
    if block_type not in BLOCK_FIELDS:
        # An error message and a log record have no text from the layout data. Thus they
        # do not have a type that is not known.
        where = f"{where} (unknown type)"
        if not _is_text(raw_block.get("text")):
            raise LayoutFormatError(
                f"{where} has a type that the layout does not have; the types are: "
                + ", ".join(BLOCK_FIELDS)
            )
        # The text of the page is more important than the type of the block.
        logger.warning("Layout %s has text; using it as a paragraph", where)
        block_type = "paragraph"
    else:
        where = f"{where} ({block_type})"
    fields = BLOCK_FIELDS[block_type]
    if reading.reply and block_type == "figure":
        fields = tuple(name for name in fields if name != "image")
    reading.note_other_content(raw_block, fields, where)
    if block_type == "table":
        return _parse_table(raw_block, where, reading)
    if block_type == "list":
        return _parse_list(raw_block, where, reading)
    if block_type == "figure":
        image = None if reading.reply else _parse_optional_text(raw_block, "image", where)
        if image is not None and not SAFE_IMAGE_PATH.fullmatch(image):
            raise LayoutFormatError(
                f"{where} image must be the path of an image file, with only ASCII letters, "
                "digits, and the characters . _ / -; a caption has the text of a figure"
            )
        return Figure(
            box=_parse_box(raw_block.get("box"), f"{where} box", reading),
            caption=_parse_optional_text(raw_block, "caption", where),
            image=image,
        )

    text = _parse_optional_text(raw_block, "text", where)
    if text is None:
        if block_type == "footnote" and _has_text(raw_block.get("mark")):
            # The mark is text of the page. It must not disappear with the empty block.
            raise LayoutFormatError(f"{where} has a mark and no text")
        return None
    if block_type == "heading":
        level = raw_block.get("level")
        return Heading(
            text=text,
            level=1
            if level is None
            else min(max(_parse_integer(level, f"{where} level"), 1), MAX_HEADING_LEVEL),
            align=_parse_block_alignment(raw_block.get("align"), f"{where} align"),
        )
    if block_type == "quote":
        return Quote(text=text)
    if block_type == "footnote":
        return Footnote(text=text, mark=_parse_optional_text(raw_block, "mark", where) or "")
    if block_type in PAGE_FURNITURE_KINDS:
        return PageFurniture(kind=block_type, text=text)
    return Paragraph(
        text=text,
        align=_parse_block_alignment(raw_block.get("align"), f"{where} align"),
        continued=_parse_flag(raw_block.get("continued"), f"{where} continued"),
    )


def _has_text(value: Any) -> bool:
    """Return True for a JSON value that has text: a string, or a container of one."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_text(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_text(item) for item in value)
    return False


def _is_text(value: Any) -> bool:
    """Return True for a string, and for an integer, which has the digits that a page prints."""
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def _parse_text(value: Any, where: str) -> str:
    if not _is_text(value):
        # A JSON number with a fraction does not keep the digits and the decimal mark
        # that the page prints.
        raise LayoutFormatError(f"{where} must be a string")
    text = str(value)
    if BROKEN_ESCAPE_CHARACTER.search(text):
        raise LayoutFormatError(
            f"{where} has a control character, which a backslash without an escape makes "
            "in a JSON string"
        )
    if SURROGATE.search(text):
        raise LayoutFormatError(
            f"{where} has one half of a character: a surrogate escape without its pair"
        )
    return text.strip()


def _parse_optional_text(raw_object: Mapping[str, Any], name: str, where: str) -> str | None:
    value = raw_object.get(name)
    if value is None:
        return None
    return _parse_text(value, f"{where} {name}") or None


def _parse_alignment(value: Any, where: str) -> str | None:
    """Return the alignment that a value names, or `None` for a value that names none.

    A value with text that is not an alignment is an error: the text can be content of
    the page in the wrong field.
    """
    name = value.strip().lower() if isinstance(value, str) else None
    if name in ALIGNMENTS:
        return ALIGNMENTS[name]
    if _has_text(value):
        raise LayoutFormatError(f"{where} must be one of: left, center, right")
    return None


def _parse_block_alignment(value: Any, where: str) -> str | None:
    """Return the alignment of a text block. `None` is the usual alignment."""
    align = _parse_alignment(value, where)
    return None if align == ALIGN_LEFT else align


def _parse_columns(value: Any, where: str) -> tuple[str, ...]:
    """Return the alignment of each column of a table.

    A table of a model can have the heads of its columns in this field. They are
    content of the page, so text that is not an alignment is an error.
    """
    if isinstance(value, str):
        # One letter for each column, as in a LaTeX table: l, c, or r.
        entries: list[Any] = [letter for letter in value.lower() if letter not in " |"]
    elif isinstance(value, list):
        entries = value
    else:
        entries = [value]
    try:
        columns = tuple(
            COLUMN_LETTERS[entry.strip().lower()]
            if isinstance(entry, str) and entry.strip().lower() in COLUMN_LETTERS
            else _parse_alignment(entry, where) or ALIGN_LEFT
            for entry in entries
        )
    except LayoutFormatError:
        raise LayoutFormatError(
            f"{where} must have only the alignments left, center, and right; write the "
            "heads of the columns as the first row of the table"
        ) from None
    return columns if isinstance(value, (str, list)) else ()


def _parse_rules(value: Any, where: str) -> str:
    """Return the rules of a table. A value with other text is an error."""
    if isinstance(value, bool):
        return RULES_HORIZONTAL if value else RULES_NONE
    name = value.strip().lower() if isinstance(value, str) else None
    if name in RULE_NAMES:
        return RULE_NAMES[name]
    if _has_text(value):
        raise LayoutFormatError(f"{where} must be one of: " + ", ".join(TABLE_RULES))
    return RULES_HORIZONTAL


def _parse_flag(value: Any, where: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    flag = FLAG_VALUES.get(value.strip().lower()) if isinstance(value, str) else None
    if flag is None:
        raise LayoutFormatError(f"{where} must be true or false")
    return flag


def _integer_value(value: Any) -> int | None:
    """Return the integer of a value, which can also be a string of digits or a whole float."""
    if isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        return int(value.strip())
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _parse_integer(value: Any, where: str) -> int:
    number = _integer_value(value)
    if number is None:
        raise LayoutFormatError(f"{where} must be an integer")
    return number


def _parse_span(value: Any, where: str) -> int:
    span = 1 if value is None else _integer_value(value)
    if span is None or span < 1:
        raise LayoutFormatError(f"{where} must be an integer of 1 or more")
    return span


def _parse_box(value: Any, where: str, reading: _Reading) -> Box:
    if isinstance(value, Mapping):
        reading.note_other_content(value, BOX_SIDES, where)
        numbers = [value.get(side) for side in BOX_SIDES]
    elif isinstance(value, list) and len(value) == len(BOX_SIDES):
        # The array form has the sides in the sequence of `BOX_SIDES`.
        numbers = value
    else:
        numbers = []
    if not numbers or not all(
        isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number)
        for number in numbers
    ):
        raise LayoutFormatError(
            f"{where} must be an object with the numbers " + ", ".join(BOX_SIDES)
        )
    if all(0 <= number <= 1 for number in numbers):
        raise LayoutFormatError(
            f"{where} must use the 0-to-{GRID_MAX:.0f} grid of the image, not a scale of 0 to 1"
        )
    if not all(-GRID_TOLERANCE <= number <= GRID_MAX + GRID_TOLERANCE for number in numbers):
        raise LayoutFormatError(
            f"{where} must use the 0-to-{GRID_MAX:.0f} grid of the image, not its pixels"
        )
    left, top, right, bottom = (min(max(float(number), 0.0), GRID_MAX) for number in numbers)
    if right <= left or bottom <= top:
        raise LayoutFormatError(f"{where} must have left < right and top < bottom")
    return Box(left=left, top=top, right=right, bottom=bottom)


def _parse_list(raw_block: Mapping[str, Any], where: str, reading: _Reading) -> ListBlock | None:
    raw_items = raw_block.get("items")
    if not isinstance(raw_items, list):
        raise LayoutFormatError(f"{where} must have an items array")
    items: list[ListItem] = []
    for item_index, raw_item in enumerate(raw_items, start=1):
        item_where = f"{where} item {item_index}"
        if isinstance(raw_item, Mapping):
            # The layout has no list in a list. The items of such a list are items of
            # the one list, in reading order.
            reading.note_other_content(raw_item, LIST_ITEM_FIELDS, item_where)
            text = _parse_optional_text(raw_item, "text", item_where) or ""
            marker = _parse_optional_text(raw_item, "marker", item_where)
        elif _is_text(raw_item):
            text, marker = _parse_text(raw_item, item_where), None
        else:
            raise LayoutFormatError(f"{item_where} must be a string or an object with text")
        if text or marker:
            items.append(ListItem(text=text, marker=marker))
    if not items:
        return None
    return ListBlock(
        items=tuple(items), ordered=_parse_flag(raw_block.get("ordered"), f"{where} ordered")
    )


def _parse_table(raw_block: Mapping[str, Any], where: str, reading: _Reading) -> Table | None:
    raw_rows = raw_block.get("rows")
    if not isinstance(raw_rows, list):
        raise LayoutFormatError(f"{where} must have a rows array")
    # A value can have its header rows in a field of their own, before the other rows.
    raw_header = raw_block.get("header")
    if raw_header is not None and not isinstance(raw_header, list):
        raise LayoutFormatError(f"{where} header must be an array of cells")
    header: list[Any] = raw_header or []
    if header and not all(isinstance(row, list) for row in header):
        header = [header]
    rows: list[tuple[TableCell, ...]] = []
    for row_index, raw_row in enumerate([*header, *raw_rows], start=1):
        if not isinstance(raw_row, list):
            raise LayoutFormatError(f"{where} row {row_index} must be an array of cells")
        rows.append(
            tuple(
                _parse_cell(raw_cell, f"{where} row {row_index} cell {cell_index}", reading)
                for cell_index, raw_cell in enumerate(raw_row, start=1)
            )
        )
    caption = _parse_optional_text(raw_block, "caption", where)
    columns = _parse_columns(raw_block.get("columns"), f"{where} columns")
    rules = _parse_rules(raw_block.get("rules"), f"{where} rules")
    if not any(rows):
        if caption is not None:
            raise LayoutFormatError(
                f"{where} has a caption and no cells; write the rows of the table, or "
                "write the text as a paragraph"
            )
        return None

    raw_header_rows = raw_block.get("header_rows")
    if raw_header_rows is None or isinstance(raw_header_rows, bool):
        # A flag tells that the table has a header row.
        header_rows = int(bool(raw_header_rows))
    else:
        header_rows = _parse_integer(raw_header_rows, f"{where} header_rows")
    return Table(
        rows=tuple(rows),
        caption=caption,
        header_rows=min(max(header_rows, len(header), 0), len(rows)),
        columns=columns,
        rules=rules,
    )


def _parse_cell(raw_cell: Any, where: str, reading: _Reading) -> TableCell:
    if raw_cell is None:
        return TableCell(text="")
    if isinstance(raw_cell, Mapping):
        reading.note_other_content(raw_cell, CELL_FIELDS, where)
        return TableCell(
            text=_parse_optional_text(raw_cell, "text", where) or "",
            colspan=_parse_span(raw_cell.get("colspan"), f"{where} colspan"),
            rowspan=_parse_span(raw_cell.get("rowspan"), f"{where} rowspan"),
        )
    if not _is_text(raw_cell):
        raise LayoutFormatError(f"{where} must be a string or an object with text")
    return TableCell(text=_parse_text(raw_cell, where))


def _without_defaults(values: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON fields that have a value."""
    return {name: value for name, value in values.items() if value is not None}
