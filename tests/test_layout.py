import json
from typing import Any
from unittest import TestCase

from scanwich.layout import (
    Box,
    Figure,
    Footnote,
    FootnoteReference,
    Heading,
    LayoutFormatError,
    LineBreak,
    ListBlock,
    ListItem,
    PageFurniture,
    PageLayout,
    Paragraph,
    Quote,
    Span,
    Table,
    TableCell,
    page_layout_from_json,
    parse_inline,
    plain_text,
)

LOGGER_NAME = "scanwich.layout"


def styles(*names: str) -> frozenset[str]:
    return frozenset(names)


class InlineMarkupTests(TestCase):
    def test_reads_nested_styles_as_flat_spans(self) -> None:
        self.assertEqual(
            parse_inline("plain <b>bold <i>both</i></b> <sc>caps</sc> x<sup>2</sup><sub>n</sub>"),
            [
                Span("plain "),
                Span("bold ", styles("b")),
                Span("both", styles("b", "i")),
                Span(" "),
                Span("caps", styles("sc")),
                Span(" x"),
                Span("2", styles("sup")),
                Span("n", styles("sub")),
            ],
        )

    def test_reads_tag_synonyms_and_capital_letters(self) -> None:
        self.assertEqual(
            parse_inline("<strong>a</strong><EM>b</EM><U>c</U>"),
            [Span("a", styles("b")), Span("b", styles("i")), Span("c", styles("u"))],
        )

    def test_reads_line_breaks_and_footnote_references(self) -> None:
        self.assertEqual(
            parse_inline("one<br>two<br/>three<BR />four<fn> 1 </fn><fn><sup>*</sup></fn>"),
            [
                Span("one"),
                LineBreak(),
                Span("two"),
                LineBreak(),
                Span("three"),
                LineBreak(),
                Span("four"),
                FootnoteReference("1"),
                FootnoteReference("*"),
            ],
        )

    def test_keeps_text_that_is_not_markup(self) -> None:
        text = "a < b & c > d <table> </p> <fn>open <x>"
        self.assertEqual(parse_inline(text), [Span(text)])

    def test_reads_the_entities_of_literal_characters(self) -> None:
        self.assertEqual(
            parse_inline("&lt;b&gt; R&amp;D &quot;q&quot; &apos;a&#39; x&nbsp;y &copy;"),
            [Span("<b> R&D \"q\" 'a' x\u00a0y &copy;")],
        )

    def test_accepts_tags_without_a_partner(self) -> None:
        # A closing tag without its opening tag has no effect.
        self.assertEqual(parse_inline("a</i>b"), [Span("ab")])
        # A style without a closing tag continues to the end of the text.
        self.assertEqual(parse_inline("a<i>b"), [Span("a"), Span("b", styles("i"))])
        # A style that opens two times needs two closing tags.
        self.assertEqual(
            parse_inline("<i>a<i>b</i>c</i>d"),
            [Span("abc", styles("i")), Span("d")],
        )

    def test_plain_text_has_no_markup(self) -> None:
        self.assertEqual(plain_text("<b>a</b><br>b<fn>1</fn> &amp; c"), "a b & c")


def full_page() -> dict[str, Any]:
    """Return the JSON of a page that has each block type and each optional field."""
    return {
        "page_top": "top",
        "content_box": {"left": 100.0, "top": 50.0, "right": 900.0, "bottom": 950.0},
        "blocks": [
            {"type": "page_header", "text": "Running head"},
            {"type": "page_number", "text": "406"},
            {"type": "heading", "level": 2, "text": "A heading", "align": "center"},
            {"type": "paragraph", "text": "Text <i>one</i>.", "continued": True},
            {"type": "paragraph", "text": "Centered", "align": "center"},
            {"type": "quote", "text": "A quotation."},
            {
                "type": "list",
                "ordered": True,
                "items": [{"marker": "a)", "text": "first"}, {"text": "second"}],
            },
            {
                "type": "table",
                "caption": "A caption",
                "header_rows": 1,
                "columns": ["left", "right", "center"],
                "rules": "columns",
                "rows": [
                    ["a", "b", "c"],
                    [{"text": "wide", "colspan": 2}, {"text": "tall", "rowspan": 2}],
                    ["d", ""],
                ],
            },
            {
                "type": "figure",
                "box": {"left": 200.0, "top": 300.0, "right": 700.0, "bottom": 600.0},
                "caption": "Fig. 1",
                "image": "book-figures/page-0001-figure-01.jpg",
            },
            {"type": "footnote", "mark": "1", "text": "A footnote."},
            {"type": "page_footer", "text": "Footer"},
        ],
    }


def blocks_of(*blocks: dict[str, Any]) -> tuple[Any, ...]:
    """Return the blocks that the layout check makes from the JSON of blocks."""
    return page_layout_from_json({"blocks": list(blocks)}).blocks


class PageLayoutJsonTests(TestCase):
    def test_reads_each_block_type(self) -> None:
        self.assertEqual(
            page_layout_from_json(full_page()),
            PageLayout(
                content_box=Box(100, 50, 900, 950),
                blocks=(
                    PageFurniture("page_header", "Running head"),
                    PageFurniture("page_number", "406"),
                    Heading("A heading", level=2, align="center"),
                    Paragraph("Text <i>one</i>.", continued=True),
                    Paragraph("Centered", align="center"),
                    Quote("A quotation."),
                    ListBlock(
                        items=(ListItem("first", marker="a)"), ListItem("second")),
                        ordered=True,
                    ),
                    Table(
                        rows=(
                            (TableCell("a"), TableCell("b"), TableCell("c")),
                            (TableCell("wide", colspan=2), TableCell("tall", rowspan=2)),
                            (TableCell("d"), TableCell("")),
                        ),
                        caption="A caption",
                        header_rows=1,
                        columns=("left", "right", "center"),
                        rules="columns",
                    ),
                    Figure(
                        box=Box(200, 300, 700, 600),
                        caption="Fig. 1",
                        image="book-figures/page-0001-figure-01.jpg",
                    ),
                    Footnote("A footnote.", mark="1"),
                    PageFurniture("page_footer", "Footer"),
                ),
            ),
        )

    def test_json_form_gives_the_same_layout_again(self) -> None:
        layout = page_layout_from_json(full_page())
        self.assertEqual(layout.to_json(), full_page())
        self.assertEqual(page_layout_from_json(layout.to_json()), layout)

    def test_uses_defaults_for_fields_that_are_absent(self) -> None:
        layout = page_layout_from_json(
            {
                "blocks": [
                    {"type": "heading", "text": "H"},
                    {"type": "paragraph", "text": " P "},
                    {"type": "list", "items": ["one"]},
                    {"type": "table", "rows": [["a", None]]},
                    {"type": "figure", "box": [0, 0, 500, 500]},
                    {"type": "footnote", "text": "F"},
                ]
            }
        )
        self.assertEqual(
            layout,
            PageLayout(
                blocks=(
                    Heading("H"),
                    Paragraph("P"),
                    ListBlock(items=(ListItem("one"),)),
                    Table(rows=((TableCell("a"), TableCell("")),)),
                    Figure(box=Box(0, 0, 500, 500)),
                    Footnote("F"),
                )
            ),
        )
        self.assertEqual(layout.page_top, "top")
        self.assertIsNone(layout.content_box)

    def test_reads_a_box_with_named_sides_or_as_an_array(self) -> None:
        # The names leave no doubt about the sequence of the sides. The array has the
        # sides in the sequence left, top, right, bottom.
        named = {"top": 20, "left": 10, "bottom": 400, "right": 300}
        for box in (named, [10, 20, 300, 400]):
            with self.subTest(box=box):
                layout = page_layout_from_json(
                    {"content_box": box, "blocks": [{"type": "figure", "box": box}]}
                )
                self.assertEqual(layout.content_box, Box(10, 20, 300, 400))
                self.assertEqual(layout.blocks, (Figure(box=Box(10, 20, 300, 400)),))

    def test_accepts_values_that_have_one_clear_meaning(self) -> None:
        cases: list[tuple[dict[str, Any], Any]] = [
            # An integer has the digits that the page prints.
            ({"type": "page_number", "text": 406}, PageFurniture("page_number", "406")),
            ({"type": "footnote", "mark": 3, "text": "F"}, Footnote("F", mark="3")),
            ({"type": "footnote", "mark": None, "text": "F"}, Footnote("F")),
            (
                {"type": "table", "rows": [["a", 7, {"text": 8}]]},
                Table(rows=((TableCell("a"), TableCell("7"), TableCell("8")),)),
            ),
            (
                {"type": "list", "items": [5, {"marker": 1, "text": "a"}]},
                ListBlock(items=(ListItem("5"), ListItem("a", marker="1"))),
            ),
            ({"type": "heading", "level": "2", "text": "H"}, Heading("H", level=2)),
            ({"type": "heading", "level": 2.0, "text": "H"}, Heading("H", level=2)),
            ({"type": "heading", "level": None, "text": "H"}, Heading("H", level=1)),
            (
                {"type": "paragraph", "text": "P", "continued": "true"},
                Paragraph("P", continued=True),
            ),
            ({"type": "paragraph", "text": "P", "continued": " False "}, Paragraph("P")),
            ({"type": "paragraph", "text": "P", "continued": None}, Paragraph("P")),
            (
                {"type": "list", "ordered": "TRUE", "items": ["a"]},
                ListBlock(items=(ListItem("a"),), ordered=True),
            ),
            (
                {"type": "table", "header_rows": True, "rows": [["a"], ["b"]]},
                Table(rows=((TableCell("a"),), (TableCell("b"),)), header_rows=1),
            ),
            (
                {"type": "table", "header_rows": "1", "rows": [["a"], ["b"]]},
                Table(rows=((TableCell("a"),), (TableCell("b"),)), header_rows=1),
            ),
            (
                {"type": "table", "header_rows": None, "rows": [["a"]]},
                Table(rows=((TableCell("a"),),)),
            ),
            (
                {"type": "table", "rows": [[{"text": "a", "colspan": "2", "rowspan": 2.0}]]},
                Table(rows=((TableCell("a", colspan=2, rowspan=2),),)),
            ),
            (
                # One letter for each column, as in a LaTeX table.
                {"type": "table", "columns": "lcr", "rows": [["a", "b", "c"]]},
                Table(
                    rows=((TableCell("a"), TableCell("b"), TableCell("c")),),
                    columns=("left", "center", "right"),
                ),
            ),
        ]
        for raw_block, block in cases:
            with self.subTest(raw_block=raw_block):
                self.assertEqual(blocks_of(raw_block), (block,))

    def test_reads_header_rows_that_are_in_a_field_of_their_own(self) -> None:
        self.assertEqual(
            blocks_of(
                {"type": "table", "header": ["h1", "h2"], "rows": [["a", "b"]]},
                {"type": "table", "header": [["g"], ["h"]], "header_rows": 1, "rows": [["a"]]},
                {"type": "table", "header": [], "rows": [["a"]]},
            ),
            (
                Table(
                    rows=((TableCell("h1"), TableCell("h2")), (TableCell("a"), TableCell("b"))),
                    header_rows=1,
                ),
                Table(
                    rows=((TableCell("g"),), (TableCell("h"),), (TableCell("a"),)),
                    header_rows=2,
                ),
                Table(rows=((TableCell("a"),),)),
            ),
        )

    def test_reads_the_side_of_the_page_top(self) -> None:
        for value, side in ((None, "top"), ("Top", "top"), (" RIGHT ", "right"), ("left", "left")):
            with self.subTest(value=value):
                layout = page_layout_from_json({"page_top": value, "blocks": []})
                self.assertEqual(layout.page_top, side)

    def test_reads_an_array_of_blocks_as_a_page(self) -> None:
        self.assertEqual(
            page_layout_from_json([{"type": "paragraph", "text": "P"}]),
            PageLayout(blocks=(Paragraph("P"),)),
        )

    def test_ignores_hints_that_have_no_text(self) -> None:
        layout = page_layout_from_json(
            {
                "blocks": [
                    {"type": "heading", "level": 9, "text": "H", "align": 3},
                    {"type": "paragraph", "text": "P", "align": "left"},
                    {
                        "type": "table",
                        "rules": None,
                        "header_rows": 7,
                        "columns": [None, 2, "centre"],
                        "rows": [["a", "b", "c"]],
                    },
                    {"type": "table", "columns": 3, "rows": [["a"]]},
                    # A field without text is a hint, not content.
                    {"type": "quote", "text": "Q", "id": 4, "confidence": 0.5, "bold": True},
                    {"type": "paragraph", "text": "P", "box": [1, 2, 3, 4], "note": " "},
                ],
                "page": 3,
            }
        )
        self.assertEqual(
            layout.blocks,
            (
                Heading("H", level=3),
                Paragraph("P"),
                Table(
                    rows=((TableCell("a"), TableCell("b"), TableCell("c")),),
                    header_rows=1,
                    columns=("left", "left", "center"),
                ),
                Table(rows=((TableCell("a"),),)),
                Quote("Q"),
                Paragraph("P"),
            ),
        )

    def test_reads_the_names_of_alignments_and_rules(self) -> None:
        cases: list[tuple[dict[str, Any], Any]] = [
            ({"type": "paragraph", "text": "P", "align": "Middle"}, Paragraph("P", align="center")),
            ({"type": "paragraph", "text": "P", "align": "end"}, Paragraph("P", align="right")),
            # The usual alignment of a paragraph.
            ({"type": "paragraph", "text": "P", "align": "justify"}, Paragraph("P")),
            ({"type": "paragraph", "text": "P", "align": "start"}, Paragraph("P")),
            ({"type": "heading", "text": "H", "align": "justified"}, Heading("H")),
            (
                {"type": "table", "columns": ["l", "R", "centre", "justify"], "rows": [["a"]]},
                Table(rows=((TableCell("a"),),), columns=("left", "right", "center", "left")),
            ),
            (
                # The column specification of a LaTeX table.
                {"type": "table", "columns": "l | c|r", "rows": [["a"]]},
                Table(rows=((TableCell("a"),),), columns=("left", "center", "right")),
            ),
            (
                {"type": "table", "rules": "All", "rows": [["a"]]},
                Table(rows=((TableCell("a"),),), rules="grid"),
            ),
            (
                {"type": "table", "rules": "vertical", "rows": [["a"]]},
                Table(rows=((TableCell("a"),),), rules="columns"),
            ),
            (
                {"type": "table", "rules": False, "rows": [["a"]]},
                Table(rows=((TableCell("a"),),), rules="none"),
            ),
            (
                {"type": "table", "rules": True, "rows": [["a"]]},
                Table(rows=((TableCell("a"),),), rules="horizontal"),
            ),
        ]
        for raw_block, block in cases:
            with self.subTest(raw_block=raw_block):
                self.assertEqual(blocks_of(raw_block), (block,))

    def test_rejects_text_in_a_field_that_it_reads_but_cannot_keep(self) -> None:
        # A model can put content of the page in a field that has a different purpose.
        # The text must not disappear, and it must not become a default value.
        columns = (
            "block 1 (table) columns must have only the alignments left, center, and right; "
            "write the heads of the columns as the first row of the table"
        )
        box_shape = "must be an object with the numbers left, top, right, bottom"
        cases: list[tuple[Any, str]] = [
            (
                # The heads of the columns, as many table formats have them.
                {"blocks": [{"type": "table", "columns": ["Part", "Mass"], "rows": [["a", "b"]]}]},
                columns,
            ),
            (
                {
                    "blocks": [
                        {
                            "type": "table",
                            "columns": [{"name": "Part", "align": "left"}],
                            "rows": [["a"]],
                        }
                    ]
                },
                columns,
            ),
            ({"blocks": [{"type": "table", "columns": "Part, Mass", "rows": [["a"]]}]}, columns),
            (
                {"blocks": [{"type": "table", "columns": {"1": "Part"}, "rows": [["a"]]}]},
                columns,
            ),
            # The check finds the text also in a table that has no cells.
            ({"blocks": [{"type": "table", "columns": ["Part"], "rows": []}]}, columns),
            (
                {"blocks": [{"type": "table", "rules": "see the note", "rows": [["a"]]}]},
                "block 1 (table) rules must be one of: none, horizontal, columns, grid",
            ),
            (
                {"blocks": [{"type": "paragraph", "text": "P", "align": "see the note"}]},
                "block 1 (paragraph) align must be one of: left, center, right",
            ),
            (
                {"blocks": [{"type": "heading", "text": "H", "align": ["center", "A title"]}]},
                "block 1 (heading) align must be one of: left, center, right",
            ),
            ({"content_box": "the full page", "blocks": []}, f"content_box {box_shape}"),
            (
                {"blocks": [{"type": "footnote", "mark": "3"}]},
                "block 1 (footnote) has a mark and no text",
            ),
            (
                {"content_box": {"left": 1, "top": 2, "note": "a stain"}, "blocks": []},
                f"content_box {box_shape}",
            ),
            (
                # A description of the picture is not the path of an image file.
                {
                    "blocks": [
                        {"type": "figure", "box": [1, 2, 300, 400], "image": "A wheel, drawn"}
                    ]
                },
                (
                    "block 1 (figure) image must be the path of an image file, with only ASCII "
                    "letters, digits, and the characters . _ / -; a caption has the text of a figure"
                ),
            ),
        ]
        for payload, message in cases:
            self.assert_rejects(payload, message)

    def test_reply_of_a_model_cannot_name_the_image_file_of_a_figure(self) -> None:
        # The pipeline makes the image file from the box and sets the field. A value in a
        # reply would disappear.
        payload = {"blocks": [{"type": "figure", "box": [1, 2, 300, 400], "image": "wheel.jpg"}]}
        self.assertEqual(
            page_layout_from_json(payload).blocks,
            (Figure(box=Box(1, 2, 300, 400), image="wheel.jpg"),),
        )
        with self.assertRaises(LayoutFormatError) as raised:
            page_layout_from_json(payload, reply=True)
        self.assertEqual(
            str(raised.exception),
            "block 1 (figure) has text in a field that the layout does not read; "
            "its fields are: type, box, caption",
        )
        self.assertIn('"image" in 1 object (block 1 (figure))', raised.exception.hint)

    def test_error_hint_names_each_field_that_the_layout_does_not_read(self) -> None:
        paragraphs = [
            {"type": "paragraph", "text": f"Paragraph {number}.", "shelfmark": f"b{number}"}
            for number in range(1, 5)
        ]
        table = {"type": "table", "rows": [[{"text": "a", "tone": "dark", "shelfmark": "t"}]]}
        figure = {
            "type": "figure",
            "box": {"left": 1, "top": 2, "right": 300, "bottom": 400, "tone": "x"},
        }
        with self.assertRaises(LayoutFormatError) as raised:
            page_layout_from_json({"remark": "A note.", "blocks": [*paragraphs, table, figure]})
        # The message is for a log record. It has the first object and no name from the data.
        self.assertEqual(
            str(raised.exception),
            "the page layout has text in a field that the layout does not read; "
            "its fields are: page_top, content_box, blocks",
        )
        # The hint is for the author of the data. It names each field, and tells in how
        # many objects the field is.
        self.assertEqual(
            raised.exception.hint,
            'The layout does not read these fields: "remark" in 1 object (the page layout); '
            '"shelfmark" in 5 objects (the first one is block 1 (paragraph)); '
            '"tone" in 2 objects (the first one is block 5 (table) row 1 cell 1). '
            "This rule is for each object of the reply. Remove these fields from all objects, "
            "or put their text in a field that the layout reads.",
        )

    def test_error_hint_has_a_limited_length(self) -> None:
        fields = {f"field_{number:02d}_{'x' * 60}": "text" for number in range(12)}
        with self.assertRaises(LayoutFormatError) as raised:
            page_layout_from_json({"blocks": [{"type": "quote", "text": "Q", **fields}]})
        hint = raised.exception.hint
        # The hint has 8 names of 40 characters, and the number of the other fields.
        self.assertEqual(hint.count('" in 1 object'), 8)
        self.assertIn(f'"field_00_{"x" * 31}" in 1 object', hint)
        self.assertIn("; and 4 more fields. This rule", hint)

    def test_error_for_a_different_defect_also_has_the_hint(self) -> None:
        # The author of the data gets all the defects that the check found.
        payload = {
            "blocks": [
                {"type": "paragraph", "text": "P", "shelfmark": "b1"},
                {"type": "figure", "box": [0.1, 0.2, 0.3, 0.4]},
            ]
        }
        with self.assertRaises(LayoutFormatError) as raised:
            page_layout_from_json(payload)
        self.assertEqual(
            str(raised.exception),
            "block 2 (figure) box must use the 0-to-1000 grid of the image, not a scale of 0 to 1",
        )
        self.assertIn('"shelfmark" in 1 object (block 1 (paragraph))', raised.exception.hint)
        # An error with no field of this type has no hint.
        with self.assertRaises(LayoutFormatError) as raised:
            page_layout_from_json({"blocks": [{"type": "figure", "box": [0.1, 0.2, 0.3, 0.4]}]})
        self.assertEqual(raised.exception.hint, "")

    def test_ignores_a_content_box_that_is_not_usable(self) -> None:
        # The box is only a hint for the width of an illustration. Numbers that are not a
        # usable box are thus no reason to reject the page. Text is not a hint.
        for content_box in ([], [0, 0, 0, 0], [300, 400, 3000, 4500], {"left": 1}, 5):
            with (
                self.subTest(content_box=content_box),
                self.assertLogs(LOGGER_NAME, level="INFO") as captured_logs,
            ):
                layout = page_layout_from_json(
                    {"content_box": content_box, "blocks": [{"type": "paragraph", "text": "P"}]}
                )
                self.assertIsNone(layout.content_box)
                self.assertEqual(layout.blocks, (Paragraph("P"),))
                self.assertIn("Ignoring the content_box of a page layout", captured_logs.output[0])

    def test_leaves_out_blocks_that_have_no_content(self) -> None:
        layout = page_layout_from_json(
            {
                "blocks": [
                    {"type": "paragraph", "text": "  "},
                    {"type": "paragraph", "text": None},
                    {"type": "paragraph"},
                    {"type": "heading", "text": ""},
                    {"type": "list", "items": ["", {"text": " "}, {}]},
                    {"type": "table", "rows": []},
                    {"type": "table", "rows": [[]]},
                    {"type": "paragraph", "text": "kept"},
                ]
            }
        )
        self.assertEqual(layout.blocks, (Paragraph("kept"),))

    def test_keeps_the_text_of_a_block_with_an_unknown_type(self) -> None:
        with self.assertLogs(LOGGER_NAME, level="WARNING") as captured_logs:
            layout = page_layout_from_json(
                {"blocks": [{"type": "private-type-name", "text": "kept text", "align": "center"}]}
            )
        self.assertEqual(layout.blocks, (Paragraph("kept text", align="center"),))
        # A log record has no text from the layout data.
        self.assertEqual(
            captured_logs.output,
            [
                (
                    f"WARNING:{LOGGER_NAME}:Layout block 1 (unknown type) has text; "
                    "using it as a paragraph"
                )
            ],
        )

    def test_limits_a_box_to_the_grid(self) -> None:
        layout = page_layout_from_json(
            {"content_box": [-20, 0, 1010.5, 1000], "blocks": []},
        )
        self.assertEqual(layout.content_box, Box(0, 0, 1000, 1000))

    def test_page_that_is_not_upright_needs_no_blocks(self) -> None:
        layout = page_layout_from_json({"page_top": "right", "blocks": []})
        self.assertEqual(layout, PageLayout(page_top="right"))
        self.assertEqual(layout.to_json(), {"page_top": "right", "blocks": []})

    def assert_rejects(self, payload: Any, message: str) -> None:
        with self.subTest(message=message), self.assertRaises(LayoutFormatError) as raised:
            page_layout_from_json(payload)
        self.assertEqual(str(raised.exception), message)

    def test_rejects_data_that_does_not_have_the_shape_of_a_layout(self) -> None:
        box_shape = "box must be an object with the numbers left, top, right, bottom"
        cases: list[tuple[Any, str]] = [
            ("text", "the page layout must be a JSON object"),
            ({}, "the page layout must have a blocks array"),
            ({"layout": {"blocks": []}}, "the page layout must have a blocks array"),
            (
                {"page_top": "north", "blocks": []},
                "page_top must be one of: top, right, bottom, left",
            ),
            ({"page_top": 90, "blocks": []}, "page_top must be one of: top, right, bottom, left"),
            ({"blocks": ["text"]}, "block 1 must be an object"),
            ({"blocks": [{"text": "x"}]}, "block 1 must have a type"),
            (
                {"blocks": [{"type": "paragraph", "text": ["line one", "line two"]}]},
                "block 1 (paragraph) text must be a string",
            ),
            (
                {"blocks": [{"type": "heading", "text": "x", "level": "two"}]},
                "block 1 (heading) level must be an integer",
            ),
            (
                {"blocks": [{"type": "heading", "text": "x", "level": True}]},
                "block 1 (heading) level must be an integer",
            ),
            (
                {"blocks": [{"type": "footnote", "text": "x", "mark": ["1"]}]},
                "block 1 (footnote) mark must be a string",
            ),
            (
                # The value has no clear meaning as a flag.
                {"blocks": [{"type": "paragraph", "text": "x", "continued": "yes"}]},
                "block 1 (paragraph) continued must be true or false",
            ),
            (
                {"blocks": [{"type": "list", "ordered": 1, "items": ["a"]}]},
                "block 1 (list) ordered must be true or false",
            ),
            ({"blocks": [{"type": "list"}]}, "block 1 (list) must have an items array"),
            (
                {"blocks": [{"type": "list", "items": [[1]]}]},
                "block 1 (list) item 1 must be a string or an object with text",
            ),
            ({"blocks": [{"type": "table"}]}, "block 1 (table) must have a rows array"),
            (
                {"blocks": [{"type": "table", "rows": [{"cells": []}]}]},
                "block 1 (table) row 1 must be an array of cells",
            ),
            (
                {"blocks": [{"type": "table", "header": {}, "rows": [["a"]]}]},
                "block 1 (table) header must be an array of cells",
            ),
            (
                # A number with a fraction does not keep the digits that the page prints.
                {"blocks": [{"type": "table", "rows": [["a", 0.5]]}]},
                "block 1 (table) row 1 cell 2 must be a string or an object with text",
            ),
            (
                {"blocks": [{"type": "table", "rows": [[True]]}]},
                "block 1 (table) row 1 cell 1 must be a string or an object with text",
            ),
            (
                {"blocks": [{"type": "table", "rows": [[{"text": "a", "colspan": 0}]]}]},
                "block 1 (table) row 1 cell 1 colspan must be an integer of 1 or more",
            ),
            (
                {"blocks": [{"type": "table", "rows": [[{"text": "a", "rowspan": True}]]}]},
                "block 1 (table) row 1 cell 1 rowspan must be an integer of 1 or more",
            ),
            (
                {"blocks": [{"type": "table", "rows": [["a"]], "header_rows": "one"}]},
                "block 1 (table) header_rows must be an integer",
            ),
            ({"blocks": [{"type": "figure"}]}, f"block 1 (figure) {box_shape}"),
            ({"blocks": [{"type": "figure", "box": [0, 0, 5]}]}, f"block 1 (figure) {box_shape}"),
            (
                {"blocks": [{"type": "figure", "box": ["100", "100", "500", "500"]}]},
                f"block 1 (figure) {box_shape}",
            ),
            (
                {"blocks": [{"type": "figure", "box": [0, 0, float("inf"), 5]}]},
                f"block 1 (figure) {box_shape}",
            ),
            (
                {"blocks": [{"type": "figure", "box": {"x1": 1, "y1": 1, "x2": 9, "y2": 9}}]},
                f"block 1 (figure) {box_shape}",
            ),
            (
                {"blocks": [{"type": "figure", "box": [500, 0, 500, 100]}]},
                "block 1 (figure) box must have left < right and top < bottom",
            ),
        ]
        for payload, message in cases:
            self.assert_rejects(payload, message)

    def test_rejects_a_box_that_is_not_on_the_grid(self) -> None:
        pixels = "block 1 (figure) box must use the 0-to-1000 grid of the image, not its pixels"
        scale = (
            "block 1 (figure) box must use the 0-to-1000 grid of the image, not a scale of 0 to 1"
        )
        cases = [
            ([100, 200, 2400, 3100], pixels),
            # A small part of a large image, in pixels.
            ([300, 400, 1200, 1100], pixels),
            ([0.25, 0.5, 0.75, 1], scale),
            ({"left": 0.1, "top": 0.2, "right": 0.3, "bottom": 0.4}, scale),
        ]
        for box, message in cases:
            self.assert_rejects({"blocks": [{"type": "figure", "box": box}]}, message)

    def test_rejects_content_that_it_cannot_keep(self) -> None:
        other_text = "has text in a field that the layout does not read; its fields are:"
        types = (
            "has a type that the layout does not have; the types are: heading, paragraph, "
            "quote, list, table, figure, footnote, page_header, page_footer, page_number"
        )
        cases: list[tuple[Any, str]] = [
            (
                {"blocks": [{"type": "table", "caption": "Table 3", "rows": []}]},
                (
                    "block 1 (table) has a caption and no cells; write the rows of the table, or "
                    "write the text as a paragraph"
                ),
            ),
            (
                {"blocks": [{"type": "table", "rows": [["a"]], "footnotes": ["* note"]}]},
                (
                    f"block 1 (table) {other_text} type, caption, header_rows, columns, rules, "
                    "rows, header"
                ),
            ),
            (
                {"blocks": [{"type": "table", "text": "| a | b |", "rows": []}]},
                (
                    f"block 1 (table) {other_text} type, caption, header_rows, columns, rules, "
                    "rows, header"
                ),
            ),
            (
                {"blocks": [{"type": "table", "rows": [[{"content": "a", "colspan": 2}]]}]},
                f"block 1 (table) row 1 cell 1 {other_text} text, colspan, rowspan",
            ),
            (
                {
                    "blocks": [
                        {
                            "type": "list",
                            "items": [{"text": "a", "items": [{"marker": "a)", "text": "b"}]}],
                        }
                    ]
                },
                f"block 1 (list) item 1 {other_text} marker, text",
            ),
            (
                {"blocks": [{"type": "list", "items": [{"marker": "1.", "content": "a"}]}]},
                f"block 1 (list) item 1 {other_text} marker, text",
            ),
            (
                {"blocks": [{"type": "figure", "box": [1, 1, 500, 500], "text": "labels"}]},
                f"block 1 (figure) {other_text} type, box, caption, image",
            ),
            (
                {"blocks": [{"type": "paragraph", "content": "Some text."}]},
                f"block 1 (paragraph) {other_text} type, text, align, continued",
            ),
            (
                {"blocks": [{"type": "quote", "text": "Q", "source": {"author": "A"}}]},
                f"block 1 (quote) {other_text} type, text",
            ),
            (
                {"blocks": [], "footnotes": [{"mark": "1", "text": "n"}]},
                f"the page layout {other_text} page_top, content_box, blocks",
            ),
            ({"blocks": [{"type": "formula", "latex": "x^2"}]}, f"block 1 (unknown type) {types}"),
            (
                {"blocks": [{"type": "image", "box": [1, 1, 500, 500]}]},
                f"block 1 (unknown type) {types}",
            ),
            (
                {"blocks": [{"type": "section", "title": "T", "blocks": []}]},
                f"block 1 (unknown type) {types}",
            ),
            (
                {"blocks": [{"type": "caption", "text": "C", "label": "Fig. 2"}]},
                f"block 1 (unknown type) {other_text} type, text, align, continued",
            ),
        ]
        for payload, message in cases:
            self.assert_rejects(payload, message)

    def test_rejects_text_that_a_broken_json_escape_changed(self) -> None:
        # In JSON, `\f` and `\b` are escapes of control characters. A model that writes
        # a LaTeX command without an escape for its backslash makes such a character.
        control = (
            "has a control character, which a backslash without an escape makes in a JSON string"
        )
        surrogate = "has one half of a character: a surrogate escape without its pair"
        cases = [
            (
                json.loads('{"blocks": [{"type": "paragraph", "text": "a \\frac b"}]}'),
                f"block 1 (paragraph) text {control}",
            ),
            (
                json.loads('{"blocks": [{"type": "table", "rows": [["a \\beta"]]}]}'),
                f"block 1 (table) row 1 cell 1 {control}",
            ),
            (
                json.loads(
                    '{"blocks": [{"type": "figure", "box": [1, 2, 30, 40], "caption": "\\u0000"}]}'
                ),
                f"block 1 (figure) caption {control}",
            ),
            (
                json.loads('{"blocks": [{"type": "list", "items": [{"marker": "a \\ud83d b"}]}]}'),
                f"block 1 (list) item 1 marker {surrogate}",
            ),
        ]
        for payload, message in cases:
            self.assert_rejects(payload, message)
        # Usual whitespace and a complete character are correct text.
        layout = page_layout_from_json(
            json.loads('{"blocks": [{"type": "paragraph", "text": "a\\tb\\nc \\ud83d\\ude00"}]}')
        )
        self.assertEqual(layout.blocks, (Paragraph("a\tb\nc \U0001f600"),))

    def test_error_message_has_no_text_from_the_data(self) -> None:
        secret = "private text of the page"
        payloads = [
            {"blocks": [{"type": secret, secret: secret}]},
            {"blocks": [{"type": "paragraph", "text": "x", secret: secret}]},
            {"blocks": [{"type": "table", "rows": [[{secret: secret}]]}]},
            {"blocks": [{"type": "figure", "box": {secret: 1}, "caption": secret}]},
            {"blocks": [{"type": "paragraph", "text": "x", "continued": secret}]},
            {"blocks": [{"type": "heading", "text": secret, "level": secret}]},
            {"page_top": secret, "blocks": []},
            {"blocks": [], secret: secret},
        ]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(LayoutFormatError) as raised:
                page_layout_from_json(payload)
            self.assertNotIn("private", str(raised.exception))
            self.assertNotIn("private", repr(raised.exception))
