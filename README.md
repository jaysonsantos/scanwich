# Scanwich

Scanwich turns an image-based PDF into a searchable sandwich PDF. PDFium renders each source
page, an OCR backend returns text polygons, and the tool rebuilds each page with:

- the original rendered page as the visible layer; and
- invisible, selectable text aligned with the OCR polygons.

EasyOCR is the default backend for the CLI and the standard image.
For pip installations, use `pip install "scanwich[easyocr]"`.
The PDF pipeline itself does not import or depend on
EasyOCR-specific result types.

Scanwich can also write a PDF or a photo of a page as a LaTeX document.
See [LaTeX documents](#latex-documents).

## Run with Nix

The flake provides Python, EasyOCR, pypdfium2/PDFium, Pillow, and ReportLab:

```console
nix run . -- \
  /path/to/document.pdf \
  ./document-searchable.pdf \
  -l de pt en \
  --ocr-output-dir ./ocr-results
```

EasyOCR downloads its model files on first use. CPU OCR is the default. To opt into a
supported GPU runtime, pass a backend option:

```console
nix run . -- input.pdf output.pdf --backend-option gpu=true
```

Enter the development shell with `nix develop`. The shell has an editable install: the
`scanwich` and `scanwich-api` commands run the source in `src/` of the working tree.
The shell does not build the package and does not run the tests.
Run the packaged command with `nix run .`. With direnv installed,
run `direnv allow` once and `.envrc` will enter the same default development shell
automatically.

## Container image

Build the image with Docker or Podman. The build performs a small EasyOCR pass so the
German, Portuguese, and English models are stored in the image instead of downloaded on
first use:

```console
podman build --tag localhost/scanwich:dev .
podman run --rm localhost/scanwich:dev --list-backends
```

Pushes to `main` publish a prewarmed `linux/amd64` image to
`ghcr.io/jaysonsantos/scanwich:latest`.

Mount input and output directories when converting a document:

```console
podman run --rm \
  --volume /path/to/input:/input:ro \
  --volume /path/to/output:/output \
  localhost/scanwich:dev \
  /input/document.pdf /output/document-searchable.pdf -l de pt en
```

### OpenAI-compatible image

Build this target to include only the OpenAI-compatible provider.
It excludes EasyOCR, PyTorch, torchvision, OpenCV, and EasyOCR models.

```console
docker build --target openai-compatible --tag scanwich:openai-compatible .
docker run --rm scanwich:openai-compatible --list-backends
```

The image selects `openai-compatible` by default. Pass the API key from your environment:

```console
export OPENROUTER_API_KEY=...
docker run --rm \
  --env OPENROUTER_API_KEY \
  --volume /path/to/input:/input:ro \
  --volume /path/to/output:/output \
  scanwich:openai-compatible \
  /input/document.pdf /output/document-searchable.pdf -l de pt en
```

Use the same commands with `podman` instead of `docker` if you use Podman.
The default build target and published `latest` image retain EasyOCR and its models.
Pushes to `main` also publish `ghcr.io/jaysonsantos/scanwich:openai-compatible` for `linux/amd64`.
Manual workflow runs publish both images too.
The workflow adds commit tags: `sha-<short-sha>` for the standard image and `openai-compatible-sha-<short-sha>` for the OpenAI-compatible image.
Use `nix build .#scanwich-openai-compatible` to build the package without a container.
For direct CLI use, pass `--ocr-backend openai-compatible`.

## LaTeX documents

Scanwich can write PDFs and photos of pages as one LaTeX document.
The purpose is an archive copy of an old book.
The document has the text and the structure of the pages. It is not an image of the pages.
Give the output file a name that ends in `.tex`:

```console
export OPENROUTER_API_KEY=...
scanwich book.pdf book.tex --ocr-backend openai-compatible -l de
scanwich page.ARW page.tex --ocr-backend openai-compatible -l de --rotate 90
scanwich pages/*.ARW book.tex --ocr-backend openai-compatible -l de --rotate 90 \
  --ocr-output-dir book-layouts
```

An input is a PDF, an image that Pillow can read, or a camera raw file.
All the inputs of a command become one document. A PDF gives all its pages, and an image is one page.
For a book of many files, see "A book from many files".

### Model for LaTeX output

For LaTeX output, the `openai-compatible` backend uses the model `~openai/gpt-luna-latest` with the reasoning effort `medium`, on OpenRouter.
You do not give these two values. The searchable PDF and the HTTP API keep their own defaults, see "OpenAI-compatible vision backend".

- **Why this effort:** with `low`, the model writes a line in small capitals as lowercase letters. With `medium`, it keeps the capital forms and marks the small capitals.
- **Cost and time:** on one test page, a photo of 24 megapixels with one drawing, a conversion needed 2 requests, about 0.005 USD, and about 50 seconds. A page with no illustration needs 1 request.
- **Your own values:** `--backend-option model=...` and `--backend-option reasoning_effort=...` replace the defaults for all requests. A different `base_url` needs a `model` of that service.
- **The alias `luna` is a different model ID.** It is `gpt-5.6-luna`, for a Hermes endpoint. See "Hermes and Luna".

`~openai/gpt-luna-latest` is an alias. The provider moves it to a newer model. That has these effects:

- A later conversion of the same book can give different text, different formatting, a different cost, and a different time. Scanwich cannot make it the same.
- The result of a conversion is the `.tex` file, its figure directory, and the layout JSON files of `--ocr-output-dir`. Keep these files.
  With the layout files, a new conversion makes the same document and sends no request. Without them, a new conversion does not give the same document.
- The log names the model that made each reply, when the response has that name: `the response names the model ...`. To convert more pages with that model, give its ID with `--backend-option model=...`.
For a camera raw file, install the decoder with `pip install "scanwich[raw]"`.
LibRaw then decodes all the sensor data. Scanwich does not use the small preview image of the file.
Both Nix packages and both image targets include the decoder.

The document has these parts of the source, in reading order:

- headings, paragraphs, block quotations, and lists;
- bold, italic, underlined, small-capital, superscript, and subscript text;
- tables, with cells that span columns or rows;
- illustrations as image files, which the document includes with `\includegraphics`;
- footnotes, with the reference marks of the source;
- the running head, the running foot, and the printed page number of each page, at their places.

### A book from many files

Give the files of a book in the sequence of its pages, and the document as the last name:

```console
scanwich pages/*.ARW book.tex --ocr-backend openai-compatible -l de --rotate 90 \
  --ocr-output-dir book-layouts
```

- The pages of the document have the sequence of the command. A PDF in the list gives its pages at its place.
- The rules for the pages of one PDF are also the rules across the files.
  These are: a paragraph that continues, a word that the end of a page divides, the names of the illustrations, and one source page for one page of the document.
- A shell sorts the names of a pattern by their characters, so `page10.jpg` is before `page2.jpg`.
  Scanwich then writes a `WARNING` log record. It does not change the sequence. Give such names leading zeros.
- `--rotate` is for all the files. For files with different directions, do not use it. The backend then finds the direction of each page, see "Page direction".
- An option can be between the names. `-l` takes each word after it, so put the names of the files before `-l`, or put `--` after the last language code.
  A language code that has a dot or a path separator is a usage error.

#### Saved layouts

Use `--ocr-output-dir DIRECTORY` for a book.
Scanwich then saves the layout of each page in that directory when the page is complete.
A later conversion sends no request for a page that has a saved layout there.

- **After a failure or a stop:** run the same command again. It sends requests only for the pages that have no saved layout.
- **After a correction by hand:** change the layout file of a page, and run the same command again.
  Scanwich makes the document from the saved layouts and sends no request. It makes each illustration again from the `box` of the layout.
- **After you add, remove, or move a page:** the other pages keep their layouts.
- **For a different model on some pages:** remove the layout files of those pages, and run the command with `--backend-option model=...`. The other pages keep their layouts.

Each layout file has a `source` record. The record tells Scanwich which page the layout is for:

| Field | Content |
| --- | --- |
| `sha256` | The SHA-256 of the bytes of the input file |
| `page` | The number of the page in the input file. An image has the page 1 |
| `turn` | The clockwise angle in degrees that the page image turned by before the backend read it |
| `reading` | The backend, its settings such as the model and the languages, and the DPI of a PDF page |
| `file` | The name of the input file. It is only for you |

- **The input file:** a saved layout is for a page when `sha256` and `page` are those of the input. The name and the place of the input file do not matter.
- **A changed input file:** different bytes give a different hash. The page gets a new request and a new layout file, and the earlier file stays.
- **The direction:** with `--rotate`, the `turn` of the record must be the angle of the option. When it is not, the page fails, and the message tells you what to do.
  Without `--rotate`, Scanwich turns the page image by the `turn` of the record. The boxes of the layout are on that image.
- **The model:** a saved layout from a different model, effort, language list, or DPI stays in use.
  Scanwich then writes a `WARNING` log record with the number of such layouts. Thus you can read the difficult pages with a different model, and the other pages get no new request.

The name of a layout file is `NAME-HASH.json`. `NAME` is the name of the input file, and `HASH` is the first 8 characters of its `sha256`.
A page of a PDF has the name `NAME-page-0001-HASH.json`.
The log names the input file and the layout file of each page.

Scanwich does not change and does not replace a layout file that it uses. To get a new layout:

- For one page, remove its layout file.
- For all pages, use `--ignore-saved-layouts`. Each page then gets its requests, and the new layout replaces the saved one.

A layout file that Scanwich cannot use is a failed page. An example is a file that is not valid JSON after a change by hand.
Scanwich sends no request for such a page: a new layout replaces the file, and the file can have your work. The message has the name of the file and the defect.
Scanwich ignores the other files of the directory.

- **A directory that Scanwich cannot write to:** Scanwich checks the layout directory and the directory of the document before the first request. Such a directory stops the conversion with one error message.
- **Two layout files with the same `sha256` and `page`:** the file with the usual name is in use. When no file has that name, Scanwich uses the first name in the sort sequence and writes a `WARNING` log record.
- **Two conversions with one layout directory:** they can run at the same time, for example a document with pages and a document with `--flow`. Do not convert to the same document two times at the same time.

#### Failed pages

- **With `--ocr-output-dir`:** a page that fails does not stop the other pages.
  At the end, the conversion writes no document. It has the exit code 1, and its message lists each failed page with its input file and its error.
  The same command again sends requests only for the pages of that list. Read the error of each page first: a page that fails again needs a correction of its cause.
- **Without `--ocr-output-dir`:** the conversion stops at the first failed page.
  Each more page has a cost, and a new run sends the request of each page again.
  A conversion of more than one page writes a `WARNING` log record about this before the first request.
- **A service that gives no reply:** when 3 pages in sequence fail with no reply from the service, the conversion starts no more pages.
  No reply is a connection that fails, a timeout, an HTTP error status, a rate refusal after its waits, or a key that is not there.
  The cause is then probably the service, the key, the quota, or an option, and not a page. When the service gives replies again, run the same command.
- **A difficult page:** a page with a reply that Scanwich cannot use does not count for that rule, and a page that gets a layout starts the count again.
  Thus difficult pages at the start of a book do not stop the other pages.
- **A stop:** Ctrl-C, SIGTERM, and SIGHUP stop the conversion in a clean way. SIGHUP is the signal of a terminal that closes or a connection that is lost.
  The exit codes are 130, 143, and 129. The layouts that are saved stay, and no work file stays. A page that is in work at that moment is not saved.
  Run the same command again to continue.

A document with a page that is not there is not a correct archive copy. Thus a conversion with a failed page writes no document.

#### Parallel pages

A request for a photo of a book page needs 30 to 40 seconds, so Scanwich reads some pages at the same time.

- **The number:** the default is 4 pages. `--parallel-pages NUMBER` sets it, and `--parallel-pages 1` reads one page after the other.
- **The document:** it does not depend on the number, or on the sequence of the replies. The cost is the same also.
- **The memory:** Scanwich makes the image of a page a short time before it reads the page, and removes the image when the illustrations of the page are complete.
  Not more than `NUMBER` + 2 page images exist at one time, in memory and in the temporary directory of the system. This is so for a book of each size.
  A failed page keeps only its error message.
- **The log:** each record of the work for a page starts with the page, for example `INFO: [page 12/200] ...`.
- **A service that limits the rate:** when the service answers with the HTTP status 429, all pages wait before their next request. Then the request goes again.
  The waits are 30, 60, and 120 seconds, or the time that the service asks for, to a maximum of 300 seconds. After the third wait, the page fails.
  A refused request has no reply and no cost. The replies that a page got before the refusal count for its limit of requests.
  When you see such waits, use a smaller number.
- **A backend with only synchronous methods:** it gets one request at a time, from one thread, see "OCR plugins".

#### From a Python program

`scanwich.pipeline.convert_to_latex(inputs, output_tex, backend=backend)` is the function of the command. Its options have the names of the options of the command.

- **The event loop:** the function starts an event loop of its own. A thread with a running event loop cannot call it, and gets a `RuntimeError` that tells this.
  Call it from a different thread there, for example with `await asyncio.to_thread(convert_to_latex, ...)`.
- **A stop:** `stop_signals` has the signals that stop the conversion as Ctrl-C does, for example `(signal.SIGTERM,)`. The function then raises `ConversionStopped`, which has the `signal_number`.
  This needs the main thread. In a different thread, a signal keeps its usual effect. The default is no signal.

### Pages of the document

Each page of the source is one page of the document, in the same sequence. Thus you can compare the document with the source page by page.
The paper is A4, and the text area is 420pt by 672pt.

- The content of a source page is in a box, in the `sourcepage` environment. A source page without content is an empty page.
- When the content is too high for the text area, the page shows it at a smaller size. Thus a source page never becomes two pages.
  The TeX engine then writes a warning that has the number of the source page and the size, for example `Package scanwich Warning: Source page 12 is at 87 percent of its size`.
  LuaLaTeX, XeLaTeX, and pdfLaTeX write the warning to the terminal and to the log file. Tectonic writes it only to the log file, and it keeps that file only with `--keep-logs`.
  Scanwich does not start a TeX engine, so the conversion cannot tell you which page this is. Its log tells you to read the warnings of the engine.
  The size has no lower limit: a page with very much content can be too small to read. `--flow` is the layout for such a book.
- The running head and the page number are in the head line or the foot line of the page: at the left, at the center, or at the right. The page number is the printed number of the source.
- A paragraph that continues on the next page stays in two parts. The second part has no indent.
- When the page boundary divides a word with a hyphen, the complete word is on the first of the two pages, and the next page starts after it.
  A search then finds the word, and a copy of the text has it in one piece. A comment line in the document shows the place. Compare that word with the source.
- A footnote is below the content of its page, with its reference mark of the source.
- An illustration is at its place in the sequence of the page, at the center of the line.

`--flow` gives a document with flowing text. Its pages are not the pages of the source:
a paragraph that continues on the next page becomes one paragraph, a table is a `longtable` environment that can continue on a new page, and LaTeX gives new numbers to the footnotes.
The running head and the page number are then only in a comment line.

Scanwich crops each illustration from the page image.
The illustrations of `book.tex` are JPEG files of quality 95 in the directory `book-figures` next to the document.
Their names are `page-0001-figure-01.jpg`, `page-0001-figure-02.jpg`, and so on.
Keep that directory with the document.
LaTeX reads the name of the directory, so the name has only ASCII letters, digits, and the characters `.`, `_`, and `-`.
For a document name with other characters, a hyphen replaces them, and the name gets a part of a hash of the document name.
For example, `my book.tex` has the directory `my-book-2c7e412c-figures`. Thus two documents in one directory do not share their illustrations.
A new conversion replaces the files of the earlier conversion that have a name of the form `page-NNNN-figure-NN.jpg`. It does not remove a file with a different name.

Scanwich writes the document after all pages are complete. A conversion that fails does not change an earlier document or its illustrations.
The new illustrations first go to the figure directory under temporary names. Only when all of them are there do the new illustrations and the new document get their names, and then Scanwich removes the earlier illustrations.
A temporary name starts with a dot and ends with `.scanwich-new` or `.scanwich-old`. A conversion removes these files.
During a conversion, the new illustrations wait in a directory next to the document. Its name starts with a dot and ends with `.scanwich-tmp`, and the conversion removes it at its end.
The illustrations of a book are large in sum, and the temporary directory of a system can be in memory.
If the system stops a conversion in its last step, such files can stay. A file that ends with `.scanwich-old` is then an illustration of the earlier document: remove the dot and the suffix from its name to use it again.
A conversion fails when a page has no usable layout, and when no page has content. The error message has the number of the page.
For the other pages of such a conversion, see "Failed pages".
A page with no content in a document that has content gives a `WARNING` log record.

Scanwich does not start a TeX engine. Compile the document with LuaLaTeX, XeLaTeX, or Tectonic:

```console
latexmk -lualatex book.tex
tectonic --keep-logs book.tex
```

With `--keep-logs`, Tectonic keeps `book.log`. Search that file for `scanwich Warning` to find each source page that is at a smaller size.

These engines read all Unicode text.
pdfLaTeX also compiles the document when the text has only characters that pdfLaTeX knows.
The document uses the packages `iftex`, `fontspec`, `graphicx`, and `array`. With `--flow`, it also uses `longtable`.
It uses `multirow` only when that package is installed.

Small capitals are capital letters at a smaller size, from the command `\sourcesmallcaps` of the document.
The document thus does not print a lowercase letter where the source has a capital form: not in bold or italic text, and not with a font that has no small capitals.
The text of the PDF has the capital letters. The preamble of the document shows how to use the small capitals of the font.

The text font of the document does not have some characters, for example a prime mark, a not-equal sign, and Greek letters.
With LuaLaTeX, XeLaTeX, and Tectonic, such a character comes from the font Latin Modern Math when that font is installed. TeX Live has it in the package `lm-math`.
The text of the PDF then has the character of the source, so a search finds it and a copy is correct.
Without that font, and with pdfLaTeX, a math command prints the character. The page then looks the same, but the text of the PDF can have a different character.
With pdfLaTeX, the text of the PDF has `R`, `P`, and `Q` for the integral, sum, and product signs.

### Page layout JSON

A layout backend returns the structure of each page as JSON. The LaTeX writer makes all the LaTeX from that JSON.
The backend does not write LaTeX. This has three effects:

- The writer escapes each character of the text. Thus the text of a page cannot add a LaTeX command.
  The writer also leaves out what stops LuaLaTeX, XeLaTeX, and Tectonic, for example a line break that has no line to end.
  The compile tests give text that is hard for LaTeX to each engine that is installed.
- Scanwich can check a reply before it uses it. The `openai-compatible` backend asks the model to correct a reply that it cannot use.
- An illustration has a box on the page image. Scanwich needs that box to crop the illustration.

`--ocr-output-dir` saves the JSON of each page in one file, with a `source` record. See "Saved layouts" for the record and for the name of the file.
The module `scanwich.layout` defines the JSON.
One page has `blocks` in reading order. The block types are `heading`, `paragraph`, `quote`, `list`, `table`, `figure`, `footnote`, `page_header`, `page_footer`, and `page_number`.
Text can have the tags `<b>`, `<i>`, `<u>`, `<sc>`, `<sup>`, `<sub>`, `<br>`, and `<fn>mark</fn>`.
The text in `<sc>` has the small capitals as capital letters, as the page prints them, and a large capital letter is before the tag: `C<sc>HAPTER</sc> II`.
When a model does not give the tag, the document then prints capital letters of the usual size, and not lowercase letters.
A text in `<sc>` that has lowercase letters uses them for the small capitals, and its capital letters are large capitals.
A `page_header`, a `page_footer`, and a `page_number` can have a `position`: `top-left`, `top-center`, `top-right`, `bottom-left`, `bottom-center`, or `bottom-right`.
The check also accepts a value that has one clear meaning: the two words in each sequence, `upper` and `lower`, and only one of the two words, for example `top` or `right`.
The document gives the usual place to the part that the layout does not have:

- A running head is at the top center, and a running foot at the bottom center.
- A page number is at the top when it comes before the content of the page, and else at the bottom.
- At the bottom, a page number is at the center. At the top, it is at the outer side of the page: at the left for an even number and at the right for an odd number.
  The reason is that a book prints an even number on a left page. A number that is not in digits is at the right.
A table can have `"double_rule": true` for a double rule above the table and below it.
A box is `{"left": 120, "top": 300, "right": 880, "bottom": 560}` on a 0-to-1000 grid of the page image.
The names of the sides leave no doubt about their sequence. The check also accepts the array `[left, top, right, bottom]`.

Text of a reply does not disappear without an error. The check keeps the text, or it rejects the reply:

- It accepts a value that has one clear meaning. Examples are an integer where the page prints digits, `"continued": "true"`, header rows in a `header` field, and `"align": "middle"`.
- It rejects a reply that has text in a field that the layout does not read. Examples are a list in a list item, and a table with a caption and no rows.
- It rejects text that it cannot keep in a field that the layout reads. Examples are the heads of the columns in the `columns` field of a table, and a value of `align` or `rules` that is not one of the names.
  A figure of a reply has no `image` field, because Scanwich makes the image file from the box.
  A saved layout has no such field also. In a layout file that you change, Scanwich ignores the field.
- It rejects a box that is not on the grid: a box on a scale of 0 to 1, and a box with a value above 1020.
- It rejects text that has a control character. In JSON, a backslash without an escape makes such a character, for example in `\frac`.
- It ignores a field that has no text and that the layout does not read, for example a number or a flag.

The check cannot find each wrong reply:

- It ignores a number in a field that the layout does not read, also when the number is content of the page. Examples are a table cell that has its value as a number in a different field, and the first number of a numbered list.
  An integer is text only in a field that the layout reads as text.
- It accepts a box that has its sides in a different sequence, and a box in pixels that has no value above 1020.
- It cannot tell the JSON escapes `\n`, `\r`, and `\t` of a backslash without an escape from whitespace.
- A JSON object that has one key two times keeps only the last value.

An error message of the check has no text from the reply, so a log record can have it.
The correction request to the model has more: it names each field that the layout does not read, and the number of objects that have the field.

LaTeX output needs a backend that has the method `recognize_layout(image_path)`.
The `openai-compatible` backend has it. EasyOCR returns text lines without structure, so it does not have it.

### Box of an illustration

A model that reads a full page gives a box for an illustration that is frequently not exact.
On a photo of a page, the box can be too low by a third of the illustration, and the crop then cuts the illustration.
Thus Scanwich sends one more request for each illustration, when the backend has the method `locate_figure(image_path, estimate=..., caption=...)`.
The `openai-compatible` backend has it.

- The request has only the part of the page image around the first box. The margin at each side is 100 units of the grid, or half of the size of the box when that is more.
- The backend returns the box of the illustration in that part, and the sides where the part cuts the illustration.
  The caption and the center of the first box tell the model which illustration to find when the part shows more than one.
- Scanwich sends a larger part when the backend reports a cut side, and when the new box is nearer than 30 units to a side of the part. A label of the illustration can then be outside the part.
- The crop has the new box and a margin of 5 units at each side. The layout JSON also has the new box.
- An `INFO` log record has the new box and the box of the page layout.

A crop that is too large loses nothing, and a crop that cuts the illustration loses a part of it. Thus each case of doubt gives a `WARNING` log record, and the larger crop when there are two boxes. Compare that illustration with the source.

- The backend still reports a cut side after the last request: the crop goes to the border of the part at that side.
- The new box has no part of the box of the page layout, or it has more than 4 times its area, or less than a quarter: the crop has the two boxes.
- A request for the box fails with an error of each type, or the backend finds no illustration in the part: there is only the box that the backend gave before, and the crop uses it. The conversion continues.

The number of requests has these limits:

- A request for a box has no correction request. One illustration gets a maximum of 3 requests for its box.
- One page gets a maximum of 12 requests for boxes. The other illustrations of that page use the box of the page layout, with one `WARNING` log record.
- The request for the layout of a page has a maximum of `max_attempts` requests, which is 3. Without `--rotate`, a page that is not upright gets that request two times.
- Thus one page has a maximum of 18 requests, and of 15 with `--rotate`. The usual numbers are 1 request for a page with no illustration and 2 requests for a page with one illustration.
- These limits are for the requests that get a reply. A request that the service refuses for the rate has no reply and no cost. It goes again after a wait, see "Parallel pages".

Each request has a cost. The request for a box is small: it has a part of the page, and its reply has only a box.

### Page direction

A photo can show a page that is not upright. A model reads such a page badly.
Without `--rotate`, the backend first finds the side of the image where the top of the page is.
The model does not read a page that is not upright. Scanwich turns the page image and sends a second request for that page.
When the backend reports that the page is not upright after that turn, the conversion fails.

`--rotate DEGREES` gives the direction of all pages: Scanwich turns each page image clockwise by 0, 90, 180, or 270 degrees before recognition.
The backend then gets no request to find the direction, and each page needs one request for its layout.
Use it when all pages have the same direction. `--rotate 0` is for pages that are upright.
Scanwich uses the blocks of the backend also when the backend reports a different direction, and writes a `WARNING` log record.

### Differences from the source

Compare the document with the source. A model can read a character incorrectly.
These differences are intended:

- The lines of a paragraph are not the lines of the source. LaTeX makes new lines for the width of the text area.
- A word that the page boundary divides is complete on the first of the two pages. See "Pages of the document".
- A running head and a page number are at the left, at the center, or at the right of the head line or the foot line. Their distance from the text is that of the document.
  The start of each page also has a comment line, for example `% Source page 3 (printed page number: 47)`.
- A table has the rules of one of four styles: no rules, horizontal rules, also rules between the columns, or a rule around each cell.
  The rule above the table and the rule below it can be double rules. A different double rule becomes one rule.
- All text has the size of the document. A caption or a note that the source prints in a smaller size is not smaller.
- A paragraph has the width of the text area, also when the source prints it on a narrow measure.
- Scanwich estimates the widths of the table columns from the number of characters in the cells.
- The document has one column. Text that is at the side of an illustration comes after the illustration.
- A formula is plain text. Scanwich does not write LaTeX math for it.
- An illustration is a part of the page image. Scanwich does not correct its colors or its perspective.

The default font of LaTeX has Latin text only.
LaTeX prints nothing for a character that the font does not have. It then writes a `Missing character` message to the terminal and to the log.
Scanwich writes common mathematical symbols, and a Greek letter that is alone, with a math font, so the document has them.
For text in other scripts, set a font that has them with `\setmainfont` in the document.
A document without content has one empty page, because LaTeX makes no PDF for a document that has no page.

The HTTP API has no LaTeX output.

## OCR plugins

Select a provider with `--ocr-backend NAME`; inspect installed providers with
`scanwich --list-backends`. Providers are Python entry points in the
`scanwich.ocr_backends` group. An external package registers a factory like this:

```toml
[project.entry-points."scanwich.ocr_backends"]
tesseract = "my_ocr_package:make_backend"
```

The factory receives keyword-only `languages` and `options` arguments and returns an
object with this method:

```python
def recognize(self, image_path: Path) -> Sequence[OcrRegion]: ...
```

Every `OcrRegion` contains text, optional confidence, and four clockwise pixel points
starting at the top-left. This normalized boundary keeps rasterization and PDF assembly
unchanged when another OCR engine is installed.

A provider that can recognize the structure of a page also has this method:

```python
def recognize_layout(self, image_path: Path, *, upright: bool = False) -> PageLayout: ...
```

LaTeX output uses that method. `scanwich.layout.page_layout_from_json` makes a `PageLayout` from JSON and checks it.
The page image has no EXIF orientation.
For a page that is not upright, a provider can return `PageLayout(page_top="right")` without blocks. Scanwich then turns the page image and calls the method again.
With `upright=True`, the caller knows that the page is upright. The provider then does not look for the direction: it reads the page and returns its blocks.

A provider can also have this method, for the exact box of one illustration:

```python
def locate_figure(
    self, image_path: Path, *, estimate: Box | None = None, caption: str | None = None
) -> FigureLocation | None: ...
```

The image shows a part of a page. `estimate` is a first box of the illustration on the 0-to-1000 grid of that image, and `caption` is its caption.
The result has the box on the same grid and the sides where the image cuts the illustration. It is `None` when the image shows no illustration.
`scanwich.layout.figure_location_from_json` makes a `FigureLocation` from JSON and checks it.

For a book, a provider can also have these methods:

```python
async def recognize_layout_async(self, image_path: Path, *, upright: bool = False) -> PageLayout: ...
async def locate_figure_async(
    self, image_path: Path, *, estimate: Box | None = None, caption: str | None = None
) -> FigureLocation | None: ...
def layout_settings(self) -> Mapping[str, Any]: ...
```

- **The async methods:** LaTeX output awaits them for `--parallel-pages` pages at the same time, in one event loop. They are additions to the synchronous methods.
  When the provider has a coroutine method `aclose()`, Scanwich awaits it before the event loop ends.
- **A provider with only the synchronous methods:** it gets one call at a time, and all its calls come from one thread, the same thread for the whole conversion.
  Scanwich does not know that such a provider permits calls from more than one thread. Ctrl-C then has an effect only after the call that is in work.
  That thread is not the thread of the constructor: the command makes the provider in the main thread. Thus do not make an object in the constructor that only its thread can use, for example a `sqlite3` connection. Make it at the first call.
- **`layout_settings()`:** it returns the settings that change a layout, for example the model, as JSON values with no secret. They go to the `source` record of each saved layout.
- **`scanwich.ocr.NoReplyError`:** raise it from these methods when a request fails and the page is not the cause. Examples are a connection that fails, a timeout, and an HTTP error status.
  LaTeX output starts no more pages after 3 pages in sequence with this error. For a reply that the provider cannot use, raise an error of a different type.
- **`scanwich.ocr.RateLimitError`:** it is a type of `NoReplyError`. Raise it when the service answers that it gets too many requests. `retry_after` is the wait in seconds that the service asks for.
  Scanwich then makes all its requests wait, and sends the request again: it calls the method again.
  When the call had replies with a cost before the refusal, give `resume`: a function with no argument that sends the refused request again and returns an awaitable for the result of the call. Scanwich then awaits `resume()` and does not call the method again.
- **A layout that the provider returns as an object:** with `--ocr-output-dir`, Scanwich saves it as JSON and reads it back with the check of `scanwich.layout.page_layout_from_json`.
  A layout that the check refuses is a failed page, and it gets no file. The document is from the layout that the file gives, so that a later conversion gives the same document.

Backend options use `KEY=VALUE`; JSON values are decoded automatically. EasyOCR passes
top-level options to `easyocr.Reader`. A `readtext` JSON object is passed to
`Reader.readtext`, for example:

```console
scanwich input.pdf output.pdf \
  --backend-option 'readtext={"batch_size": 4}'
```

### OpenAI-compatible vision backend

The built-in `openai-compatible` backend sends each rasterized page to an OpenAI-compatible
chat-completions endpoint. It defaults to OpenRouter and the image-capable
`deepseek/deepseek-v4-flash-vision-exp` model.
LaTeX output has a different default model and effort, see "Model for LaTeX output". That model did not get a test for the text regions of a searchable PDF.
Install the SDK with `pip install "scanwich[openai-compatible]"`.
Both Nix packages and image targets include the SDK.
Set the API key in the environment:

```console
export OPENROUTER_API_KEY=...
scanwich input.pdf output.pdf \
  --ocr-backend openai-compatible \
  -l de pt en
```

The backend accepts `model`, `base_url`, `api_key_env`, `timeout`, `max_tokens`,
`reasoning_effort`, and `max_attempts` backend options. Set `base_url` and `api_key_env` to use
another OpenAI-compatible service:

```console
scanwich input.pdf output.pdf \
  --ocr-backend openai-compatible \
  --backend-option base_url=https://example.invalid/v1 \
  --backend-option api_key_env=EXAMPLE_API_KEY \
  --backend-option model=provider/vision-model
```

The backend accepts PNG, JPEG, WebP, and GIF page images.
It rejects a page image when the decoded image and its copies need more than 512 MiB of memory.
For an RGB page, that limit is about 44 million pixels. For a page with transparency, it is about 33 million pixels.
It sends each page image as a JPEG in base64 text to the configured service and its
selected model provider. Do not use this backend for documents that must remain local.
The page image file does not change, so the PDF keeps the original image.
The JPEG keeps the EXIF orientation of the page image and no other metadata. The backend does not turn the pixels.
Some providers reject an image when its base64 text is larger than 5 MiB.
When the JPEG is above that limit, the backend makes the image smaller and writes a `WARNING` log record.
The regions in the result always use the pixels of the original page image.

One backend instance is one session. It makes one random session ID.
The CLI and the API make one backend instance for each document.
With the default `base_url`, the backend sends that ID in the `X-Session-Id` header of each request.
OpenRouter then shows the requests of one document as one session.
Other services do not receive the header.

When a response has `usage.cost`, the backend writes one `INFO` log record for the request.
The record has the session ID, the model, the cost of the request, and the total of the session.
The last record of a session thus has the cost of the document. OpenRouter reports the cost in credits.
A failed request has no `usage.cost`, so the total does not include it.
When a response has `usage.prompt_tokens` and `usage.completion_tokens`, the backend also writes one `INFO` log record with the two numbers.
The record also has the reasoning tokens when the response has `usage.completion_tokens_details.reasoning_tokens`. They are a part of the completion tokens.
When the response names a model, the record ends with `; the response names the model ...`. For a model ID that is an alias, this is the model that made the reply.
Each record names the model of its request, so the records of LaTeX output name the model for LaTeX output.

```text
INFO: OpenAI-compatible session 2f0c6f0e-5d0b-4c53-9d57-3f6a4f1f6f0a, model provider/vision-model: usage.cost=0.001500 for this request, total 0.003000 for 2 request(s)
```

A model can return a reply that is not the requested JSON.
An example is a text that has a double quote without an escape.
The backend then sends a second request. That request adds the unusable reply and a description of the defect, and it asks the model for a correct reply.
The backend does not send the first request again. With a temperature of 0, the model mostly repeats the defect.
`max_attempts` is the maximum number of requests for the regions of one page. The default is 3. Set it to 1 to send no correction request.
The request for the layout of a page, which LaTeX output uses, has the same correction requests.
Each correction request has a cost. The session total includes it.
The backend writes one `WARNING` log record for each unusable reply. The record has the error and no text from the reply.
If the last reply is also unusable, the page fails with the error of that reply.
The backend sends no correction request when the request fails, when the provider stops the reply before its end, or when the reply has no text.
A correction request has text from the model reply, and a provider can repeat request input in an error.
Thus the error of a failed correction attempt has no text from the provider. It has the error type and the HTTP status, or the `finish_reason`.
The request for plain text has no JSON, so it has no correction request.

```text
WARNING: OpenAI-compatible model provider/vision-model returned an unusable reply in attempt 1 of 3 (RuntimeError: OpenAI-compatible service returned invalid JSON: Expecting ',' delimiter: line 1 column 1343 (char 1342))
INFO: OpenAI-compatible model provider/vision-model corrected its reply in attempt 2 of 3
```

## HTTP API

Install the API and a backend:

```console
pip install '.[api,openai-compatible]'
scanwich-api --config api.example.json
```

The server listens on `127.0.0.1:8000`. Open `/docs` for the request schema.
`GET /backends` lists installed backends that the configuration enables.
`POST /ocr/{backend_name}` accepts a PDF or an image in the `file` multipart field.
The API accepts PDFs with any number of pages. It makes one output page for each source page.
The older `image` field name stays valid. Send only one of the two fields.
Set the typed `output` multipart field to `pdf`, `text`, or `pdf+text`. The default is `pdf`.
The JSON response uses `output` to identify its Pydantic response model:

| Output | Response fields |
| --- | --- |
| `pdf` | `output`, `pdf_base64` |
| `text` | `output`, `text` |
| `pdf+text` | `output`, `pdf_base64`, `text` |

Decode `pdf_base64` to obtain a searchable PDF with the source pages and invisible text.
For a PDF upload, the API infers each page's DPI as the CLI does. Output pages keep the source
page dimensions. For an image upload, the API uses 300 DPI for the PDF page dimensions.
Set the optional `dpi` multipart field to override the DPI for every page.
The server limits that field to `max_dpi`, which is 600 by default. Higher values return 422.
This limit bounds the rendered page size, and with it the memory of one request.
The API recognizes pages in order, one page at a time, and sends one backend request for each page.
A document with many pages needs proportionally more time and provider quota.
Text output preserves the provider's line breaks and spacing.
Text output separates pages with a form feed (`\f`).
The OpenAI-compatible backend requests plain text without JSON or coordinates for text output.
Combined output makes two provider calls: one for PDF polygons and one for plain text.
Other plugins provide text from their regions, with one line per region.
Invalid output choices return 422.

```console
curl http://127.0.0.1:8000/ocr/openai-compatible \
  -F file=@document.pdf -F model=deepseek -F output=text -F languages=en -F languages=pt
```

Set provider options in `backends.openai-compatible.options` in the JSON configuration.
Set `model_aliases` to map aliases such as `deepseek` and `glm-ocr` to exact provider model IDs.
The built-in `glm-ocr` alias uses `zai-org/GLM-OCR`.
Select an endpoint that serves that model, or change the alias to your provider's model ID.
The [GLM-OCR model card](https://huggingface.co/zai-org/GLM-OCR) describes its server setup and supported prompts.
For PDF output, the endpoint must return text polygons in the requested JSON format.
Model IDs, including a leading `~`, pass to the provider unchanged.
The built-in `deepseek` alias uses the existing default model. Configuration can replace that alias.
Unknown aliases pass through as model IDs. Alias resolution uses one lookup.
The CLI also accepts `--backend-option 'model_aliases={"deepseek":"~deepseek/your-model"}'`.
Clients can select a model on backends that allow it. Server configuration controls endpoint URLs and credential environment variables.
A backend allows the `model` field when its entry-point factory declares `request_options = frozenset({"model"})`.
The `openai-compatible` backend declares it. Other backends reject the `model` field with 422.

The OpenAI-compatible backend exposes `await backend.recognize_async(path)` and `await backend.aclose()`.
Use `await backend.recognize_text_async(path)` for plain text.
It shares request construction and response checks with `recognize(path)`.
The API uses native async calls when the backend provides them.
It runs synchronous plugins in a worker thread. Each request creates its own backend instance.
The API closes async clients and removes temporary images after each request.

The default upload limit is 20 MiB. Set `max_upload_bytes` to change it.
The former name `max_image_bytes` still applies when `max_upload_bytes` is absent.
Middleware rejects the request body before the multipart parser reads it.
The body limit is `max_upload_bytes` plus 64 KiB for multipart overhead.
The page count has no limit by default. Set `max_pages` to reject longer documents.
Set `max_dpi` to change the DPI ceiling for the `dpi` field.
Invalid images and PDFs return 422; oversized uploads and excessive page counts return 413;
backend failures return 502.
Backend cleanup errors are logged. They do not change the response.
`scanwich-api` writes log records of level `INFO` and higher to standard error.
When you run Uvicorn directly, set its `--log-config` option to see the `INFO` records of Scanwich.
The server has no authentication. Keep the default local address or put an authenticated proxy before the server.
Set `SCANWICH_API_CONFIG` to a JSON file path for `uvicorn scanwich.api:create_app --factory`.
Pydantic Settings loads and checks the configuration.
Explicit arguments take priority over environment variables, then JSON file values, then defaults.
Use `SCANWICH_API_MAX_UPLOAD_BYTES`, `SCANWICH_API_MAX_PAGES`, `SCANWICH_API_MAX_DPI`,
`SCANWICH_API_HOST`, and `SCANWICH_API_PORT` for environment overrides.
Host and port settings apply to `scanwich-api`. Uvicorn controls its own bind address when you run it directly.
Use `__` for nested fields, such as `SCANWICH_API_BACKENDS__OPENAI-COMPATIBLE__OPTIONS__MODEL`.
Pydantic models define the API response schemas in `/docs`.
Both Nix packages and container targets include the API dependencies and `scanwich-api` command.
Use `--entrypoint /opt/scanwich/bin/scanwich-api` to start the API in a container.
Pass `--host 0.0.0.0` and publish port 8000 for container access.

### Hermes and Luna

The `luna` alias resolves to `gpt-5.6-luna`.
Hermes can route this model through its authenticated `openai-codex` provider.
Its API runs locally; the model runs through the upstream provider.

Add these routes under `platforms.api_server.extra` in your Hermes configuration:

```yaml
model_routes:
  luna:
    model: gpt-5.6-luna
    provider: openai-codex
  gpt-5.6-luna:
    model: gpt-5.6-luna
    provider: openai-codex
```

Enable the Hermes API server and set its `API_SERVER_KEY`.
Restart the Hermes gateway after the configuration change.
Set `base_url` in `api.hermes.example.json` to the address of your Hermes API.
Export the same `API_SERVER_KEY` in the Scanwich environment, then start the API:

```console
scanwich-api --config api.hermes.example.json
```

A generated invoice image passed live Luna tests for `text`, `pdf`, and `pdf+text` through Hermes.
The checks confirmed the invoice number and amount in plain text and searchable PDF text.
These checks used no private documents.

## PDF notes

- By default, Scanwich infers each page's DPI from a full-page image. It falls back to 300 DPI
  for vector or ambiguous pages and caps inferred values at 300 DPI.
- `--dpi` overrides automatic DPI selection for every page. Higher values can improve OCR while
  increasing memory use.
- `--ocr-output-dir` saves one normalized JSON file per page.
- Output pages retain the source PDF's page dimensions.
- The output is written atomically after every page has been processed successfully.

## License

Scanwich is licensed under the [MIT License](LICENSE). Dependencies keep their own
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
