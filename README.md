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

Scanwich can write a PDF or a photo of a page as a LaTeX document.
The purpose is an archive copy of an old book.
The document has the text and the structure of the pages. It is not an image of the pages.
Give the output file a name that ends in `.tex`:

```console
export OPENROUTER_API_KEY=...
scanwich book.pdf book.tex --ocr-backend openai-compatible -l de
scanwich page.ARW page.tex --ocr-backend openai-compatible -l de --rotate 90
```

The input is a PDF, an image that Pillow can read, or a camera raw file.
A PDF with many pages becomes one document. An image is one page.
For a camera raw file, install the decoder with `pip install "scanwich[raw]"`.
LibRaw then decodes all the sensor data. Scanwich does not use the small preview image of the file.
Both Nix packages and both image targets include the decoder.

The document has these parts of the source, in reading order:

- headings, paragraphs, block quotations, and lists;
- bold, italic, underlined, small-capital, superscript, and subscript text;
- tables as `longtable` environments, with cells that span columns or rows;
- illustrations as image files, which the document includes with `\includegraphics`;
- footnotes as LaTeX footnotes.

Scanwich crops each illustration from the page image.
The illustrations of `book.tex` are JPEG files of quality 95 in the directory `book-figures` next to the document.
Their names are `page-0001-figure-01.jpg`, `page-0001-figure-02.jpg`, and so on.
Keep that directory with the document.
LaTeX reads the name of the directory, so the name has only ASCII letters, digits, and the characters `.`, `_`, and `-`.
For a document name with other characters, a hyphen replaces them, and the name gets a part of a hash of the document name.
For example, `my book.tex` has the directory `my-book-2c7e412c-figures`. Thus two documents in one directory do not share their illustrations.
A new conversion replaces the files of the earlier conversion that have a name of the form `page-NNNN-figure-NN.jpg`. It does not remove a file with a different name.

Scanwich writes the document after all pages are complete. A conversion that fails does not change an earlier document.
A conversion fails when a page has no usable layout, and when no page has content. The error message has the number of the page.
A page with no content in a document that has content gives a `WARNING` log record.

Scanwich does not start a TeX engine. Compile the document with LuaLaTeX, XeLaTeX, or Tectonic:

```console
latexmk -lualatex book.tex
tectonic book.tex
```

These engines read all Unicode text.
pdfLaTeX also compiles the document when the text has only characters that pdfLaTeX knows.
The document uses the packages `iftex`, `fontspec`, `graphicx`, `array`, and `longtable`.
It uses `multirow` only when that package is installed.

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

`--ocr-output-dir` saves the JSON of each page as `page-000001.json`.
The module `scanwich.layout` defines the JSON.
One page has `blocks` in reading order. The block types are `heading`, `paragraph`, `quote`, `list`, `table`, `figure`, `footnote`, `page_header`, `page_footer`, and `page_number`.
Text can have the tags `<b>`, `<i>`, `<u>`, `<sc>`, `<sup>`, `<sub>`, `<br>`, and `<fn>mark</fn>`.
A box is `{"left": 120, "top": 300, "right": 880, "bottom": 560}` on a 0-to-1000 grid of the page image.
The names of the sides leave no doubt about their sequence. The check also accepts the array `[left, top, right, bottom]`.

Text of a reply does not disappear without an error. The check keeps the text, or it rejects the reply:

- It accepts a value that has one clear meaning. Examples are an integer where the page prints digits, `"continued": "true"`, header rows in a `header` field, and `"align": "middle"`.
- It rejects a reply that has text in a field that the layout does not read. Examples are a list in a list item, and a table with a caption and no rows.
- It rejects text that it cannot keep in a field that the layout reads. Examples are the heads of the columns in the `columns` field of a table, and a value of `align` or `rules` that is not one of the names.
  A figure of a reply has no `image` field, because Scanwich makes the image file from the box.
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

### Page direction

A photo can show a page that is not upright. A model reads such a page badly.
Without `--rotate`, the backend first finds the side of the image where the top of the page is.
The model does not read a page that is not upright. Scanwich turns the page image and sends a second request for that page.
When the backend reports that the page is not upright after that turn, the conversion fails.

`--rotate DEGREES` gives the direction of all pages: Scanwich turns each page image clockwise by 0, 90, 180, or 270 degrees before recognition.
The backend then gets no request to find the direction, and each page needs one request.
Use it when all pages have the same direction. `--rotate 0` is for pages that are upright.
Scanwich uses the blocks of the backend also when the backend reports a different direction, and writes a `WARNING` log record.

### Differences from the source

Compare the document with the source. A model can read a character incorrectly.
These differences are intended:

- A running head and a printed page number are not text of the document. They are in a comment line at the start of each page, for example `% Source page 3 (printed page number: 47)`.
- A paragraph that continues on the next page becomes one paragraph.
  When the page break divides a word with a hyphen, Scanwich joins the two parts and adds a comment line. Compare that word with the source.
- LaTeX gives new numbers to the footnotes, because the pages of the document are not the pages of the source.
  The footnotes of a table and of a caption keep the marks of the source and come after the block.
- A table has the rules of one of four styles: no rules, horizontal rules, also rules between the columns, or a rule around each cell. A double rule becomes one rule.
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
