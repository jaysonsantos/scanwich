# Scanwich

Scanwich turns an image-based PDF into a searchable sandwich PDF. PDFium renders each source
page, an OCR backend returns text polygons, and the tool rebuilds each page with:

- the original rendered page as the visible layer; and
- invisible, selectable text aligned with the OCR polygons.

EasyOCR is the default backend for the CLI and the standard image.
For pip installations, use `pip install "scanwich[easyocr]"`.
The PDF pipeline itself does not import or depend on
EasyOCR-specific result types.

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

Enter the development shell with `nix develop`. Inside it, the packaged `scanwich` command
is available, or run the default command directly with `nix run .`. With direnv installed,
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

The backend accepts `model`, `base_url`, `api_key_env`, `timeout`, `max_tokens`, and
`reasoning_effort` backend options. Set `base_url` and `api_key_env` to use another
OpenAI-compatible service:

```console
scanwich input.pdf output.pdf \
  --ocr-backend openai-compatible \
  --backend-option base_url=https://example.invalid/v1 \
  --backend-option api_key_env=EXAMPLE_API_KEY \
  --backend-option model=provider/vision-model
```

Page images are base64-encoded and sent to the configured service and its selected model
provider. Do not use this backend for documents that must remain local.

## Notes

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
