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

## HTTP API

Install the API and a backend:

```console
pip install '.[api,openai-compatible]'
scanwich-api --config api.example.json
```

The server listens on `127.0.0.1:8000`. Open `/docs` for the request schema.
`GET /backends` lists installed backends that the configuration enables.
`POST /ocr/{backend_name}` accepts an image in the `image` multipart field.
Set the typed `output` multipart field to `pdf`, `text`, or `pdf+text`. The default is `pdf`.
The JSON response uses `output` to identify its Pydantic response model:

| Output | Response fields |
| --- | --- |
| `pdf` | `output`, `pdf_base64` |
| `text` | `output`, `text` |
| `pdf+text` | `output`, `pdf_base64`, `text` |

Decode `pdf_base64` to obtain a searchable PDF with the source image and invisible text.
The API uses 300 DPI for the image's PDF page dimensions.
Text output preserves the provider's line breaks and spacing.
The OpenAI-compatible backend requests plain text without JSON or coordinates for text output.
Combined output makes two provider calls: one for PDF polygons and one for plain text.
Other plugins provide text from their regions, with one line per region.
Invalid output choices return 422.
The API processes one image per request.

```console
curl http://127.0.0.1:8000/ocr/openai-compatible \
  -F image=@page.png -F model=deepseek -F output=text -F languages=en -F languages=pt
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
Clients can select a model. Server configuration controls endpoint URLs and credential environment variables.

The OpenAI-compatible backend exposes `await backend.recognize_async(path)` and `await backend.aclose()`.
Use `await backend.recognize_text_async(path)` for plain text.
It shares request construction and response checks with `recognize(path)`.
The API uses native async calls when the backend provides them.
It runs synchronous plugins in a worker thread. Each request creates its own backend instance.
The API closes async clients and removes temporary images after each request.

The default upload limit is 20 MiB. Set `max_image_bytes` to change it.
Invalid images return 422; excessive image sizes return 413; backend failures return 502.
The server has no authentication. Keep the default local address or put an authenticated proxy before the server.
Set `SCANWICH_API_CONFIG` to a JSON file path for `uvicorn scanwich.api:create_app --factory`.
Pydantic Settings loads and checks the configuration.
Explicit arguments take priority over environment variables, then JSON file values, then defaults.
Use `SCANWICH_API_MAX_IMAGE_BYTES`, `SCANWICH_API_HOST`, and `SCANWICH_API_PORT` for environment overrides.
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
