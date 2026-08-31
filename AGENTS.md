# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `src/scanwich/`. `cli.py` handles arguments and logging,
`pipeline.py` coordinates rasterization, OCR, and PDF assembly, and `pdf.py`, `ocr.py`, and
`models.py` contain focused domain logic. OCR implementations belong in
`src/scanwich/backends/`; expose new providers through the `scanwich.ocr_backends` entry-point
group in `pyproject.toml`. Tests live in `tests/` and follow the source behavior they cover.
Container model warm-up code is in `docker/`, while Nix packaging is defined by `flake.nix` and
`flake.lock`.

## Build, Test, and Development Commands

- `direnv allow` activates the Nix development shell through `.envrc` on directory entry.
- `nix develop "path:$PWD"` opens the same shell manually with Python, ImageMagick,
  Ghostscript, Ruff, and test dependencies.
- `nix run . -- input.pdf output.pdf -l de pt en` builds and runs Scanwich locally.
- `nix flake check "path:$PWD"` builds the package and runs the complete unit test suite.
- `nix develop "path:$PWD" -c ruff check src tests docker` checks Python style.
- `podman build --tag localhost/scanwich:dev .` builds the release-like container image.

## Coding Style & Naming Conventions

Use Python 3.11 or newer, four-space indentation, type hints for public interfaces, and Ruff's
100-character line limit. Name modules and functions with `snake_case`, classes with
`PascalCase`, and constants with `UPPER_SNAKE_CASE`. Keep OCR-specific behavior behind the
backend protocol; pipeline and PDF code must remain provider-neutral. Send operational progress
through Python logging (stderr), not stdout or generated OCR results.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name files `test_*.py`, test cases
`Test...`, and methods `test_*`. Run focused tests with, for example,
`python -m unittest tests.test_pipeline -v`. Add regression tests for bug fixes and cover plugin
option parsing, lazy backend initialization, coordinate conversion, and PDF text extraction when
those areas change. There is no numeric coverage threshold; new behavior should have direct tests.

## Commit & Pull Request Guidelines

Before starting work, fetch `origin` and ensure the current branch includes `origin/main`.
Recent history uses short, imperative, sentence-case subjects such as `Use Node 24 container
actions`; keep commits focused the same way. Pull requests should explain the user-visible change,
list validation commands and results, and link relevant issues. For OCR or PDF changes, describe
the sample scenario without committing private source documents or generated outputs. Keep
dependency, container, `THIRD_PARTY_NOTICES.md`, and bundled source-license changes synchronized.
