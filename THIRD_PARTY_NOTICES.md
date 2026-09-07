# Third-party notices

Scanwich's source code is licensed under the MIT License. Its dependencies retain their
own licenses and are not relicensed by Scanwich.

The standard Nix package and container include these direct dependencies:

| Component | License |
| --- | --- |
| EasyOCR | Apache-2.0 |
| OpenAI Python SDK | Apache-2.0 |
| Pillow | MIT-CMU |
| pypdfium2 | Apache-2.0 OR BSD-3-Clause |
| PDFium | Apache-2.0 AND BSD-3-Clause AND MIT |
| ReportLab | BSD-3-Clause |

The `scanwich-openai-compatible` Nix package and `openai-compatible` container target
exclude EasyOCR and its dependencies.
They include the other direct dependencies in the table.

The test-only pypdf dependency is BSD-3-Clause. Transitive dependencies retain their
respective licenses as recorded by the pinned Nixpkgs revision in `flake.lock`.
