FROM docker.io/nixos/nix:2.35.2 AS base

ENV NIX_CONFIG="experimental-features = nix-command flakes"

WORKDIR /workspace
COPY . .

FROM base AS openai-compatible

RUN nix build "path:/workspace#scanwich-openai-compatible" --out-link /opt/scanwich \
    && nix-store --gc

ENTRYPOINT ["/opt/scanwich/bin/scanwich", "--ocr-backend", "openai-compatible"]
CMD ["--help"]

FROM base AS easyocr

ENV EASYOCR_MODULE_PATH="/opt/easyocr"

RUN mkdir -p "${EASYOCR_MODULE_PATH}" \
    && nix build "path:/workspace#scanwich" --out-link /opt/scanwich \
    && nix develop "path:/workspace" -c python docker/warm_models.py \
    && nix-store --gc

ENTRYPOINT ["/opt/scanwich/bin/scanwich"]
CMD ["--help"]
