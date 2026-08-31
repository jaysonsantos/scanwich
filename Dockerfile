FROM docker.io/nixos/nix:2.31.2

ENV NIX_CONFIG="experimental-features = nix-command flakes"
ENV EASYOCR_MODULE_PATH="/opt/easyocr"

WORKDIR /workspace
COPY . .

RUN mkdir -p "${EASYOCR_MODULE_PATH}" /opt/third-party-sources \
    && nix build "path:/workspace#scanwich" --out-link /opt/scanwich \
    && nix build "path:/workspace#ghostscript-source" \
        --out-link /opt/third-party-sources/ghostscript \
    && nix develop "path:/workspace" -c python docker/warm_models.py \
    && nix-store --gc

ENTRYPOINT ["/opt/scanwich/bin/scanwich"]
CMD ["--help"]
