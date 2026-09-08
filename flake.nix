{
  description = "Create searchable PDFs with pluggable OCR backends";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python3;
          makePackage =
            withEasyocr:
            python.pkgs.buildPythonApplication {
              pname = "scanwich";
              version = "0.1.0";
              pyproject = true;
              src = self;

              build-system = [ python.pkgs.setuptools ];
              dependencies =
                with python.pkgs;
                [
                  openai
                  fastapi
                  pydantic
                  uvicorn
                  python-multipart
                  pillow
                  pypdfium2
                  reportlab
                ]
                ++ pkgs.lib.optionals withEasyocr [ python.pkgs.easyocr ];

              postPatch = pkgs.lib.optionalString (!withEasyocr) ''
                substituteInPlace pyproject.toml \
                  --replace-fail 'easyocr = "scanwich.backends.easyocr:factory"' ""
              '';
              nativeCheckInputs = [
                python.pkgs.pypdf
                python.pkgs.httpx
              ];

              checkPhase = ''
                runHook preCheck
                python -m unittest discover -s tests -v
                ${pkgs.lib.optionalString (!withEasyocr) ''
                  python - <<'PYTHON'
                  from importlib.util import find_spec

                  for name in ("easyocr", "torch", "torchvision", "cv2", "scipy", "skimage"):
                      assert find_spec(name) is None, f"unexpected dependency: {name}"
                  PYTHON
                  test "$("$out/bin/scanwich" --list-backends)" = "openai-compatible"
                ''}
                runHook postCheck
              '';
              pythonImportsCheck = [ "scanwich" ];
            };
        in
        {
          default = makePackage true;
          scanwich = makePackage true;
          scanwich-openai-compatible = makePackage false;
        }
      );

      apps = forAllSystems (
        system:
        let
          package = self.packages.${system}.default;
        in
        {
          default = {
            type = "app";
            program = "${package}/bin/scanwich";
          };
        }
      );

      checks = forAllSystems (system: {
        default = self.packages.${system}.default;
        openai-compatible = self.packages.${system}.scanwich-openai-compatible;
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
            packages = [
              self.packages.${system}.default
              pkgs.python3Packages.pypdf
              pkgs.python3Packages.ruff
            ];
          };
        }
      );
    };
}
