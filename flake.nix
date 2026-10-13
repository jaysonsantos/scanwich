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
      # The package and the development shell use the same dependency lists.
      runtimeDependencies =
        pythonPackages: withEasyocr:
        with pythonPackages;
        [
          openai
          fastapi
          pydantic
          pydantic-settings
          uvicorn
          python-multipart
          pillow
          pypdfium2
          reportlab
        ]
        ++ nixpkgs.lib.optionals withEasyocr [ pythonPackages.easyocr ];
      testDependencies = pythonPackages: [
        pythonPackages.pypdf
        pythonPackages.httpx
      ];
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
              dependencies = runtimeDependencies python.pkgs withEasyocr;

              postPatch = pkgs.lib.optionalString (!withEasyocr) ''
                substituteInPlace pyproject.toml \
                  --replace-fail 'easyocr = "scanwich.backends.easyocr:factory"' ""
              '';
              nativeCheckInputs = testDependencies python.pkgs;

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
          python = pkgs.python3;
          project = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project;
          # An editable install: Python imports `src/` of the working tree, and the commands
          # and the backend entry points are available. The shell does not build the package,
          # so a test that fails does not prevent the shell. `nix flake check` runs the tests.
          editablePackage = python.pkgs.mkPythonEditablePackage {
            pname = project.name;
            inherit (project) version scripts entry-points;
            # Python expands the variable when it starts. The shell hook sets it.
            root = "$REPO_ROOT/src";
          };
          pythonEnvironment = python.withPackages (
            pythonPackages:
            [ editablePackage ] ++ runtimeDependencies pythonPackages true ++ testDependencies pythonPackages
          );
        in
        {
          default = pkgs.mkShell {
            packages = [
              pythonEnvironment
              python.pkgs.ruff
            ];
            shellHook = ''
              export REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
            '';
          };
        }
      );
    };
}
