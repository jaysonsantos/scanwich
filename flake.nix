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
          package = python.pkgs.buildPythonApplication {
            pname = "scanwich";
            version = "0.1.0";
            pyproject = true;
            src = self;

            build-system = [ python.pkgs.setuptools ];
            dependencies = with python.pkgs; [
              easyocr
              pillow
              reportlab
            ];
            nativeCheckInputs = [
              python.pkgs.pypdf
              pkgs.imagemagick
              pkgs.ghostscript
            ];

            makeWrapperArgs = [
              "--prefix PATH : ${nixpkgs.lib.makeBinPath [ pkgs.imagemagick pkgs.ghostscript ]}"
            ];

            checkPhase = ''
              runHook preCheck
              python -m unittest discover -s tests -v
              runHook postCheck
            '';
            pythonImportsCheck = [ "scanwich" ];
          };
        in
        {
          default = package;
          ghostscript-source = pkgs.ghostscript.src;
          scanwich = package;
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
              pkgs.imagemagick
              pkgs.ghostscript
              pkgs.python3Packages.pypdf
              pkgs.python3Packages.ruff
            ];
          };
        }
      );
    };
}
