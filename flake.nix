{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    (flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python313.withPackages (
          python-pkgs: with python-pkgs; [
            numpy
            notebook
            ipywidgets
            matplotlib
            scikit-learn
            pytest
            venvShellHook
            kagglehub
          ]
        );
        version = "3.13";
      in
      {
        devShells.default = pkgs.mkShell rec {
          venvDir = ".venv";

          buildInputs = [
            python
          ];
          shellHook = ''
            ln -sf ${python}/lib/python${version}/site-packages/* ${venvDir}/lib/python${version}/site-packages
            export PYTHONPATH=src
          '';        
        };
      }
    ));
}
