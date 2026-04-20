{
  description = "startup_sim development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        inferencePython = (pkgs.python313.override {
          packageOverrides = self: super: {
            nflows = self.buildPythonPackage rec {
              pname = "nflows";
              version = "0.14";
              pyproject = true;
              src = pkgs.fetchPypi {
                inherit pname version;
                hash = "sha256-YpmESmL5mZ/N8tlcstAcCRpQE2vReCbjA6umRrLRG1U=";
              };
              build-system = with self; [ setuptools ];
              propagatedBuildInputs = with self; [
                matplotlib
                numpy
                tensorboard
                torch
                tqdm
              ];
              doCheck = false;
              pythonImportsCheck = [ "nflows" ];
            };

            pyknos = self.buildPythonPackage rec {
              pname = "pyknos";
              version = "0.15.1";
              format = "wheel";
              dontBuild = true;
              src = pkgs.fetchPypi {
                inherit pname version format;
                python = "py2.py3";
                abi = "none";
                platform = "any";
                dist = "py2.py3";
                hash = "sha256-1J4XNphlgXY6yjWOkEHRg0UVB9ei29k8jFZrp0mPYA8=";
              };
              propagatedBuildInputs = with self; [
                matplotlib
                nflows
                numpy
                tensorboard
                torch
                tqdm
              ];
              doCheck = false;
              pythonImportsCheck = [ "pyknos" ];
            };

            arviz = self.buildPythonPackage rec {
              pname = "arviz";
              version = "0.18.0";
              format = "wheel";
              dontBuild = true;
              dontCheckRuntimeDeps = true;
              nativeBuildInputs = with self; [ pythonRelaxDepsHook ];
              pythonRelaxDeps = [ "numpy" ];
              src = pkgs.fetchPypi {
                inherit pname version format;
                python = "py3";
                abi = "none";
                platform = "any";
                dist = "py3";
                hash = "sha256-bqqv//T7kO1Jv1MFwXHlxoSLKxjMXbFTcxnY+2fE6PU=";
              };
              propagatedBuildInputs = with self; [
                dm-tree
                h5netcdf
                matplotlib
                numpy
                packaging
                pandas
                scipy
                setuptools
                typing-extensions
                xarray
                xarray-einstats
              ];
              doCheck = false;
              pythonImportsCheck = [ "arviz" ];
            };

            numpyro = self.buildPythonPackage rec {
              pname = "numpyro";
              version = "0.19.0";
              format = "wheel";
              dontBuild = true;
              src = pkgs.fetchPypi {
                inherit pname version format;
                python = "py3";
                abi = "none";
                platform = "any";
                dist = "py3";
                hash = "sha256-EGOiwTGgeFcZ4TyOVfG4LkGFDYFN8UlBgJdTH0297ag=";
              };
              propagatedBuildInputs = with self; [
                jax
                jaxlib
                multipledispatch
                numpy
                tqdm
              ];
              doCheck = false;
              pythonImportsCheck = [ "numpyro" ];
            };

            sbi = self.buildPythonPackage rec {
              pname = "sbi";
              version = "0.21.0";
              format = "wheel";
              dontBuild = true;
              src = pkgs.fetchPypi {
                inherit pname version format;
                python = "py2.py3";
                abi = "none";
                platform = "any";
                dist = "py2.py3";
                hash = "sha256-Kqlk1/gcUH5UzPVp5XYT+YNkDXX923pJoci6KHzPnkY=";
              };
              propagatedBuildInputs = with self; [
                arviz
                joblib
                matplotlib
                numpy
                pillow
                pyknos
                pyro-ppl
                scikit-learn
                scipy
                tensorboard
                torch
                tqdm
              ];
              doCheck = false;
              pythonImportsCheck = [ "sbi" "sbi.inference" ];
            };
          };
        }).withPackages (ps: with ps; [
          numpy
          scipy
          matplotlib
          plotly
          dash
          pytest
          jax
          jaxlib
          numpyro
          torch
          tqdm
          sbi
        ]);
      in
      {
        packages.default = inferencePython;
        devShells.default = pkgs.mkShell {
          packages = [ inferencePython ];
          shellHook = ''
            export MPLCONFIGDIR="$PWD/.cache/matplotlib"
            export JAX_PLATFORMS=cpu
            mkdir -p "$MPLCONFIGDIR"
          '';
        };
      });
}
