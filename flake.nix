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
        python = pkgs.python313.withPackages (ps: with ps; [
          numpy
          scipy
          matplotlib
          plotly
          dash
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ python ];
        };
      });
}
