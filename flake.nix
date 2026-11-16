{
  description = "jax-in-lean: JAX semantics over the reals, modelled and proven in Lean 4";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            # elan reads `lean-toolchain`, so Lean always matches the pinned Mathlib.
            pkgs.elan
            pkgs.uv
            pkgs.python315
          ];
          env = {
            # uv manages the packages, Nix provides the interpreter.
            UV_PYTHON = "${pkgs.python315}/bin/python3";
            UV_PYTHON_DOWNLOADS = "never";
          };
        };
      });
    };
}
