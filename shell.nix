{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.git
    pkgs.makeWrapper
    pkgs.gcc14
    pkgs.pkg-config
    pkgs.cmake
    pkgs.xorg.libX11
    pkgs.libGL
    pkgs.libGLU
    pkgs.glew
    pkgs.clang_18
    pkgs.gdb
  ];
}
