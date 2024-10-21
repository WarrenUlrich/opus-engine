{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.git
    pkgs.ninja
    pkgs.pkg-config
    pkgs.makeWrapper
    pkgs.gcc14
    pkgs.cmake
    pkgs.libGL
    pkgs.libGLU
    pkgs.glew
    pkgs.gdb
    pkgs.clang_18
    pkgs.glm
    
    # Vulkan and GPU Drivers
    pkgs.vulkan-headers
    pkgs.vulkan-loader
    pkgs.mesa.drivers
   # pkgs.mesa.vulkan-drivers

    # X11 and GLFW
    pkgs.xorg.libX11
    pkgs.xorg.libXext
    pkgs.xorg.libXinerama
    pkgs.xorg.libXrandr
    pkgs.xorg.libXcursor
    pkgs.xorg.libXrender
    pkgs.xorg.libXi
    pkgs.xorg.libXdamage
    pkgs.xorg.libXcomposite
    pkgs.xorg.libXfixes
    pkgs.xorg.libXtst
    pkgs.xorg.libxcb
    pkgs.xorg.libXau
    pkgs.xorg.libXdmcp
    pkgs.glfw
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH
  '';
}
