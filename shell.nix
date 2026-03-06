{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    cmake
    pkg-config
    
    # Graphics and Windowing dependencies
    libGL
    xorg.libX11
    xorg.libXi
    xorg.libXcursor
    xorg.libXrandr
    
    # Modern Sokol can also utilize Wayland
    wayland
    libxkbcommon

    glm
    gcc
  ];

  # Sokol uses dlopen() to load graphics drivers at runtime.
  # We expose them to the environment here.
  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
      libGL
      xorg.libX11
      xorg.libXi
      xorg.libXcursor
      xorg.libXrandr
      wayland
      libxkbcommon
    ])}:$LD_LIBRARY_PATH"
    
    echo "Sokol Dev Environment loaded!"
  '';
}