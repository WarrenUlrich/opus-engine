{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    # Your existing dependencies
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
    
    # Dawn dependencies
    pkgs.python3
    pkgs.gn
    
    # Vulkan and GPU Drivers
    pkgs.vulkan-headers
    pkgs.vulkan-loader
    pkgs.mesa
    
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
    
    # Function to set up Dawn properly
    setup_dawn() {
      echo "Cloning Dawn repository..."
      git clone https://dawn.googlesource.com/dawn
      cd dawn
      
      echo "Building Dawn with CMake..."
      cmake -S . -B out/Release \
        -DDAWN_FETCH_DEPENDENCIES=ON \
        -DDAWN_ENABLE_INSTALL=ON \
        -DCMAKE_BUILD_TYPE=Release
        
      cmake --build out/Release
      
      # Install to a local directory
      cmake --install out/Release --prefix ./install/Release
      
      echo "Dawn setup complete! You can link against it using:"
      echo "  export CMAKE_PREFIX_PATH=$PWD/install/Release"
      echo "  cmake -DCMAKE_BUILD_TYPE=Release ..."
    }
    
    echo "Run 'setup_dawn' to build Dawn with CMake"
  '';
}
