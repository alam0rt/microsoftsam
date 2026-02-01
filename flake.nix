{
  description = "Mumble TTS Bot - A Mumble bot using LuxTTS voice cloning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        libPath = pkgs.lib.makeLibraryPath [
          pkgs.portaudio
          pkgs.libsndfile
          pkgs.libopus
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ];
      in
      {
        # Development shell - provides system deps, use uv for Python
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python311
            pkgs.uv
            pkgs.portaudio
            pkgs.ffmpeg
            pkgs.libsndfile
            pkgs.libopus
            pkgs.git
            pkgs.openssl
            pkgs.espeak-ng
          ];

          shellHook = ''
            echo "Mumble TTS Bot Development Environment"
            echo "======================================="
            echo ""
            echo "System dependencies provided via Nix:"
            echo "  - libopus, libsndfile, ffmpeg, portaudio"
            echo ""
            echo "Python dependencies managed via uv:"
            echo "  uv sync        # Install dependencies"
            echo "  uv run python mumble_tts_bot.py --help"
            echo ""
            echo "Usage:"
            echo "  uv run python mumble_tts_bot.py --host localhost --user TTSBot --reference reference.wav"
            echo ""
            echo "Note: You need a reference.wav file (3+ seconds) for voice cloning."
          '';
          
          LD_LIBRARY_PATH = libPath;
        };
      }
    );
}
