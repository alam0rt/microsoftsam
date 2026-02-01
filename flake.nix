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
        
        # Wrapper script that sets up environment and runs the bot
        # Note: Must be run from a cloned repo directory (uv needs writable .venv)
        mumble-tts-bot = pkgs.writeShellScriptBin "mumble-tts-bot" ''
          export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
          export PATH="${pkgs.lib.makeBinPath [pkgs.ffmpeg pkgs.espeak-ng pkgs.git]}:$PATH"
          
          # Check if we're in a valid project directory
          if [ ! -f "mumble_tts_bot.py" ]; then
            echo "Error: mumble_tts_bot.py not found in current directory."
            echo ""
            echo "This bot must be run from a cloned repository:"
            echo "  git clone https://github.com/alam0rt/microsoftsam.git"
            echo "  cd microsoftsam"
            echo "  nix develop -c uv run python mumble_tts_bot.py --help"
            echo ""
            echo "Or use the dev shell:"
            echo "  nix develop"
            echo "  uv run python mumble_tts_bot.py --help"
            exit 1
          fi
          
          exec ${pkgs.uv}/bin/uv run python mumble_tts_bot.py "$@"
        '';
      in
      {
        # Runnable package (must be run from cloned repo)
        packages.default = mumble-tts-bot;
        
        # App for `nix run` (must be run from cloned repo)
        apps.default = {
          type = "app";
          program = "${mumble-tts-bot}/bin/mumble-tts-bot";
        };
        
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
