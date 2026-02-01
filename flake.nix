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
          config.cudaSupport = true;
        };

        # CUDA packages for GPU support (optional - only on systems with NVIDIA GPUs)
        cudaLibs = pkgs.lib.optionals (pkgs.stdenv.isLinux) [
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.cudnn
        ];

        libPath = pkgs.lib.makeLibraryPath ([
          pkgs.portaudio
          pkgs.libsndfile
          pkgs.libopus
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ] ++ cudaLibs);
        
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
            pkgs.python312
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
            echo ""
            echo "Qwen3-TTS (requires CUDA GPU):"
            echo "  uv run scripts/qwen3_tts_demo.py 0.0.0.0:9999           # Web UI with voice cloning"
            echo "  uv run scripts/qwen3_tts_demo.py --task CustomVoice    # Predefined speakers"
            echo "  uv run scripts/qwen3_tts_demo.py --task VoiceDesign    # Voice design from description"
          '';
          
          LD_LIBRARY_PATH = libPath;
        };
        
        # Shell with vLLM from nixpkgs (for serving Qwen3-TTS via API)
        devShells.vllm = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.uv
            pkgs.vllm
            pkgs.git
            pkgs.openssl
            pkgs.curl
          ] ++ cudaLibs;

          shellHook = ''
            echo "Qwen3-TTS vLLM Environment"
            echo "=========================="
            echo ""
            echo "vLLM ${pkgs.vllm.version} is available from nixpkgs."
            echo ""
            echo "For Qwen3-TTS support, you need vllm-omni on top:"
            echo "  uv pip install vllm-omni"
            echo ""
            echo "Then run:"
            echo "  uv run scripts/qwen3_tts_vllm.py 0.0.0.0:9999"
            echo ""
            echo "Or use the qwen-tts demo with built-in web UI:"
            echo "  uv run scripts/qwen3_tts_demo.py 0.0.0.0:9999"
          '';
          
          LD_LIBRARY_PATH = libPath;
        };
      }
    );
}
