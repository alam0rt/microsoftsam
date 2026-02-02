{
  description = "Mumble TTS Bot - A Mumble bot using LuxTTS voice cloning with Wyoming protocol support";

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

        # CUDA packages for GPU-accelerated TTS (optional - only on systems with NVIDIA GPUs)
        cudaLibs = pkgs.lib.optionals (pkgs.stdenv.isLinux) [
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.cudnn
        ];

        # Core native libraries required:
        # - libopus: Mumble audio codec
        # - libsndfile: Audio file loading (reference.wav for voice cloning)
        # - portaudio: Audio I/O
        # - zlib: Compression
        libPath = pkgs.lib.makeLibraryPath ([
          pkgs.libopus
          pkgs.libsndfile
          pkgs.portaudio
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ] ++ cudaLibs);
        
        # Wrapper script that sets up environment and runs the bot
        mumble-tts-bot = pkgs.writeShellScriptBin "mumble-tts-bot" ''
          export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
          export PATH="${pkgs.lib.makeBinPath [pkgs.ffmpeg pkgs.espeak-ng pkgs.git]}:$PATH"
          
          if [ ! -f "mumble_tts_bot.py" ]; then
            echo "Error: mumble_tts_bot.py not found in current directory."
            echo ""
            echo "This bot must be run from a cloned repository:"
            echo "  git clone https://github.com/alam0rt/microsoftsam.git"
            echo "  cd microsoftsam"
            echo "  nix develop -c uv run python mumble_tts_bot.py --help"
            exit 1
          fi
          
          exec ${pkgs.uv}/bin/uv run python mumble_tts_bot.py "$@"
        '';
      in
      {
        packages.default = mumble-tts-bot;
        
        apps.default = {
          type = "app";
          program = "${mumble-tts-bot}/bin/mumble-tts-bot";
        };
        
        # Development shell with system dependencies
        # Python packages managed via uv (see pyproject.toml)
        # External services (run separately):
        #   - Wyoming STT: wyoming-faster-whisper (localhost:10300)
        #   - LLM: vLLM or Ollama (localhost:8000 or 11434)
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Python & package management
            pkgs.python311
            pkgs.uv
            
            # Audio libraries (required for Mumble + TTS)
            pkgs.libopus      # Mumble codec
            pkgs.libsndfile   # Audio file I/O
            pkgs.portaudio    # Audio device I/O
            pkgs.ffmpeg       # Audio processing
            
            # TTS dependencies
            pkgs.espeak-ng    # Phonemization for LuxTTS
            
            # Build tools
            pkgs.git
            pkgs.openssl
          ];

          shellHook = ''
            echo "Mumble TTS Bot Development Environment"
            echo "======================================="
            echo ""
            echo "External services required:"
            echo "  - Wyoming STT: wyoming-faster-whisper on localhost:10300"
            echo "  - LLM: vLLM/Ollama on localhost:8000 (OpenAI-compatible API)"
            echo ""
            echo "Quick start:"
            echo "  uv sync"
            echo "  uv run python mumble_tts_bot.py \\"
            echo "    --host murmur.example.com \\"
            echo "    --reference reference.wav \\"
            echo "    --wyoming-stt-host localhost \\"
            echo "    --llm-endpoint http://localhost:8000/v1/chat/completions"
            echo ""
          '';
          
          LD_LIBRARY_PATH = libPath;
        };
      }
    );
}
