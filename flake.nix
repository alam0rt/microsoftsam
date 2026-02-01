{
  description = "Mumble TTS Bot - A Mumble bot using LuxTTS voice cloning";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    hf-nix = {
      url = "github:huggingface/hf-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, hf-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ hf-nix.overlays.default ];
        };
        
        # Python packages from hf-nix overlay (torch, transformers, etc.)
        py = pkgs.python311Packages;
        
        # Custom packages
        opuslib-next = py.buildPythonPackage rec {
          pname = "opuslib_next";
          version = "1.1.5";
          pyproject = true;
          
          src = py.fetchPypi {
            inherit pname version;
            sha256 = "sha256-auwQFfJfeZeU0hdgHHTqD66P1l11JXjLFj/SM47Qdc4=";
          };
          
          patches = [
            (pkgs.replaceVars ./nix/opuslib-paths.patch {
              opusLibPath = "${pkgs.libopus}/lib/libopus${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}";
            })
          ];
          
          build-system = [ py.poetry-core ];
          dependencies = [ pkgs.libopus ];
        };
        
        # pymumble from botamusique (includes the library)
        pymumble-py3 = py.buildPythonPackage rec {
          pname = "pymumble-py3";
          version = "2.0.0";
          pyproject = false;
          
          src = pkgs.fetchFromGitHub {
            owner = "algielen";
            repo = "botamusique";
            rev = "v8.4.0";
            sha256 = "sha256-uPyVq1IdHCIjD2aU5esznRnsR6t7jF5gDOqQgz/GeSM=";
          };
          
          # Only install the pymumble_py3 directory
          buildPhase = "true";
          installPhase = ''
            mkdir -p $out/${pkgs.python311.sitePackages}
            cp -r pymumble_py3 $out/${pkgs.python311.sitePackages}/
          '';
          
          propagatedBuildInputs = [
            opuslib-next
            py.protobuf
          ];
        };
        
        # linacodec - vocos vocoder used by LuxTTS
        linacodec = py.buildPythonPackage rec {
          pname = "linacodec";
          version = "0.1.0";
          pyproject = true;
          
          src = pkgs.fetchFromGitHub {
            owner = "ysharma3501";
            repo = "linacodec";
            rev = "main";
            hash = pkgs.lib.fakeHash;
          };
          
          build-system = [ py.setuptools py.wheel ];
          
          propagatedBuildInputs = [
            py.torch  # From hf-nix overlay
            py.torchaudio
            py.numpy
          ];
          
          doCheck = false;
        };
        
        # LuxTTS - vendored from GitHub
        luxtts = py.buildPythonPackage rec {
          pname = "luxtts";
          version = "0.1.0";
          pyproject = false;
          
          src = pkgs.fetchFromGitHub {
            owner = "ysharma3501";
            repo = "LuxTTS";
            rev = "main";
            hash = pkgs.lib.fakeHash;
          };
          
          # LuxTTS doesn't have a proper setup.py, we install zipvoice directly
          buildPhase = "true";
          installPhase = ''
            mkdir -p $out/${pkgs.python311.sitePackages}
            cp -r zipvoice $out/${pkgs.python311.sitePackages}/
          '';
          
          propagatedBuildInputs = [
            # Core ML deps from hf-nix
            py.torch  # From hf-nix overlay (pre-built binary)
            py.torchaudio
            py.transformers  # From hf-nix overlay
            
            # Audio processing
            py.librosa
            py.soundfile
            py.pydub
            
            # ML utilities
            py.numpy
            py.safetensors
            py.huggingface-hub
            py.lhotse
            py.onnxruntime
            
            # linacodec for vocos vocoder
            linacodec
          ];
          
          doCheck = false;
        };
        
        # Mumble TTS Bot - packaged as a proper Python application
        mumble-tts-bot = py.buildPythonApplication {
          pname = "mumble-tts-bot";
          version = "0.1.0";
          pyproject = true;
          
          src = ./.;
          
          build-system = [
            py.setuptools
            py.wheel
          ];
          
          propagatedBuildInputs = [
            pymumble-py3
            opuslib-next
            luxtts
            py.numpy
            py.protobuf
            py.soundfile
          ];
          
          # LuxTTS is not yet nixified, so we skip tests that require it
          doCheck = false;
          
          # Ensure the libopus library is available at runtime
          makeWrapperArgs = [
            "--prefix" "LD_LIBRARY_PATH" ":" (pkgs.lib.makeLibraryPath [
              pkgs.libopus
              pkgs.libsndfile
            ])
          ];
          
          meta = with pkgs.lib; {
            description = "A Mumble bot that reads text messages aloud using LuxTTS voice cloning";
            homepage = "https://github.com/alam0rt/microsoftsam";
            license = licenses.mit;
            maintainers = [];
            mainProgram = "mumble-tts-bot";
          };
        };

        # Python environment for development
        devPython = pkgs.python311.withPackages (ps: [
          pymumble-py3
          opuslib-next
          luxtts
          ps.numpy
          ps.protobuf
          ps.soundfile
          ps.torch  # From hf-nix
          ps.torchaudio
          ps.transformers  # From hf-nix
          ps.librosa
          ps.huggingface-hub
          ps.lhotse
          ps.safetensors
          ps.onnxruntime
          ps.pydub
        ]);

        libPath = pkgs.lib.makeLibraryPath [
          pkgs.portaudio
          pkgs.libsndfile
          pkgs.libopus
          pkgs.stdenv.cc.cc.lib
        ];
      in
      {
        # Expose the mumble-tts-bot package
        packages = {
          inherit mumble-tts-bot luxtts;
          default = mumble-tts-bot;
        };

        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = [
            devPython
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
            echo "All packages are now provided via Nix (including LuxTTS):"
            echo "  - pymumble_py3 (from botamusique)"
            echo "  - opuslib-next"
            echo "  - luxtts (vendored)"
            echo "  - torch (from hf-nix, pre-built binary)"
            echo "  - transformers (from hf-nix)"
            echo "  - numpy, protobuf, soundfile, librosa, etc."
            echo ""
            echo "Usage:"
            echo "  python mumble_tts_bot.py --host localhost --user TTSBot --reference reference.wav"
            echo ""
            echo "Build package:"
            echo "  nix build"
            echo ""
            echo "Note: You need a reference.wav file (3+ seconds) for voice cloning."
          '';
          
          LD_LIBRARY_PATH = libPath;
        };
      }
    );
}
