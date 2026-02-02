"""Wyoming protocol TTS server wrapping LuxTTS.

This module provides a Wyoming-compatible TTS server that wraps the
StreamingLuxTTS implementation, allowing it to be used by Home Assistant
and other Wyoming protocol clients.

Usage:
    python -m mumble_voice_bot.providers.wyoming_tts_server \\
        --port 10400 \\
        --reference voice.wav \\
        --device cuda
"""

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from typing import Optional

import numpy as np

from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.tts import Synthesize
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info, TtsProgram, TtsVoice, Attribution
from wyoming.event import Event

# Add parent paths for imports when running as script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_LOGGER = logging.getLogger(__name__)


class LuxTTSEventHandler(AsyncEventHandler):
    """Handle Wyoming TTS events using LuxTTS.
    
    This handler processes Wyoming protocol events and generates
    speech using the StreamingLuxTTS engine.
    """
    
    def __init__(
        self,
        tts,
        voice_prompt: dict,
        num_steps: int = 4,
        *args,
        **kwargs,
    ):
        """Initialize the event handler.
        
        Args:
            tts: StreamingLuxTTS instance.
            voice_prompt: Pre-encoded voice prompt dict.
            num_steps: TTS quality steps (more = better quality, slower).
        """
        super().__init__(*args, **kwargs)
        self.tts = tts
        self.voice_prompt = voice_prompt
        self.num_steps = num_steps
    
    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming events.
        
        Args:
            event: Wyoming protocol event.
            
        Returns:
            True if the event was handled.
        """
        if Describe.is_type(event.type):
            await self._handle_describe()
            return True
        
        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            await self._synthesize(synthesize.text)
            return True
        
        return False
    
    async def _handle_describe(self) -> None:
        """Respond to describe request with server capabilities."""
        await self.write_event(
            Info(
                tts=[
                    TtsProgram(
                        name="luxtts",
                        description="LuxTTS voice cloning TTS",
                        attribution=Attribution(
                            name="LuxTTS",
                            url="https://github.com/ysharma3501/LuxTTS",
                        ),
                        installed=True,
                        voices=[
                            TtsVoice(
                                name="cloned",
                                description="Voice cloned from reference audio",
                                languages=["en"],
                                installed=True,
                            )
                        ],
                    )
                ]
            ).event()
        )
    
    async def _synthesize(self, text: str) -> None:
        """Generate speech from text using LuxTTS.
        
        Args:
            text: Text to synthesize.
        """
        _LOGGER.info(f"Synthesizing: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Run TTS in executor to not block event loop
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            partial(self._generate_speech, text),
        )
        
        # LuxTTS outputs 48kHz 16-bit mono
        sample_rate = 48000
        sample_width = 2
        channels = 1
        
        # Send audio
        await self.write_event(
            AudioStart(
                rate=sample_rate,
                width=sample_width,
                channels=channels,
            ).event()
        )
        
        # Send in chunks (0.5 second each for responsive streaming)
        chunk_samples = (sample_rate * sample_width * channels) // 2
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            await self.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=sample_rate,
                    width=sample_width,
                    channels=channels,
                ).event()
            )
        
        await self.write_event(AudioStop().event())
        _LOGGER.debug(f"Sent {len(audio_data)} bytes of audio")
    
    def _generate_speech(self, text: str) -> bytes:
        """Generate speech synchronously (called in executor).
        
        Args:
            text: Text to synthesize.
            
        Returns:
            Raw PCM audio bytes (48kHz, 16-bit, mono).
        """
        # Use streaming for better responsiveness
        audio_chunks = []
        for chunk in self.tts.generate_speech_streaming(
            text,
            self.voice_prompt,
            num_steps=self.num_steps,
        ):
            audio_chunks.append(chunk.numpy().squeeze())
        
        # Concatenate chunks
        if not audio_chunks:
            return b""
        
        audio = np.concatenate(audio_chunks)
        
        # Clip to valid range and convert float32 [-1, 1] to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()


async def run_server(
    host: str,
    port: int,
    reference_audio: str,
    device: str = "cuda",
    num_steps: int = 4,
    voices_dir: str = "voices",
) -> None:
    """Run the Wyoming LuxTTS server.
    
    Args:
        host: Server bind address.
        port: Server port.
        reference_audio: Path to reference audio for voice cloning.
        device: PyTorch device (cuda/cpu/mps).
        num_steps: TTS quality steps.
        voices_dir: Directory for cached voice prompts.
    """
    # Import LuxTTS here to avoid loading at module import time
    from mumble_tts_bot import StreamingLuxTTS
    import torch
    
    _LOGGER.info(f"Loading LuxTTS model on {device}...")
    tts = StreamingLuxTTS('YatharthS/LuxTTS', device=device, threads=2)
    
    # Load or encode voice prompt
    os.makedirs(voices_dir, exist_ok=True)
    reference_name = os.path.splitext(os.path.basename(reference_audio))[0]
    saved_voice_path = os.path.join(voices_dir, f"{reference_name}.pt")
    
    if os.path.exists(saved_voice_path):
        _LOGGER.info(f"Loading cached voice: {saved_voice_path}")
        voice_prompt = torch.load(saved_voice_path, weights_only=False, map_location=device)
        # Ensure tensors are on correct device
        voice_prompt = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in voice_prompt.items()
        }
    else:
        _LOGGER.info(f"Encoding reference audio: {reference_audio}")
        voice_prompt = tts.encode_prompt(reference_audio, rms=0.01)
        torch.save(voice_prompt, saved_voice_path)
        _LOGGER.info(f"Cached voice as '{reference_name}'")
    
    _LOGGER.info(f"Starting Wyoming TTS server on {host}:{port}")
    
    server = AsyncServer.from_uri(f"tcp://{host}:{port}")
    
    await server.run(
        partial(
            LuxTTSEventHandler,
            tts=tts,
            voice_prompt=voice_prompt,
            num_steps=num_steps,
        )
    )


def main() -> None:
    """Entry point for the Wyoming LuxTTS server."""
    parser = argparse.ArgumentParser(
        description="Wyoming LuxTTS Server - Voice cloning TTS via Wyoming protocol"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=10400, help="Server port")
    parser.add_argument("--reference", required=True, help="Reference audio for voice cloning")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"],
                        help="PyTorch device")
    parser.add_argument("--steps", type=int, default=4,
                        help="TTS quality (more steps = better quality, slower)")
    parser.add_argument("--voices-dir", default="voices",
                        help="Directory for cached voice prompts")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    asyncio.run(
        run_server(
            host=args.host,
            port=args.port,
            reference_audio=args.reference,
            device=args.device,
            num_steps=args.steps,
            voices_dir=args.voices_dir,
        )
    )


if __name__ == "__main__":
    main()
