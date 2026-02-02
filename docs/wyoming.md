# Wyoming Protocol Integration Plan

This document outlines the plan to integrate the Wyoming protocol into the microsoftsam project for:
1. **Speech-to-Text (STT)**: Using an external `wyoming-faster-whisper` server
2. **Text-to-Speech (TTS)**: Wrapping our existing LuxTTS implementation as a Wyoming server
3. **Testing**: Comprehensive test suite for both components

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           microsoftsam Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐      Wyoming Protocol       ┌────────────────────────┐   │
│  │   Mumble     │◄──────────────────────────►│  wyoming-faster-whisper │   │
│  │    Bot       │         (TCP)               │  (External Server)     │   │
│  │              │                             │  STT Backend           │   │
│  └──────┬───────┘                             └────────────────────────┘   │
│         │                                                                   │
│         │              Wyoming Protocol       ┌────────────────────────┐   │
│         └────────────────────────────────────►│  wyoming-luxtts        │   │
│                        (TCP)                  │  (Our TTS Server)      │   │
│                                               │  Wraps StreamingLuxTTS │   │
│                                               └────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Wyoming Client for STT (wyoming-faster-whisper)

### 1.1 Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing deps ...
    "wyoming>=1.5.0",
]
```

### 1.2 Create Wyoming STT Client Module

Create `wyoming_stt_client.py`:

```python
"""
Wyoming protocol client for Speech-to-Text using external wyoming-faster-whisper server.
"""
import asyncio
from typing import Optional, AsyncIterator
from wyoming.client import AsyncTcpClient
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info


class WyomingSTTClient:
    """Client for connecting to a Wyoming-compatible STT server."""
    
    def __init__(self, host: str = "localhost", port: int = 10300):
        self.host = host
        self.port = port
        self._client: Optional[AsyncTcpClient] = None
    
    async def connect(self) -> None:
        """Connect to the Wyoming STT server."""
        self._client = AsyncTcpClient(self.host, self.port)
        await self._client.connect()
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._client:
            await self._client.disconnect()
            self._client = None
    
    async def get_info(self) -> Info:
        """Get server capabilities."""
        await self._client.write_event(Describe().event())
        event = await self._client.read_event()
        return Info.from_event(event)
    
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Audio sample rate in Hz
            sample_width: Bytes per sample (2 = 16-bit)
            channels: Number of audio channels
            language: Optional language hint (e.g., "en")
        
        Returns:
            Transcribed text
        """
        # Send transcribe request with optional language
        await self._client.write_event(
            Transcribe(language=language).event()
        )
        
        # Send audio start
        await self._client.write_event(
            AudioStart(
                rate=sample_rate,
                width=sample_width,
                channels=channels,
            ).event()
        )
        
        # Send audio chunks (split into reasonable sizes)
        chunk_size = sample_rate * sample_width * channels  # 1 second chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            await self._client.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=sample_rate,
                    width=sample_width,
                    channels=channels,
                ).event()
            )
        
        # Send audio stop
        await self._client.write_event(AudioStop().event())
        
        # Wait for transcript
        event = await self._client.read_event()
        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            return transcript.text
        
        return ""
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe streaming audio data.
        
        Args:
            audio_stream: Async iterator yielding audio chunks
            sample_rate: Audio sample rate in Hz
            sample_width: Bytes per sample
            channels: Number of audio channels
            language: Optional language hint
        
        Returns:
            Transcribed text
        """
        await self._client.write_event(
            Transcribe(language=language).event()
        )
        
        await self._client.write_event(
            AudioStart(
                rate=sample_rate,
                width=sample_width,
                channels=channels,
            ).event()
        )
        
        async for chunk in audio_stream:
            await self._client.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=sample_rate,
                    width=sample_width,
                    channels=channels,
                ).event()
            )
        
        await self._client.write_event(AudioStop().event())
        
        event = await self._client.read_event()
        if Transcript.is_type(event.type):
            transcript = Transcript.from_event(event)
            return transcript.text
        
        return ""
```

### 1.3 Integration Points

Modify `mumble_tts_bot.py` to use Wyoming STT instead of direct Whisper:

```python
# Add configuration options
parser.add_argument('--wyoming-stt-host', default='localhost',
                    help='Wyoming STT server host')
parser.add_argument('--wyoming-stt-port', type=int, default=10300,
                    help='Wyoming STT server port')
parser.add_argument('--use-wyoming-stt', action='store_true',
                    help='Use Wyoming protocol for STT instead of local Whisper')
```

---

## Phase 2: Wyoming TTS Server Wrapping LuxTTS

### 2.1 Create Wyoming TTS Server Module

Create `wyoming_luxtts_server.py`:

```python
"""
Wyoming protocol server wrapping LuxTTS for Text-to-Speech.

Usage:
    python wyoming_luxtts_server.py --port 10400 --reference voice.wav
"""
import argparse
import asyncio
import logging
import wave
import io
from functools import partial
from typing import Optional

from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info, TtsProgram, TtsModel, TtsVoice, Attribution
from wyoming.event import Event

# Import our StreamingLuxTTS
from mumble_tts_bot import StreamingLuxTTS

_LOGGER = logging.getLogger(__name__)


class LuxTTSEventHandler(AsyncEventHandler):
    """Handle Wyoming TTS events using LuxTTS."""
    
    def __init__(
        self,
        tts: StreamingLuxTTS,
        reference_audio: bytes,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tts = tts
        self.reference_audio = reference_audio
        self._streaming_text = ""
    
    async def handle_event(self, event: Event) -> bool:
        """Handle incoming Wyoming events."""
        
        if Describe.is_type(event.type):
            await self._handle_describe()
            return True
        
        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            await self._synthesize(synthesize.text)
            return True
        
        if SynthesizeStart.is_type(event.type):
            self._streaming_text = ""
            return True
        
        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            self._streaming_text += chunk.text
            return True
        
        if SynthesizeStop.is_type(event.type):
            if self._streaming_text:
                await self._synthesize(self._streaming_text)
                self._streaming_text = ""
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
                        models=[
                            TtsModel(
                                name="YatharthS/LuxTTS",
                                description="High-quality 48kHz voice cloning TTS",
                                languages=["en"],
                                installed=True,
                                attribution=Attribution(
                                    name="LuxTTS",
                                    url="https://luxtts.com/",
                                ),
                            )
                        ],
                        voices=[
                            TtsVoice(
                                name="cloned",
                                description="Voice cloned from reference audio",
                                languages=["en"],
                            )
                        ],
                    )
                ]
            ).event()
        )
    
    async def _synthesize(self, text: str) -> None:
        """Generate speech from text using LuxTTS."""
        _LOGGER.info(f"Synthesizing: {text[:50]}...")
        
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
        
        # Send in chunks (1 second each)
        chunk_samples = sample_rate * sample_width * channels
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
    
    def _generate_speech(self, text: str) -> bytes:
        """Generate speech synchronously (called in executor)."""
        # Use streaming for better responsiveness
        audio_chunks = []
        for chunk in self.tts.generate_speech_streaming(
            text,
            self.reference_audio,
        ):
            audio_chunks.append(chunk)
        
        # Concatenate and convert to bytes
        import numpy as np
        audio = np.concatenate(audio_chunks)
        
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()


async def main() -> None:
    """Run the Wyoming LuxTTS server."""
    parser = argparse.ArgumentParser(description="Wyoming LuxTTS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=10400, help="Server port")
    parser.add_argument("--reference", required=True, help="Reference audio for voice cloning")
    parser.add_argument("--device", default="cuda", help="PyTorch device (cuda/cpu/mps)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    _LOGGER.info("Loading LuxTTS model...")
    tts = StreamingLuxTTS(device=args.device)
    
    _LOGGER.info(f"Loading reference audio: {args.reference}")
    with open(args.reference, "rb") as f:
        reference_audio = f.read()
    
    _LOGGER.info(f"Starting Wyoming TTS server on {args.host}:{args.port}")
    
    server = AsyncServer.from_uri(f"tcp://{args.host}:{args.port}")
    
    await server.run(
        partial(
            LuxTTSEventHandler,
            tts=tts,
            reference_audio=reference_audio,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
```

### 2.2 Create Wyoming TTS Client

Create `wyoming_tts_client.py`:

```python
"""
Wyoming protocol client for Text-to-Speech.
"""
import asyncio
from typing import Optional, AsyncIterator
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info


class WyomingTTSClient:
    """Client for connecting to a Wyoming-compatible TTS server."""
    
    def __init__(self, host: str = "localhost", port: int = 10400):
        self.host = host
        self.port = port
        self._client: Optional[AsyncTcpClient] = None
    
    async def connect(self) -> None:
        """Connect to the Wyoming TTS server."""
        self._client = AsyncTcpClient(self.host, self.port)
        await self._client.connect()
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._client:
            await self._client.disconnect()
            self._client = None
    
    async def get_info(self) -> Info:
        """Get server capabilities."""
        await self._client.write_event(Describe().event())
        event = await self._client.read_event()
        return Info.from_event(event)
    
    async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            voice: Optional voice name
        
        Returns:
            Raw PCM audio bytes (48kHz, 16-bit, mono)
        """
        await self._client.write_event(Synthesize(text=text).event())
        
        audio_chunks = []
        sample_rate = 48000
        sample_width = 2
        channels = 1
        
        while True:
            event = await self._client.read_event()
            
            if AudioStart.is_type(event.type):
                audio_start = AudioStart.from_event(event)
                sample_rate = audio_start.rate
                sample_width = audio_start.width
                channels = audio_start.channels
                continue
            
            if AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                audio_chunks.append(chunk.audio)
                continue
            
            if AudioStop.is_type(event.type):
                break
        
        return b"".join(audio_chunks), sample_rate, sample_width, channels
    
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Synthesize text to audio, yielding chunks as they arrive.
        
        Args:
            text: Text to synthesize
        
        Yields:
            Raw PCM audio chunks
        """
        await self._client.write_event(Synthesize(text=text).event())
        
        while True:
            event = await self._client.read_event()
            
            if AudioStart.is_type(event.type):
                continue
            
            if AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                yield chunk.audio
                continue
            
            if AudioStop.is_type(event.type):
                break
```

---

## Phase 3: Testing

### 3.1 Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_wyoming_stt_client.py
├── test_wyoming_tts_server.py
├── test_wyoming_tts_client.py
├── test_integration.py      # End-to-end tests
└── fixtures/
    └── test_audio.wav       # Short test audio file
```

### 3.2 Test Fixtures (`tests/conftest.py`)

```python
"""Shared test fixtures for Wyoming protocol tests."""
import asyncio
import pytest
import numpy as np
import wave
import io
from pathlib import Path


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_audio_pcm() -> bytes:
    """Generate test PCM audio (1 second of 440Hz sine wave)."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()


@pytest.fixture
def test_audio_48k_pcm() -> bytes:
    """Generate test PCM audio at 48kHz (matching LuxTTS output)."""
    sample_rate = 48000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()


@pytest.fixture
def reference_audio_path(tmp_path) -> Path:
    """Create a temporary reference audio file."""
    sample_rate = 48000
    duration = 2.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    audio_int16 = (audio * 32767).astype(np.int16)
    
    wav_path = tmp_path / "reference.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    return wav_path
```

### 3.3 STT Client Tests (`tests/test_wyoming_stt_client.py`)

```python
"""Tests for Wyoming STT client."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from wyoming_stt_client import WyomingSTTClient
from wyoming.asr import Transcript
from wyoming.info import Info, AsrProgram, AsrModel, Attribution
from wyoming.event import Event


class TestWyomingSTTClient:
    """Test suite for WyomingSTTClient."""
    
    @pytest.fixture
    def client(self):
        """Create a client instance."""
        return WyomingSTTClient(host="localhost", port=10300)
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, client):
        """Test connection lifecycle."""
        with patch.object(client, '_client', new_callable=AsyncMock) as mock_client:
            mock_client.connect = AsyncMock()
            mock_client.disconnect = AsyncMock()
            
            await client.connect()
            assert client._client is not None
            
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_info(self, client):
        """Test getting server info."""
        mock_info = Info(
            asr=[
                AsrProgram(
                    name="faster-whisper",
                    models=[
                        AsrModel(
                            name="base",
                            languages=["en"],
                            installed=True,
                            attribution=Attribution(name="OpenAI", url="https://openai.com"),
                        )
                    ],
                    installed=True,
                    attribution=Attribution(name="Faster Whisper", url="https://github.com"),
                )
            ]
        )
        
        with patch.object(client, '_client', new_callable=AsyncMock) as mock_client:
            mock_client.write_event = AsyncMock()
            mock_client.read_event = AsyncMock(return_value=mock_info.event())
            
            info = await client.get_info()
            
            assert len(info.asr) == 1
            assert info.asr[0].name == "faster-whisper"
    
    @pytest.mark.asyncio
    async def test_transcribe(self, client, test_audio_pcm):
        """Test transcribing audio."""
        expected_text = "Hello, world!"
        mock_transcript = Transcript(text=expected_text)
        
        with patch.object(client, '_client', new_callable=AsyncMock) as mock_client:
            mock_client.write_event = AsyncMock()
            mock_client.read_event = AsyncMock(return_value=mock_transcript.event())
            
            result = await client.transcribe(
                audio_data=test_audio_pcm,
                sample_rate=16000,
                language="en",
            )
            
            assert result == expected_text
            # Verify correct events were sent
            calls = mock_client.write_event.call_args_list
            assert len(calls) >= 3  # transcribe, audio-start, audio-chunk(s), audio-stop


class TestWyomingSTTClientIntegration:
    """Integration tests requiring a running Wyoming STT server."""
    
    @pytest.mark.skipif(
        True,  # Set to False when server is available
        reason="Requires running Wyoming STT server"
    )
    @pytest.mark.asyncio
    async def test_real_transcription(self, test_audio_pcm):
        """Test against a real Wyoming server."""
        client = WyomingSTTClient(host="localhost", port=10300)
        await client.connect()
        
        try:
            info = await client.get_info()
            assert info.asr is not None
            
            # Note: Won't get meaningful text from sine wave
            result = await client.transcribe(test_audio_pcm)
            assert isinstance(result, str)
        finally:
            await client.disconnect()
```

### 3.4 TTS Server Tests (`tests/test_wyoming_tts_server.py`)

```python
"""Tests for Wyoming LuxTTS server."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from wyoming_luxtts_server import LuxTTSEventHandler
from wyoming.tts import Synthesize
from wyoming.info import Describe
from wyoming.event import Event


class TestLuxTTSEventHandler:
    """Test suite for LuxTTSEventHandler."""
    
    @pytest.fixture
    def mock_tts(self):
        """Create a mock TTS instance."""
        tts = MagicMock()
        # Mock streaming to return simple audio
        tts.generate_speech_streaming = MagicMock(
            return_value=[np.zeros(48000, dtype=np.float32)]  # 1 second silence
        )
        return tts
    
    @pytest.fixture
    def handler(self, mock_tts, reference_audio_path):
        """Create handler instance."""
        with open(reference_audio_path, "rb") as f:
            reference_audio = f.read()
        
        handler = LuxTTSEventHandler(
            tts=mock_tts,
            reference_audio=reference_audio,
        )
        handler.write_event = AsyncMock()
        return handler
    
    @pytest.mark.asyncio
    async def test_handle_describe(self, handler):
        """Test describe event handling."""
        event = Describe().event()
        
        result = await handler.handle_event(event)
        
        assert result is True
        handler.write_event.assert_called_once()
        
        # Verify Info was sent
        call_args = handler.write_event.call_args[0][0]
        assert call_args.type == "info"
    
    @pytest.mark.asyncio
    async def test_handle_synthesize(self, handler):
        """Test synthesize event handling."""
        event = Synthesize(text="Hello world").event()
        
        result = await handler.handle_event(event)
        
        assert result is True
        # Should have sent audio-start, audio-chunk(s), audio-stop
        assert handler.write_event.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_handle_unknown_event(self, handler):
        """Test handling unknown event types."""
        event = Event(type="unknown-event", data={})
        
        result = await handler.handle_event(event)
        
        assert result is False


import numpy as np  # Add at top of file for mock
```

### 3.5 TTS Client Tests (`tests/test_wyoming_tts_client.py`)

```python
"""Tests for Wyoming TTS client."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from wyoming_tts_client import WyomingTTSClient
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.info import Info, TtsProgram, TtsModel, Attribution


class TestWyomingTTSClient:
    """Test suite for WyomingTTSClient."""
    
    @pytest.fixture
    def client(self):
        """Create a client instance."""
        return WyomingTTSClient(host="localhost", port=10400)
    
    @pytest.mark.asyncio
    async def test_synthesize(self, client, test_audio_48k_pcm):
        """Test text synthesis."""
        events = [
            AudioStart(rate=48000, width=2, channels=1).event(),
            AudioChunk(audio=test_audio_48k_pcm, rate=48000, width=2, channels=1).event(),
            AudioStop().event(),
        ]
        
        with patch.object(client, '_client', new_callable=AsyncMock) as mock_client:
            mock_client.write_event = AsyncMock()
            mock_client.read_event = AsyncMock(side_effect=events)
            
            audio, rate, width, channels = await client.synthesize("Hello world")
            
            assert audio == test_audio_48k_pcm
            assert rate == 48000
            assert width == 2
            assert channels == 1
    
    @pytest.mark.asyncio
    async def test_synthesize_stream(self, client, test_audio_48k_pcm):
        """Test streaming synthesis."""
        chunk_size = len(test_audio_48k_pcm) // 3
        chunks = [
            test_audio_48k_pcm[i:i + chunk_size]
            for i in range(0, len(test_audio_48k_pcm), chunk_size)
        ]
        
        events = [
            AudioStart(rate=48000, width=2, channels=1).event(),
            *[AudioChunk(audio=c, rate=48000, width=2, channels=1).event() for c in chunks],
            AudioStop().event(),
        ]
        
        with patch.object(client, '_client', new_callable=AsyncMock) as mock_client:
            mock_client.write_event = AsyncMock()
            mock_client.read_event = AsyncMock(side_effect=events)
            
            received_chunks = []
            async for chunk in client.synthesize_stream("Hello world"):
                received_chunks.append(chunk)
            
            assert len(received_chunks) == len(chunks)
```

### 3.6 Integration Tests (`tests/test_integration.py`)

```python
"""End-to-end integration tests."""
import pytest
import asyncio
import subprocess
import time
import os

from wyoming_stt_client import WyomingSTTClient
from wyoming_tts_client import WyomingTTSClient


@pytest.fixture(scope="module")
def tts_server(reference_audio_path):
    """Start the Wyoming TTS server for integration tests."""
    # Skip if no GPU available for faster testing
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests")
    
    proc = subprocess.Popen([
        "python", "wyoming_luxtts_server.py",
        "--port", "10401",  # Use different port for tests
        "--reference", str(reference_audio_path),
        "--device", "cpu",  # Use CPU for CI
    ])
    
    # Wait for server to start
    time.sleep(5)
    
    yield proc
    
    proc.terminate()
    proc.wait()


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_tts_roundtrip(self, tts_server):
        """Test full TTS synthesis through Wyoming protocol."""
        client = WyomingTTSClient(host="localhost", port=10401)
        await client.connect()
        
        try:
            # Get server info
            info = await client.get_info()
            assert info.tts is not None
            assert len(info.tts) > 0
            
            # Synthesize text
            audio, rate, width, channels = await client.synthesize(
                "This is a test of the Wyoming TTS server."
            )
            
            # Verify we got audio back
            assert len(audio) > 0
            assert rate == 48000
            assert width == 2
            assert channels == 1
        finally:
            await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_stt_tts_pipeline(self, tts_server):
        """Test STT -> TTS pipeline (requires external STT server)."""
        # This test requires both servers running
        stt_client = WyomingSTTClient(host="localhost", port=10300)
        tts_client = WyomingTTSClient(host="localhost", port=10401)
        
        try:
            await tts_client.connect()
            
            # Generate some audio with TTS
            audio, rate, width, channels = await tts_client.synthesize("Hello world")
            
            # Try to transcribe it back (requires STT server)
            try:
                await stt_client.connect()
                text = await stt_client.transcribe(
                    audio_data=audio,
                    sample_rate=rate,
                    sample_width=width,
                    channels=channels,
                )
                # Should get something back (might not be exact due to audio quality)
                assert isinstance(text, str)
            except ConnectionRefusedError:
                pytest.skip("STT server not available")
            finally:
                await stt_client.disconnect()
        finally:
            await tts_client.disconnect()
```

### 3.7 Update `pyproject.toml` for Testing

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v --cov=. --cov-report=term-missing"
```

---

## Phase 4: Deployment & Configuration

### 4.1 Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  wyoming-faster-whisper:
    image: rhasspy/wyoming-faster-whisper
    ports:
      - "10300:10300"
    command: --model base --language en
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  wyoming-luxtts:
    build: .
    ports:
      - "10400:10400"
    volumes:
      - ./reference.wav:/app/reference.wav:ro
    command: >
      python wyoming_luxtts_server.py
      --host 0.0.0.0
      --port 10400
      --reference /app/reference.wav
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  mumble-bot:
    build: .
    depends_on:
      - wyoming-faster-whisper
      - wyoming-luxtts
    command: >
      python mumble_tts_bot.py
      --host mumble-server
      --user "TTS Bot"
      --use-wyoming-stt
      --wyoming-stt-host wyoming-faster-whisper
      --wyoming-stt-port 10300
      --wyoming-tts-host wyoming-luxtts
      --wyoming-tts-port 10400
```

### 4.2 Environment Configuration

Create `.env.example`:

```env
# Wyoming STT Configuration
WYOMING_STT_HOST=localhost
WYOMING_STT_PORT=10300

# Wyoming TTS Configuration  
WYOMING_TTS_HOST=localhost
WYOMING_TTS_PORT=10400
WYOMING_TTS_REFERENCE=reference.wav

# Mumble Configuration
MUMBLE_HOST=localhost
MUMBLE_USER="TTS Bot"
```

---

## Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Wyoming STT Client | 1-2 days |
| 2 | Wyoming LuxTTS Server + Client | 2-3 days |
| 3 | Test Suite | 1-2 days |
| 4 | Integration & Deployment | 1 day |

**Total estimated time: 5-8 days**

---

## References

- [Wyoming Protocol Specification](https://github.com/OHF-Voice/wyoming)
- [wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper)
- [wyoming-piper (TTS reference implementation)](https://github.com/rhasspy/wyoming-piper)
- [LuxTTS](https://github.com/ysharma3501/LuxTTS)
