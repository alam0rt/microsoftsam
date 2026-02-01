#!/usr/bin/env python3
"""
Mumble TTS Bot - Reads text messages aloud using LuxTTS voice cloning.

Usage:
    python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference voice.wav
"""
import argparse
import io
import re
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

# pymumble_py3 is added to path via the cloned botamusique repo
import pymumble_py3 as pymumble
from pymumble_py3.constants import (
    PYMUMBLE_CLBK_SOUNDRECEIVED,
    PYMUMBLE_CLBK_TEXTMESSAGERECEIVED,
)

from zipvoice.luxvoice import LuxTTS


def strip_html(text: str) -> str:
    """Remove HTML tags from text (Mumble messages can contain HTML)."""
    return re.sub(r'<[^>]+>', '', text)


class MumbleTTSBot:
    """A Mumble bot that speaks text messages using LuxTTS."""
    
    def __init__(
        self,
        host: str,
        user: str,
        port: int = 64738,
        password: str = '',
        channel: str = None,
        reference_audio: str = 'reference.wav',
        device: str = 'cpu',
        num_steps: int = 4,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.channel = channel
        self.device = device
        self.num_steps = num_steps
        
        # Initialize TTS
        print(f"Loading LuxTTS model on {device}...")
        self.tts = LuxTTS('YatharthS/LuxTTS', device=device, threads=2)
        
        print(f"Encoding reference audio: {reference_audio}")
        self.voice_prompt = self.tts.encode_prompt(reference_audio, rms=0.01)
        
        # Recording state for @clone command
        self.recording = False
        self.recording_lock = threading.Lock()
        self.recorded_audio = []
        self.recording_end_time = None
        
        # Command pattern for @clone
        self.clone_pattern = re.compile(r'@clone\s+(\d+)', re.IGNORECASE)
        
        # Initialize Mumble connection
        print(f"Connecting to {host}:{port} as '{user}'...")
        self.mumble = pymumble.Mumble(
            host=host,
            user=user,
            port=port,
            password=password,
            reconnect=True,
        )
        
        # Set up text message callback
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_TEXTMESSAGERECEIVED,
            self.on_message
        )
        
        # Set up sound received callback for recording
        self.mumble.callbacks.set_callback(
            PYMUMBLE_CLBK_SOUNDRECEIVED,
            self.on_sound_received
        )
        
    def start(self):
        """Start the bot and connect to the server."""
        self.mumble.start()
        self.mumble.is_ready()  # Blocks until connected
        print("Connected to Mumble server!")
        
        # Join channel if specified
        if self.channel:
            self.join_channel(self.channel)
            
    def join_channel(self, channel_name: str):
        """Join a channel by name."""
        try:
            channel = self.mumble.channels.find_by_name(channel_name)
            channel.move_in()
            print(f"Joined channel: {channel_name}")
        except Exception as e:
            print(f"Failed to join channel '{channel_name}': {e}")
    
    def on_message(self, message):
        """Callback for received text messages."""
        # Extract and clean the message text
        text = strip_html(message.message)
        
        if not text.strip():
            return
            
        # Get sender name if available
        sender = "Someone"
        if hasattr(message, 'actor') and message.actor in self.mumble.users:
            sender = self.mumble.users[message.actor]['name']
        
        print(f"[{sender}]: {text}")
        
        # Check for @clone command
        clone_match = self.clone_pattern.search(text)
        if clone_match:
            seconds = int(clone_match.group(1))
            self.handle_clone_command(seconds)
            return
        
        # Generate and play speech
        try:
            self.speak(text)
        except Exception as e:
            print(f"TTS error: {e}")
    
    def on_sound_received(self, user, sound_chunk):
        """Callback for received audio - used for recording."""
        with self.recording_lock:
            if self.recording and time.time() < self.recording_end_time:
                # Append PCM data (16-bit signed, 48kHz)
                self.recorded_audio.append(sound_chunk.pcm)
    
    def handle_clone_command(self, seconds: int):
        """Handle the @clone command to record and reinit voice."""
        # Clamp seconds between 3 and 10
        seconds = max(3, min(10, seconds))
        
        # Check if already recording
        with self.recording_lock:
            if self.recording:
                self.send_channel_message("Already recording!")
                return
        
        self.send_channel_message(f"Recording {seconds} seconds of audio for voice cloning...")
        
        # Enable receiving sound
        self.mumble.set_receive_sound(True)
        
        # Start recording
        with self.recording_lock:
            self.recording = True
            self.recorded_audio = []
            self.recording_end_time = time.time() + seconds
        
        # Wait for recording to complete
        time.sleep(seconds + 0.5)  # Extra buffer for audio processing
        
        # Stop recording
        with self.recording_lock:
            self.recording = False
            audio_chunks = self.recorded_audio
            self.recorded_audio = []
        
        # Disable receiving sound to save resources
        self.mumble.set_receive_sound(False)
        
        if not audio_chunks:
            self.send_channel_message("No audio recorded! Make sure someone is speaking.")
            return
        
        # Combine all audio chunks
        combined_pcm = b''.join(audio_chunks)
        
        if len(combined_pcm) < 48000 * 2:  # Less than 1 second of audio
            self.send_channel_message("Not enough audio recorded. Please try again with more speech.")
            return
        
        # Convert PCM bytes to numpy array (16-bit signed, mono, 48kHz)
        audio_array = np.frombuffer(combined_pcm, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Save to temporary WAV file
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio_float, 48000)
            
            # Reinitialize voice prompt with recorded audio
            print(f"Encoding new reference audio from recording...")
            self.voice_prompt = self.tts.encode_prompt(tmp_path, rms=0.01)
            
            self.send_channel_message("Voice cloned successfully! I'll now speak with the new voice.")
            print("Voice prompt updated from recorded audio")
            
        except Exception as e:
            self.send_channel_message(f"Failed to clone voice: {e}")
            print(f"Clone error: {e}")
    
    def send_channel_message(self, text: str):
        """Send a text message to the bot's current channel."""
        try:
            channel = self.mumble.my_channel()
            channel.send_text_message(text)
        except Exception as e:
            print(f"Failed to send message: {e}")
    
    def speak(self, text: str):
        """Generate speech from text and send to Mumble."""
        # Generate speech with LuxTTS
        wav = self.tts.generate_speech(text, self.voice_prompt, num_steps=self.num_steps)
        wav_float = wav.numpy().squeeze()
        
        # Convert float32 [-1, 1] to int16 PCM
        # Clip to prevent overflow
        wav_float = np.clip(wav_float, -1.0, 1.0)
        pcm = (wav_float * 32767).astype(np.int16)
        
        # Send to Mumble (48kHz 16-bit mono PCM)
        self.mumble.sound_output.add_sound(pcm.tobytes())
        
    def run_forever(self):
        """Keep the bot running."""
        print("Bot is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(
        description='Mumble TTS Bot - Reads text messages using LuxTTS'
    )
    parser.add_argument('--host', default='localhost',
                        help='Mumble server address')
    parser.add_argument('--port', type=int, default=64738,
                        help='Mumble server port')
    parser.add_argument('--user', default='TTSBot',
                        help='Bot username')
    parser.add_argument('--password', default='',
                        help='Server password')
    parser.add_argument('--channel', default=None,
                        help='Channel to join')
    parser.add_argument('--reference', default='reference.wav',
                        help='Reference audio file for voice cloning')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Compute device')
    parser.add_argument('--steps', type=int, default=4,
                        help='Number of inference steps (quality vs speed)')
    
    args = parser.parse_args()
    
    bot = MumbleTTSBot(
        host=args.host,
        user=args.user,
        port=args.port,
        password=args.password,
        channel=args.channel,
        reference_audio=args.reference,
        device=args.device,
        num_steps=args.steps,
    )
    
    bot.start()
    bot.run_forever()


if __name__ == '__main__':
    main()
