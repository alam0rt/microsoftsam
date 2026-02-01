#!/usr/bin/env python3
"""
Mumble TTS Bot - Reads text messages aloud using LuxTTS voice cloning.

Usage:
    python mumble_tts_bot.py --host localhost --user "TTS Bot" --reference voice.wav
"""
import argparse
import os
import re
import sys
import time
import threading
from queue import Queue

# Add vendor paths for pymumble and LuxTTS
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "botamusique"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LuxTTS"))
sys.path.insert(0, os.path.join(_THIS_DIR, "vendor", "LinaCodec", "src"))

import numpy as np

import pymumble_py3 as pymumble
from pymumble_py3.constants import PYMUMBLE_CLBK_TEXTMESSAGERECEIVED

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
        
        # Generate and play speech
        try:
            self.speak(text)
        except Exception as e:
            print(f"TTS error: {e}")
    
    def speak(self, text: str):
        """Generate speech from text and send to Mumble with streaming.
        
        Uses streaming API to start playback as soon as the first chunk is ready,
        significantly reducing perceived latency for longer messages.
        """
        # Use streaming generation - yields audio chunks as they're ready
        for wav_chunk in self.tts.generate_speech_streaming(
            text, self.voice_prompt, num_steps=self.num_steps
        ):
            wav_float = wav_chunk.numpy().squeeze()
            
            # Convert float32 [-1, 1] to int16 PCM
            # Clip to prevent overflow
            wav_float = np.clip(wav_float, -1.0, 1.0)
            pcm = (wav_float * 32767).astype(np.int16)
            
            # Send chunk to Mumble immediately (48kHz 16-bit mono PCM)
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
