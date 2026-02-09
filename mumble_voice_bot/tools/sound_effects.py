"""Sound effects tool for playing audio clips from a library or the web.

This tool allows the bot to browse and play sound effects from:
1. A local sound library (pre-downloaded sounds)
2. MyInstants.com (searched and downloaded on-demand)

Sound effects can be triggered by the LLM when contextually appropriate
(e.g., playing an "Among Us" sound when something suspicious is mentioned)
or explicitly requested by users.

The bot is encouraged to use sound effects proactively to be funny and
entertaining - like a soundboard DJ in voice chat.
"""

import asyncio
import hashlib
import json
import re
import time
import wave
from pathlib import Path
from typing import Any, Callable, Awaitable
from urllib.parse import urljoin, quote_plus

import numpy as np

from mumble_voice_bot.tools.base import Tool
from mumble_voice_bot.logging_config import get_logger

logger = get_logger(__name__)

# MyInstants configuration
MYINSTANTS_BASE_URL = "https://www.myinstants.com"
MYINSTANTS_SEARCH_URL = "https://www.myinstants.com/en/search/?name={query}"

# Request headers to avoid being blocked
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class SoundEffectsTool(Tool):
    """Tool to browse and play sound effects from a library or web search.

    The tool can:
    - Search local sound library by keyword
    - Search MyInstants.com for any sound effect
    - Download and cache sounds from the web
    - Play sounds through the voice channel
    
    The bot should use this tool proactively to:
    - React to funny moments with appropriate sounds
    - Play meme sounds when contextually relevant
    - Add comedic timing with sound effects
    - Be like a soundboard DJ in voice chat
    """

    def __init__(
        self,
        sounds_dir: str | Path,
        play_callback: Callable[[bytes, int], Awaitable[None]] | None = None,
        auto_play: bool = True,
        sample_rate: int = 48000,
        enable_web_search: bool = True,
        cache_web_sounds: bool = True,
        request_timeout: float = 10.0,
        verify_ssl: bool = True,
    ):
        """Initialize the sound effects tool.

        Args:
            sounds_dir: Directory containing sound effect files (also used for cache).
            play_callback: Async callback to play audio. Takes (pcm_bytes, sample_rate).
            auto_play: If True, LLM can autonomously play sounds it deems appropriate.
            sample_rate: Target sample rate for audio output (default 48kHz for Mumble).
            enable_web_search: If True, can search MyInstants.com for sounds.
            cache_web_sounds: If True, cache downloaded sounds locally.
            request_timeout: Timeout for web requests in seconds.
            verify_ssl: If False, disable SSL certificate verification (use for systems with cert issues).
        """
        self.sounds_dir = Path(sounds_dir)
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        self._play_callback = play_callback
        self.auto_play = auto_play
        self.sample_rate = sample_rate
        self.enable_web_search = enable_web_search
        self.cache_web_sounds = cache_web_sounds
        self.request_timeout = request_timeout
        self.verify_ssl = verify_ssl
        self._sound_index: dict[str, dict] | None = None
        
        # Web search cache (in-memory, for recent searches)
        self._web_search_cache: dict[str, tuple[float, list[dict]]] = {}
        self._web_cache_ttl = 300  # 5 minutes

    @property
    def name(self) -> str:
        return "sound_effects"

    @property
    def description(self) -> str:
        base = (
            "Play sound effects and meme sounds! You're like a soundboard DJ. "
            "Use action='search' to find sounds (searches local library AND MyInstants.com), "
            "action='play' to play a sound by name or search term. "
        )
        if self.auto_play:
            base += (
                "BE PROACTIVE AND FUNNY! Play sounds to enhance the vibe:\n"
                "- Someone says something sus? Play 'among us' sound\n"
                "- Victory moment? Play a fanfare or airhorn\n"
                "- Someone fails or something breaks? 'sad trombone' or 'bruh'\n"
                "- Dramatic revelation? 'dun dun dun' or dramatic chipmunk\n"
                "- Good news? 'hallelujah' or celebration sounds\n"
                "- Awkward moment? 'crickets' or 'curb your enthusiasm'\n"
                "Think like a funny soundboard operator - timing is everything!"
            )
        return base

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "play", "list", "web_search"],
                    "description": (
                        "'search' - find sounds locally and on MyInstants.com. "
                        "'play' - play a sound (will search if not found locally). "
                        "'list' - list cached/local sounds. "
                        "'web_search' - search only MyInstants.com (more results)."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "What sound to find or play. Be descriptive! "
                        "Examples: 'among us', 'sad trombone', 'bruh', 'airhorn', "
                        "'dramatic sound', 'victory fanfare', 'fail sound', 'brannigan'"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 5).",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["action"],
        }

    @property
    def example_call(self) -> str:
        return 'sound_effects(action="play", query="airhorn")'

    @property
    def usage_hint(self) -> str:
        return "MUST use when asked to play sounds, sound effects, or meme sounds"

    def _build_index(self) -> dict[str, dict]:
        """Build an index of available local/cached sounds.
        
        Returns:
            Dict mapping sound name to metadata.
        """
        if self._sound_index is not None:
            return self._sound_index

        index = {}
        
        if not self.sounds_dir.exists():
            logger.warning(f"Sounds directory not found: {self.sounds_dir}")
            self._sound_index = index
            return index

        audio_extensions = {".mp3", ".wav", ".ogg", ".flac"}
        
        for file_path in sorted(self.sounds_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in audio_extensions:
                continue

            base_name = file_path.stem
            
            # Try to load metadata from JSON
            json_path = file_path.with_suffix(".json")
            metadata = {
                "name": base_name,
                "file": str(file_path),
                "title": base_name.replace("_", " ").replace("-", " "),
                "tags": [],
                "description": "",
                "source": "local",
            }
            
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        json_data = json.load(f)
                    metadata["title"] = json_data.get("title", metadata["title"])
                    metadata["tags"] = json_data.get("tags", [])
                    metadata["description"] = json_data.get("description", "")
                    metadata["slug"] = json_data.get("slug", "")
                    metadata["source"] = json_data.get("source", "local")
                    metadata["page_url"] = json_data.get("page_url", "")
                except Exception as e:
                    logger.debug(f"Failed to load metadata for {base_name}: {e}")

            # Create searchable text
            search_parts = [
                metadata["title"].lower(),
                metadata["description"].lower(),
                base_name.lower(),
            ] + [t.lower() for t in metadata["tags"]]
            metadata["_searchable"] = " ".join(search_parts)

            index[base_name] = metadata

        logger.info(f"Indexed {len(index)} sound effects from {self.sounds_dir}")
        self._sound_index = index
        return index

    def refresh_index(self) -> int:
        """Refresh the sound index from disk.
        
        Returns:
            Number of sounds indexed.
        """
        self._sound_index = None
        return len(self._build_index())

    def search_sounds(self, query: str, limit: int = 10) -> list[dict]:
        """Search for sounds in local library.
        
        Args:
            query: Search keywords.
            limit: Maximum results to return.
            
        Returns:
            List of matching sound metadata dicts.
        """
        index = self._build_index()
        query_lower = query.lower()
        query_parts = query_lower.split()
        
        scored = []
        for name, metadata in index.items():
            searchable = metadata.get("_searchable", "")
            
            score = 0
            
            # Exact title match
            if query_lower == metadata["title"].lower():
                score += 100
            elif query_lower in metadata["title"].lower():
                score += 50
            
            # All query parts found
            if all(part in searchable for part in query_parts):
                score += 25
            
            for part in query_parts:
                if part in searchable:
                    score += 5
                if any(part in tag.lower() for tag in metadata.get("tags", [])):
                    score += 10
            
            if score > 0:
                scored.append((score, name, metadata))
        
        scored.sort(key=lambda x: -x[0])
        
        return [
            {
                "name": name,
                "title": meta["title"],
                "tags": meta.get("tags", [])[:5],
                "source": meta.get("source", "local"),
            }
            for score, name, meta in scored[:limit]
        ]

    async def search_myinstants(self, query: str, limit: int = 10) -> list[dict]:
        """Search MyInstants.com for sounds.
        
        Args:
            query: Search keywords.
            limit: Maximum results to return.
            
        Returns:
            List of sound info dicts with title, slug, page_url, audio_url.
        """
        if not self.enable_web_search:
            return []
        
        # Check cache
        cache_key = f"{query}:{limit}"
        if cache_key in self._web_search_cache:
            cached_time, cached_results = self._web_search_cache[cache_key]
            if time.time() - cached_time < self._web_cache_ttl:
                return cached_results
        
        try:
            import aiohttp
            import ssl
            
            search_url = MYINSTANTS_SEARCH_URL.format(query=quote_plus(query))
            
            # Create SSL context (disable verification if configured)
            ssl_context = None if self.verify_ssl else ssl.create_default_context()
            if not self.verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(headers=REQUEST_HEADERS, connector=connector) as session:
                async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as response:
                    if response.status != 200:
                        logger.warning(f"MyInstants search failed: {response.status}")
                        return []
                    
                    html = await response.text()
            
            # Parse the HTML to find sound buttons
            sounds = self._parse_myinstants_search(html, limit)
            
            # Cache results
            self._web_search_cache[cache_key] = (time.time(), sounds)
            
            return sounds
            
        except ImportError:
            logger.warning("aiohttp not available for web search")
            return []
        except asyncio.TimeoutError:
            logger.warning("MyInstants search timed out")
            return []
        except Exception as e:
            logger.error(f"MyInstants search error: {e}")
            return []

    def _parse_myinstants_search(self, html: str, limit: int) -> list[dict]:
        """Parse MyInstants search results HTML.
        
        Args:
            html: HTML content from search page.
            limit: Maximum results to return.
            
        Returns:
            List of sound info dicts.
        """
        sounds = []
        seen_slugs = set()
        
        # Find instant links with their titles
        # Pattern matches: <a href="/en/instant/slug/">Title</a>
        instant_pattern = re.compile(
            r'<a[^>]+href="(/en/instant/([^/"]+)/)"[^>]*>([^<]+)</a>',
            re.IGNORECASE
        )
        
        # Find all play() calls with audio URLs
        # Pattern matches: play('/media/sounds/file.mp3')
        audio_pattern = re.compile(
            r"play\(['\"]([^'\"]+\.mp3)['\"]",
            re.IGNORECASE
        )
        
        # Find all audio URLs in the page
        audio_urls = {}
        for match in audio_pattern.finditer(html):
            audio_path = match.group(1)
            # Try to extract slug from path
            # Format: /media/sounds/slug.mp3 or /media/sounds/something_slug.mp3
            audio_filename = audio_path.split('/')[-1].replace('.mp3', '')
            audio_urls[audio_filename] = urljoin(MYINSTANTS_BASE_URL, audio_path)
        
        # Find all instant links
        for link_match in instant_pattern.finditer(html):
            page_path, slug, title = link_match.groups()
            
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            
            title = title.strip()
            if not title or title.lower() in ['instant buttons', 'my instants']:
                continue
            
            # Try to find matching audio URL
            audio_url = None
            # Check various patterns
            for audio_key, url in audio_urls.items():
                if slug in audio_key or audio_key in slug:
                    audio_url = url
                    break
            
            sounds.append({
                "title": title,
                "slug": slug,
                "page_url": urljoin(MYINSTANTS_BASE_URL, page_path),
                "audio_url": audio_url,
                "source": "myinstants",
            })
            
            if len(sounds) >= limit:
                break
        
        return sounds

    async def _get_audio_url_from_page(self, page_url: str) -> str | None:
        """Fetch the audio URL from a MyInstants page.
        
        Args:
            page_url: URL of the instant page.
            
        Returns:
            Audio URL or None if not found.
        """
        try:
            import aiohttp
            import ssl
            
            # Create SSL context (disable verification if configured)
            ssl_context = None if self.verify_ssl else ssl.create_default_context()
            if not self.verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(headers=REQUEST_HEADERS, connector=connector) as session:
                async with session.get(page_url, timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as response:
                    if response.status != 200:
                        return None
                    
                    html = await response.text()
            
            # Look for play() call with audio URL
            audio_pattern = re.compile(r"play\(['\"]([^'\"]+\.mp3)['\"]")
            match = audio_pattern.search(html)
            
            if match:
                return urljoin(MYINSTANTS_BASE_URL, match.group(1))
            
            # Try data-sound attribute
            data_pattern = re.compile(r'data-sound="([^"]+\.mp3)"')
            match = data_pattern.search(html)
            
            if match:
                return urljoin(MYINSTANTS_BASE_URL, match.group(1))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get audio URL from {page_url}: {e}")
            return None

    async def download_sound(self, sound_info: dict) -> str | None:
        """Download a sound from the web and cache it locally.
        
        Args:
            sound_info: Sound metadata dict with audio_url or page_url.
            
        Returns:
            Local file path or None on error.
        """
        audio_url = sound_info.get("audio_url")
        
        # If no audio URL, try to get it from the page
        if not audio_url and sound_info.get("page_url"):
            audio_url = await self._get_audio_url_from_page(sound_info["page_url"])
        
        if not audio_url:
            logger.error(f"No audio URL for sound: {sound_info.get('title', 'unknown')}")
            return None
        
        try:
            import aiohttp
            import ssl
            
            # Generate a safe filename
            slug = sound_info.get("slug", "")
            if not slug:
                # Generate from title
                slug = re.sub(r'[^a-z0-9]+', '_', sound_info.get("title", "sound").lower())
                slug = slug.strip('_')[:50]
            
            # Add hash to avoid collisions
            url_hash = hashlib.md5(audio_url.encode()).hexdigest()[:8]
            filename = f"web_{slug}_{url_hash}.mp3"
            filepath = self.sounds_dir / filename
            
            # Check if already downloaded
            if filepath.exists():
                logger.debug(f"Sound already cached: {filename}")
                return str(filepath)
            
            # Create SSL context (disable verification if configured)
            ssl_context = None if self.verify_ssl else ssl.create_default_context()
            if not self.verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Download the file
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(headers=REQUEST_HEADERS, connector=connector) as session:
                async with session.get(audio_url, timeout=aiohttp.ClientTimeout(total=self.request_timeout)) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download sound: {response.status}")
                        return None
                    
                    audio_data = await response.read()
            
            # Save the audio file
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            # Save metadata
            if self.cache_web_sounds:
                metadata = {
                    "title": sound_info.get("title", slug),
                    "slug": slug,
                    "source": "myinstants",
                    "audio_url": audio_url,
                    "page_url": sound_info.get("page_url", ""),
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                json_path = filepath.with_suffix(".json")
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Downloaded sound: {filename}")
            
            # Refresh index to include new sound
            self.refresh_index()
            
            return str(filepath)
            
        except ImportError:
            logger.error("aiohttp not available for downloading")
            return None
        except Exception as e:
            logger.error(f"Failed to download sound: {e}")
            return None

    def _load_audio(self, file_path: str) -> tuple[bytes, int] | None:
        """Load audio file and convert to PCM.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Tuple of (pcm_bytes, sample_rate) or None on error.
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Audio file not found: {path}")
            return None

        try:
            # Try to use pydub for format conversion (handles mp3, etc.)
            try:
                from pydub import AudioSegment
                
                audio = AudioSegment.from_file(str(path))
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(self.sample_rate)
                audio = audio.set_sample_width(2)
                
                pcm_bytes = audio.raw_data
                return pcm_bytes, self.sample_rate
                
            except ImportError:
                logger.debug("pydub not available, trying wave module")
            
            # Fallback for WAV files only
            if path.suffix.lower() != ".wav":
                logger.error(f"Cannot load {path.suffix} without pydub installed")
                return None
                
            with wave.open(str(path), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                file_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                pcm_data = wav_file.readframes(n_frames)
            
            if sample_width == 2:
                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            elif sample_width == 1:
                audio_array = np.frombuffer(pcm_data, dtype=np.uint8)
                audio_array = (audio_array.astype(np.int16) - 128) * 256
            else:
                logger.error(f"Unsupported sample width: {sample_width}")
                return None
            
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            if file_rate != self.sample_rate:
                num_samples = int(len(audio_array) * self.sample_rate / file_rate)
                indices = np.linspace(0, len(audio_array) - 1, num_samples)
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
                audio_array = audio_array.astype(np.int16)
            
            return audio_array.tobytes(), self.sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio {path}: {e}")
            return None

    async def play_sound(self, query: str, search_web: bool = True) -> str:
        """Find and play a sound by name or search term.
        
        Args:
            query: Sound name or search keywords.
            search_web: If True, search web if not found locally.
            
        Returns:
            Status message.
        """
        if not self._play_callback:
            return "Sound playback not available (no playback callback configured)."
        
        # First, try local library
        index = self._build_index()
        metadata = index.get(query)
        
        if not metadata:
            # Try local search
            local_matches = self.search_sounds(query, limit=1)
            if local_matches:
                sound_name = local_matches[0]["name"]
                metadata = index.get(sound_name)
        
        # If not found locally and web search is enabled, search online
        if not metadata and search_web and self.enable_web_search:
            logger.info(f"Sound '{query}' not found locally, searching web...")
            web_results = await self.search_myinstants(query, limit=1)
            
            if web_results:
                # Download the first result
                filepath = await self.download_sound(web_results[0])
                if filepath:
                    # Reload index and get the new sound
                    index = self._build_index()
                    for name, meta in index.items():
                        if meta["file"] == filepath:
                            metadata = meta
                            break
        
        if not metadata:
            return f"Sound not found: '{query}'. Try a different search term!"
        
        # Load and play
        result = self._load_audio(metadata["file"])
        if not result:
            return f"Failed to load sound: '{metadata.get('title', query)}'."
        
        pcm_bytes, sample_rate = result
        
        try:
            await self._play_callback(pcm_bytes, sample_rate)
            source = metadata.get("source", "local")
            source_info = f" (from {source})" if source != "local" else ""
            return f"*plays {metadata['title']}*{source_info}"
        except Exception as e:
            logger.error(f"Playback failed: {e}")
            return f"Failed to play sound: {e}"

    async def execute(
        self,
        action: str,
        query: str | None = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> str:
        """Execute the sound effects action.

        Args:
            action: 'search', 'play', 'list', or 'web_search'
            query: Search keywords or sound name
            limit: Max results for search/list

        Returns:
            Result message.
        """
        if action == "list":
            index = self._build_index()
            if not index:
                return "No sounds cached locally. Use 'search' to find sounds online!"
            
            sounds = list(index.values())[:limit]
            lines = [f"Cached sounds ({len(index)} total):"]
            for sound in sounds:
                source = f" [{sound.get('source', 'local')}]" if sound.get('source') != 'local' else ""
                lines.append(f"- {sound['title']}{source}")
            
            if len(index) > limit:
                lines.append(f"... and {len(index) - limit} more.")
            
            return "\n".join(lines)

        elif action == "search":
            if not query:
                return "What sound are you looking for? Give me a search term!"
            
            results = []
            
            # Search local first
            local_results = self.search_sounds(query, limit=limit)
            for r in local_results:
                r["source"] = "cached"
            results.extend(local_results)
            
            # Then search web
            if self.enable_web_search and len(results) < limit:
                web_results = await self.search_myinstants(query, limit=limit - len(results))
                results.extend(web_results)
            
            if not results:
                return f"No sounds found for '{query}'. Try different keywords!"
            
            lines = [f"Sounds matching '{query}':"]
            for sound in results[:limit]:
                source = f" [{sound.get('source', 'local')}]"
                lines.append(f"- {sound['title']}{source}")
            
            lines.append("\nUse action='play' with query='<title>' to play one!")
            return "\n".join(lines)

        elif action == "web_search":
            if not query:
                return "What sound are you looking for?"
            
            if not self.enable_web_search:
                return "Web search is disabled. Use 'search' to find local sounds."
            
            results = await self.search_myinstants(query, limit=limit)
            
            if not results:
                return f"No sounds found on MyInstants for '{query}'."
            
            lines = [f"MyInstants results for '{query}':"]
            for sound in results:
                lines.append(f"- {sound['title']}")
            
            lines.append("\nUse action='play' to download and play one!")
            return "\n".join(lines)

        elif action == "play":
            if not query:
                return "What sound should I play? Give me a name or search term!"
            
            return await self.play_sound(query)

        else:
            return f"Unknown action: '{action}'. Use 'search', 'play', 'list', or 'web_search'."
