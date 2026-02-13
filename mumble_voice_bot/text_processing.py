"""Text processing utilities for TTS preparation.

Extracted from mumble_tts_bot.py: split_into_sentences(), _pad_tts_text(),
_sanitize_for_tts(). These functions prepare text for TTS synthesis.

Note: PhraseChunker and SentenceChunker remain in phrase_chunker.py as they
handle streaming LLM token buffering, which is a different concern from
TTS text preparation.
"""

import re
from typing import List


def split_into_sentences(text: str, max_chars: int = 120) -> List[str]:
    """Split text into speakable chunks optimized for streaming TTS.

    Strategy:
    - Split on sentence boundaries first
    - Split long sentences on clause boundaries
    - Ensure minimum chunk size for natural speech

    Args:
        text: The text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks suitable for TTS.
    """
    MIN_CHUNK = 20  # Don't create tiny chunks

    # First pass: split on sentence endings
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())

    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Split long sentences on clause boundaries
            clause_pattern = r'(?<=[,;:])\s+'
            clauses = re.split(clause_pattern, sentence)

            # Merge tiny clauses
            current = ""
            for clause in clauses:
                if len(current) + len(clause) < MIN_CHUNK:
                    current += (" " if current else "") + clause
                else:
                    if current:
                        chunks.append(current)
                    current = clause
            if current:
                chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def pad_tts_text(text: str, min_chars: int = 20) -> str:
    """Pad text to avoid very short TTS inputs that can crash the vocoder.

    Uses ellipses as padding -- these generate natural pauses without producing
    transcribable words that would cause feedback loops.

    Only minimal padding is applied (20 chars) to prevent vocoder crashes
    while keeping audio duration reasonable.

    Args:
        text: Text to pad.
        min_chars: Minimum character count after padding.

    Returns:
        Padded text, or empty string if input is empty.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    # Only pad if very short - add trailing ellipses for natural pause
    while len(cleaned) < min_chars:
        cleaned = f"{cleaned}..."

    return cleaned


def sanitize_for_tts(text: str) -> str:
    """Remove emojis and non-speakable characters from text for TTS.

    Strips: emojis, asterisks, dashes used for formatting, timestamps,
    self-identification prefixes, etc.
    Keeps: letters, numbers, basic punctuation (.,!?'), spaces.

    Args:
        text: Raw text (e.g., from LLM output).

    Returns:
        Cleaned text suitable for TTS synthesis.
    """
    # Remove timestamp prefixes like "[11:40 AM]" or "[2:30 PM]" at the start
    text = re.sub(r'^\s*\[\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\]\s*', '', text)

    # Remove self-identification prefixes like "Raf:" or "Bot:" at the start
    text = re.sub(r'^[A-Za-z]+:\s*', '', text)

    # Remove emojis (Unicode emoji ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended
        "\U00002600-\U000026FF"  # misc symbols (sun, clouds, etc)
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # Remove emoji modifiers, joiners, and variation selectors
    text = re.sub(r'[\u200d\ufe0e\ufe0f]', '', text)

    # Remove formatting characters: *, _, ~, `, #, |, <, >, [], {}
    text = re.sub(r'[\*_~`#|<>\[\]{}]', '', text)

    # Replace em-dash and en-dash with space
    text = re.sub(r'[—–]', ' ', text)

    # Replace multiple dashes with single dash, then remove standalone dashes
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'\s-\s', ' ', text)

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_question(text: str) -> bool:
    """Detect if text is likely a question using simple heuristics.

    Args:
        text: The user's utterance.

    Returns:
        True if likely a question.
    """
    text = text.strip().lower()

    # Ends with question mark
    if text.endswith('?'):
        return True

    # Starts with question words
    question_starters = (
        'who', 'what', 'when', 'where', 'why', 'how',
        'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
        'can', 'do', 'does', 'did', 'have', 'has', 'had',
    )
    first_word = text.split()[0] if text.split() else ''
    if first_word in question_starters:
        return True

    return False
