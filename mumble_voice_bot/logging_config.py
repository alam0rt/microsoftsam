"""Structured logging configuration for the Mumble voice bot.

Provides consistent, structured logging with:
- JSON output for production (machine-parseable)
- Human-readable output for development
- Latency and event tracking
- Context-aware log records
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
        if hasattr(record, "user"):
            log_data["user"] = record.user
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, "component"):
            log_data["component"] = record.component
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter with colors for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Color the level name
        color = self.COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:8}{self.RESET}"

        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Build the base message
        msg = f"{timestamp} {level} [{record.name}] {record.getMessage()}"

        # Add context if present
        context_parts = []
        if hasattr(record, "user"):
            context_parts.append(f"user={record.user}")
        if hasattr(record, "latency_ms"):
            context_parts.append(f"latency={record.latency_ms:.0f}ms")
        if hasattr(record, "duration_ms"):
            context_parts.append(f"duration={record.duration_ms:.0f}ms")

        if context_parts:
            msg += f" ({', '.join(context_parts)})"

        return msg


class BotLogger(logging.LoggerAdapter):
    """Logger adapter with convenience methods for structured logging."""

    def event(
        self,
        event_type: str,
        message: str,
        user: str | None = None,
        session_id: int | None = None,
        **kwargs: Any,
    ):
        """Log an event with structured data."""
        extra = {
            "event_type": event_type,
            **kwargs,
        }
        if user:
            extra["user"] = user
        if session_id:
            extra["session_id"] = session_id
        self.info(message, extra=extra)

    def latency(
        self,
        component: str,
        latency_ms: float,
        message: str | None = None,
        **kwargs: Any,
    ):
        """Log a latency measurement."""
        msg = message or f"{component} completed"
        extra = {
            "component": component,
            "latency_ms": latency_ms,
            **kwargs,
        }
        self.info(msg, extra=extra)

    def asr(
        self,
        user: str,
        transcript: str,
        duration_ms: float,
        latency_ms: float | None = None,
    ):
        """Log an ASR result."""
        extra: dict[str, Any] = {
            "component": "asr",
            "user": user,
            "duration_ms": duration_ms,
            "transcript_length": len(transcript),
        }
        if latency_ms:
            extra["latency_ms"] = latency_ms

        preview = transcript[:50] + "..." if len(transcript) > 50 else transcript
        self.info(f'ASR: "{preview}"', extra=extra)

    def llm(
        self,
        prompt_length: int,
        response_length: int,
        latency_ms: float,
        model: str | None = None,
    ):
        """Log an LLM completion."""
        extra: dict[str, Any] = {
            "component": "llm",
            "latency_ms": latency_ms,
            "prompt_length": prompt_length,
            "response_length": response_length,
        }
        if model:
            extra["model"] = model
        self.info(f"LLM response ({response_length} chars)", extra=extra)

    def tts(
        self,
        text_length: int,
        audio_duration_ms: float,
        latency_ms: float,
    ):
        """Log a TTS synthesis."""
        extra = {
            "component": "tts",
            "latency_ms": latency_ms,
            "text_length": text_length,
            "audio_duration_ms": audio_duration_ms,
        }
        self.info(f"TTS synthesized {audio_duration_ms:.0f}ms audio", extra=extra)

    def turn_complete(
        self,
        user: str,
        total_latency_ms: float,
        asr_ms: float,
        llm_ms: float,
        tts_ms: float,
    ):
        """Log a complete voice turn with breakdown."""
        extra = {
            "event_type": "turn_complete",
            "user": user,
            "latency_ms": total_latency_ms,
            "extra_data": {
                "asr_ms": asr_ms,
                "llm_ms": llm_ms,
                "tts_ms": tts_ms,
            },
        }
        self.info(
            f"Turn complete: TTFA={total_latency_ms:.0f}ms "
            f"(ASR={asr_ms:.0f}ms, LLM={llm_ms:.0f}ms, TTS={tts_ms:.0f}ms)",
            extra=extra,
        )


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure logging for the bot.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_output: Use JSON format (for production/parsing).
        log_file: Optional file to write logs to.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_output:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    # File handler (always JSON for parsing)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def get_logger(name: str) -> BotLogger:
    """Get a structured logger for a component.

    Args:
        name: Logger name (typically __name__).

    Returns:
        BotLogger with structured logging methods.
    """
    return BotLogger(logging.getLogger(name), {})
