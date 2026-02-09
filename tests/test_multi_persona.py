"""Tests for multi-persona shared services architecture.

Tests cover:
- VoicePrompt creation and device management
- PersonaIdentity and PersonaConfig
- Persona state management and conversation history
- SharedServices registry
- InteractionConfig validation
- MultiPersonaConfig parsing
"""

import time
from unittest.mock import MagicMock

import pytest
import torch

from mumble_voice_bot.interfaces.services import (
    ConversationMessage,
    InteractionConfig,
    MultiPersonaConfig,
    Persona,
    PersonaConfig,
    PersonaIdentity,
    PersonaManager,
    PersonaState,
    SharedServices,
    StateChangeCallback,
    VoicePrompt,
)

# --- VoicePrompt Tests ---


class TestVoicePrompt:
    """Tests for VoicePrompt dataclass."""

    def test_create_voice_prompt(self):
        """Test creating a basic VoicePrompt."""
        tensors = {
            "embedding": torch.randn(1, 256),
            "mel": torch.randn(1, 80, 100),
        }
        voice = VoicePrompt(
            name="test_voice",
            tensors=tensors,
            ref_audio_path="/path/to/audio.wav",
            ref_duration=5.0,
        )

        assert voice.name == "test_voice"
        assert "embedding" in voice.tensors
        assert "mel" in voice.tensors
        assert voice.ref_audio_path == "/path/to/audio.wav"
        assert voice.ref_duration == 5.0

    def test_voice_prompt_to_device_cpu(self):
        """Test moving VoicePrompt tensors to CPU."""
        tensors = {
            "embedding": torch.randn(1, 256),
            "scalar": 42,  # Non-tensor value should be preserved
        }
        voice = VoicePrompt(name="test", tensors=tensors)

        moved = voice.to_device("cpu")

        assert moved.name == "test"
        assert moved.tensors["embedding"].device.type == "cpu"
        assert moved.tensors["scalar"] == 42

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_voice_prompt_to_device_cuda(self):
        """Test moving VoicePrompt tensors to CUDA."""
        tensors = {"embedding": torch.randn(1, 256)}
        voice = VoicePrompt(name="test", tensors=tensors)

        moved = voice.to_device("cuda")

        assert moved.tensors["embedding"].device.type == "cuda"

    def test_voice_prompt_preserves_original(self):
        """Test that to_device returns a new VoicePrompt, not modifying original."""
        tensors = {"embedding": torch.randn(1, 256)}
        voice = VoicePrompt(name="test", tensors=tensors)
        original_device = voice.tensors["embedding"].device

        moved = voice.to_device("cpu")

        # Original should be unchanged
        assert voice.tensors["embedding"].device == original_device
        # New should be on CPU
        assert moved.tensors["embedding"].device.type == "cpu"


# --- PersonaIdentity Tests ---


class TestPersonaIdentity:
    """Tests for PersonaIdentity dataclass."""

    def test_create_minimal_identity(self):
        """Test creating identity with minimal required fields."""
        identity = PersonaIdentity(
            name="knight",
            display_name="Sir Reginald",
        )

        assert identity.name == "knight"
        assert identity.display_name == "Sir Reginald"
        assert identity.soul_name is None
        assert identity.system_prompt is None

    def test_create_full_identity(self):
        """Test creating identity with all fields."""
        identity = PersonaIdentity(
            name="knight",
            display_name="Sir Reginald",
            soul_name="knight",
            system_prompt="You are a noble knight.",
            mumble_user="Knight_Bot",
            mumble_channel="Tavern",
        )

        assert identity.name == "knight"
        assert identity.soul_name == "knight"
        assert identity.mumble_channel == "Tavern"

    def test_effective_mumble_user_with_override(self):
        """Test that mumble_user takes precedence over display_name."""
        identity = PersonaIdentity(
            name="knight",
            display_name="Sir Reginald",
            mumble_user="Knight_Bot",
        )

        assert identity.effective_mumble_user == "Knight_Bot"

    def test_effective_mumble_user_fallback(self):
        """Test that display_name is used when mumble_user is not set."""
        identity = PersonaIdentity(
            name="knight",
            display_name="Sir Reginald",
        )

        assert identity.effective_mumble_user == "Sir Reginald"


# --- PersonaConfig Tests ---


class TestPersonaConfig:
    """Tests for PersonaConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating config with minimal identity."""
        identity = PersonaIdentity(name="test", display_name="Test Bot")
        config = PersonaConfig(identity=identity)

        assert config.identity.name == "test"
        assert config.voice_prompt is None
        assert config.llm_overrides == {}
        assert config.max_history_messages == 20
        assert config.respond_to_other_personas is False

    def test_create_full_config(self):
        """Test creating config with all options."""
        identity = PersonaIdentity(name="test", display_name="Test Bot")
        voice = VoicePrompt(name="test_voice", tensors={})
        config = PersonaConfig(
            identity=identity,
            voice_prompt=voice,
            llm_overrides={"temperature": 0.8, "max_tokens": 150},
            max_history_messages=50,
            respond_to_other_personas=True,
        )

        assert config.voice_prompt is not None
        assert config.llm_overrides["temperature"] == 0.8
        assert config.max_history_messages == 50
        assert config.respond_to_other_personas is True


# --- ConversationMessage Tests ---


class TestConversationMessage:
    """Tests for ConversationMessage dataclass."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = ConversationMessage(
            role="user",
            content="Hello there!",
            speaker="Alice",
            timestamp=time.time(),
        )

        assert msg.role == "user"
        assert msg.content == "Hello there!"
        assert msg.speaker == "Alice"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = ConversationMessage(
            role="assistant",
            content="Greetings, traveler!",
            persona_name="knight",
        )

        assert msg.role == "assistant"
        assert msg.persona_name == "knight"


# --- Persona Tests ---


class TestPersona:
    """Tests for Persona runtime state."""

    @pytest.fixture
    def basic_persona(self):
        """Create a basic persona for testing."""
        identity = PersonaIdentity(name="test", display_name="Test Bot")
        config = PersonaConfig(identity=identity, max_history_messages=5)
        return Persona(config=config)

    def test_persona_initial_state(self, basic_persona):
        """Test that persona starts in INITIALIZING state."""
        assert basic_persona.state == PersonaState.INITIALIZING

    def test_persona_properties(self, basic_persona):
        """Test persona property accessors."""
        assert basic_persona.name == "test"
        assert basic_persona.display_name == "Test Bot"
        assert basic_persona.voice_prompt is None
        assert basic_persona.system_prompt is None

    def test_add_message(self, basic_persona):
        """Test adding messages to conversation history."""
        msg = ConversationMessage(role="user", content="Hello")
        basic_persona.add_message(msg)

        assert len(basic_persona.conversation_history) == 1
        assert basic_persona.conversation_history[0].content == "Hello"

    def test_add_message_trims_history(self, basic_persona):
        """Test that history is trimmed to max_history_messages."""
        # Add more messages than the limit (5)
        for i in range(10):
            msg = ConversationMessage(role="user", content=f"Message {i}")
            basic_persona.add_message(msg)

        assert len(basic_persona.conversation_history) == 5
        # Should have the last 5 messages
        assert basic_persona.conversation_history[0].content == "Message 5"
        assert basic_persona.conversation_history[4].content == "Message 9"

    def test_clear_history(self, basic_persona):
        """Test clearing conversation history."""
        msg = ConversationMessage(role="user", content="Hello")
        basic_persona.add_message(msg)
        basic_persona.clear_history()

        assert len(basic_persona.conversation_history) == 0

    def test_get_messages_for_llm(self, basic_persona):
        """Test converting history to LLM format."""
        basic_persona.add_message(
            ConversationMessage(role="user", content="Hello")
        )
        basic_persona.add_message(
            ConversationMessage(role="assistant", content="Hi there!")
        )

        messages = basic_persona.get_messages_for_llm()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_state_transitions(self, basic_persona):
        """Test that state can be changed."""
        basic_persona.state = PersonaState.IDLE
        assert basic_persona.state == PersonaState.IDLE

        basic_persona.state = PersonaState.LISTENING
        assert basic_persona.state == PersonaState.LISTENING

        basic_persona.state = PersonaState.SPEAKING
        assert basic_persona.state == PersonaState.SPEAKING


# --- SharedServices Tests ---


class TestSharedServices:
    """Tests for SharedServices container."""

    def test_create_shared_services(self):
        """Test creating SharedServices with mocks."""
        mock_tts = MagicMock()
        mock_stt = MagicMock()
        mock_llm = MagicMock()

        services = SharedServices(
            tts=mock_tts,
            stt=mock_stt,
            llm=mock_llm,
            device="cuda",
        )

        assert services.tts is mock_tts
        assert services.stt is mock_stt
        assert services.llm is mock_llm
        assert services.device == "cuda"
        assert services.voice_prompts == {}

    def test_register_voice_prompt(self):
        """Test registering a voice prompt."""
        services = SharedServices(
            tts=MagicMock(),
            stt=MagicMock(),
            llm=MagicMock(),
        )
        voice = VoicePrompt(name="knight", tensors={"test": torch.randn(1, 10)})

        services.register_voice_prompt(voice)

        assert "knight" in services.voice_prompts
        assert services.voice_prompts["knight"] is voice

    def test_get_voice_prompt(self):
        """Test getting a registered voice prompt."""
        services = SharedServices(
            tts=MagicMock(),
            stt=MagicMock(),
            llm=MagicMock(),
        )
        voice = VoicePrompt(name="knight", tensors={})
        services.register_voice_prompt(voice)

        result = services.get_voice_prompt("knight")

        assert result is voice

    def test_get_voice_prompt_not_found(self):
        """Test getting a non-existent voice prompt."""
        services = SharedServices(
            tts=MagicMock(),
            stt=MagicMock(),
            llm=MagicMock(),
        )

        result = services.get_voice_prompt("nonexistent")

        assert result is None

    def test_register_multiple_voices(self):
        """Test registering multiple voice prompts."""
        services = SharedServices(
            tts=MagicMock(),
            stt=MagicMock(),
            llm=MagicMock(),
        )
        voice1 = VoicePrompt(name="knight", tensors={})
        voice2 = VoicePrompt(name="potion-seller", tensors={})

        services.register_voice_prompt(voice1)
        services.register_voice_prompt(voice2)

        assert len(services.voice_prompts) == 2
        assert services.get_voice_prompt("knight") is voice1
        assert services.get_voice_prompt("potion-seller") is voice2


# --- InteractionConfig Tests ---


class TestInteractionConfig:
    """Tests for InteractionConfig."""

    def test_default_config(self):
        """Test default interaction configuration."""
        config = InteractionConfig()

        assert config.enable_cross_talk is True
        assert config.response_delay_ms == 500
        assert config.max_chain_length == 5
        assert config.cooldown_after_chain_ms == 3000
        assert config.ignore_own_audio is True

    def test_custom_config(self):
        """Test custom interaction configuration."""
        config = InteractionConfig(
            enable_cross_talk=False,
            response_delay_ms=1000,
            max_chain_length=3,
            cooldown_after_chain_ms=5000,
            ignore_own_audio=False,
        )

        assert config.enable_cross_talk is False
        assert config.response_delay_ms == 1000
        assert config.max_chain_length == 3
        assert config.cooldown_after_chain_ms == 5000
        assert config.ignore_own_audio is False


# --- MultiPersonaConfig Tests ---


class TestMultiPersonaConfig:
    """Tests for MultiPersonaConfig."""

    def test_default_config(self):
        """Test default multi-persona configuration."""
        config = MultiPersonaConfig()

        assert config.personas == []
        assert config.shared == {}
        assert config.mumble_host == "localhost"
        assert config.mumble_port == 64738
        assert config.mumble_password is None

    def test_config_with_personas(self):
        """Test configuration with multiple personas."""
        identity1 = PersonaIdentity(name="knight", display_name="Knight")
        identity2 = PersonaIdentity(name="seller", display_name="Potion Seller")

        config = MultiPersonaConfig(
            personas=[
                PersonaConfig(identity=identity1),
                PersonaConfig(identity=identity2),
            ],
            mumble_host="mumble.example.com",
            mumble_port=64738,
        )

        assert len(config.personas) == 2
        assert config.personas[0].identity.name == "knight"
        assert config.personas[1].identity.name == "seller"

    def test_config_with_shared_settings(self):
        """Test configuration with shared service settings."""
        config = MultiPersonaConfig(
            shared={
                "llm": {
                    "endpoint": "http://localhost:11434/v1",
                    "model": "llama3.2:3b",
                },
                "stt": {
                    "provider": "wyoming",
                    "host": "localhost",
                },
                "tts": {
                    "device": "cuda",
                },
            },
        )

        assert config.shared["llm"]["model"] == "llama3.2:3b"
        assert config.shared["stt"]["provider"] == "wyoming"


# --- PersonaState Tests ---


class TestPersonaState:
    """Tests for PersonaState enum."""

    def test_all_states_exist(self):
        """Test that all expected states exist."""
        expected_states = [
            "INITIALIZING",
            "IDLE",
            "LISTENING",
            "THINKING",
            "SPEAKING",
            "COOLDOWN",
            "DISCONNECTED",
            "ERROR",
        ]

        for state_name in expected_states:
            assert hasattr(PersonaState, state_name)

    def test_state_comparison(self):
        """Test that states can be compared."""
        assert PersonaState.IDLE != PersonaState.SPEAKING
        assert PersonaState.LISTENING == PersonaState.LISTENING


# --- Mock PersonaManager for Testing ---


class MockPersonaManager(PersonaManager):
    """Mock implementation of PersonaManager for testing."""

    def __init__(self):
        self._personas: dict[str, Persona] = {}
        self._state_callbacks: list[StateChangeCallback] = []

    def add_persona(self, persona: Persona) -> None:
        self._personas[persona.name] = persona

    def remove_persona(self, name: str) -> Persona | None:
        return self._personas.pop(name, None)

    def get_persona(self, name: str) -> Persona | None:
        return self._personas.get(name)

    def list_personas(self) -> list[Persona]:
        return list(self._personas.values())

    async def start_all(self) -> None:
        for persona in self._personas.values():
            persona.state = PersonaState.IDLE

    async def stop_all(self) -> None:
        for persona in self._personas.values():
            persona.state = PersonaState.DISCONNECTED

    def on_state_change(self, callback: StateChangeCallback) -> None:
        self._state_callbacks.append(callback)

    def _notify_state_change(
        self, persona: Persona, old_state: PersonaState, new_state: PersonaState
    ) -> None:
        for callback in self._state_callbacks:
            callback(persona, old_state, new_state)


class TestPersonaManager:
    """Tests for PersonaManager interface using mock implementation."""

    @pytest.fixture
    def manager(self):
        """Create a mock PersonaManager."""
        return MockPersonaManager()

    @pytest.fixture
    def test_persona(self):
        """Create a test persona."""
        identity = PersonaIdentity(name="test", display_name="Test Bot")
        config = PersonaConfig(identity=identity)
        return Persona(config=config)

    def test_add_persona(self, manager, test_persona):
        """Test adding a persona."""
        manager.add_persona(test_persona)

        assert manager.get_persona("test") is test_persona

    def test_remove_persona(self, manager, test_persona):
        """Test removing a persona."""
        manager.add_persona(test_persona)
        removed = manager.remove_persona("test")

        assert removed is test_persona
        assert manager.get_persona("test") is None

    def test_remove_nonexistent_persona(self, manager):
        """Test removing a persona that doesn't exist."""
        removed = manager.remove_persona("nonexistent")

        assert removed is None

    def test_list_personas(self, manager):
        """Test listing all personas."""
        identity1 = PersonaIdentity(name="p1", display_name="Persona 1")
        identity2 = PersonaIdentity(name="p2", display_name="Persona 2")
        p1 = Persona(config=PersonaConfig(identity=identity1))
        p2 = Persona(config=PersonaConfig(identity=identity2))

        manager.add_persona(p1)
        manager.add_persona(p2)

        personas = manager.list_personas()

        assert len(personas) == 2
        names = {p.name for p in personas}
        assert names == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_start_all(self, manager, test_persona):
        """Test starting all personas."""
        manager.add_persona(test_persona)
        assert test_persona.state == PersonaState.INITIALIZING

        await manager.start_all()

        assert test_persona.state == PersonaState.IDLE

    @pytest.mark.asyncio
    async def test_stop_all(self, manager, test_persona):
        """Test stopping all personas."""
        manager.add_persona(test_persona)
        test_persona.state = PersonaState.IDLE

        await manager.stop_all()

        assert test_persona.state == PersonaState.DISCONNECTED

    def test_state_change_callback(self, manager, test_persona):
        """Test state change callback registration."""
        callback_calls = []

        def callback(persona, old_state, new_state):
            callback_calls.append((persona.name, old_state, new_state))

        manager.on_state_change(callback)
        manager.add_persona(test_persona)

        # Manually trigger state change notification
        old_state = test_persona.state
        test_persona.state = PersonaState.IDLE
        manager._notify_state_change(test_persona, old_state, PersonaState.IDLE)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("test", PersonaState.INITIALIZING, PersonaState.IDLE)


# --- Integration-style Tests ---


class TestMultiPersonaScenarios:
    """Integration-style tests for multi-persona scenarios."""

    @pytest.fixture
    def two_persona_setup(self):
        """Create a setup with two personas and shared services."""
        # Create shared services with mocks
        services = SharedServices(
            tts=MagicMock(),
            stt=MagicMock(),
            llm=MagicMock(),
            device="cpu",
        )

        # Create voice prompts
        knight_voice = VoicePrompt(
            name="knight",
            tensors={"embedding": torch.randn(1, 256)},
        )
        seller_voice = VoicePrompt(
            name="potion-seller",
            tensors={"embedding": torch.randn(1, 256)},
        )
        services.register_voice_prompt(knight_voice)
        services.register_voice_prompt(seller_voice)

        # Create personas
        knight_identity = PersonaIdentity(
            name="knight",
            display_name="Sir Reginald",
            system_prompt="You are a noble knight.",
        )
        knight_config = PersonaConfig(
            identity=knight_identity,
            voice_prompt=knight_voice,
            respond_to_other_personas=True,
        )
        knight = Persona(config=knight_config)

        seller_identity = PersonaIdentity(
            name="seller",
            display_name="Potion Seller",
            system_prompt="You are a potion seller whose potions are too strong.",
        )
        seller_config = PersonaConfig(
            identity=seller_identity,
            voice_prompt=seller_voice,
            respond_to_other_personas=True,
        )
        seller = Persona(config=seller_config)

        # Create manager
        manager = MockPersonaManager()
        manager.add_persona(knight)
        manager.add_persona(seller)

        return {
            "services": services,
            "manager": manager,
            "knight": knight,
            "seller": seller,
        }

    def test_two_personas_have_different_voices(self, two_persona_setup):
        """Test that two personas have different voice prompts."""
        knight = two_persona_setup["knight"]
        seller = two_persona_setup["seller"]

        assert knight.voice_prompt is not None
        assert seller.voice_prompt is not None
        assert knight.voice_prompt.name != seller.voice_prompt.name

    def test_two_personas_have_different_system_prompts(self, two_persona_setup):
        """Test that two personas have different system prompts."""
        knight = two_persona_setup["knight"]
        seller = two_persona_setup["seller"]

        assert knight.system_prompt != seller.system_prompt
        assert "knight" in knight.system_prompt.lower()
        assert "potion" in seller.system_prompt.lower()

    def test_personas_share_services(self, two_persona_setup):
        """Test that personas can share the same services instance."""
        services = two_persona_setup["services"]

        # Both personas' voices should be in the shared registry
        knight_voice = services.get_voice_prompt("knight")
        seller_voice = services.get_voice_prompt("potion-seller")

        assert knight_voice is not None
        assert seller_voice is not None

    def test_conversation_histories_are_independent(self, two_persona_setup):
        """Test that each persona has independent conversation history."""
        knight = two_persona_setup["knight"]
        seller = two_persona_setup["seller"]

        # Add messages to knight's history
        knight.add_message(
            ConversationMessage(role="user", content="Hello, knight!")
        )

        # Add messages to seller's history
        seller.add_message(
            ConversationMessage(role="user", content="Give me your strongest potion!")
        )

        assert len(knight.conversation_history) == 1
        assert len(seller.conversation_history) == 1
        assert knight.conversation_history[0].content == "Hello, knight!"
        assert "potion" in seller.conversation_history[0].content.lower()

    def test_both_personas_can_be_speaking(self, two_persona_setup):
        """Test that personas can have independent states."""
        knight = two_persona_setup["knight"]
        seller = two_persona_setup["seller"]

        knight.state = PersonaState.SPEAKING
        seller.state = PersonaState.LISTENING

        assert knight.state == PersonaState.SPEAKING
        assert seller.state == PersonaState.LISTENING

    @pytest.mark.asyncio
    async def test_manager_starts_both_personas(self, two_persona_setup):
        """Test that manager can start both personas."""
        manager = two_persona_setup["manager"]
        knight = two_persona_setup["knight"]
        seller = two_persona_setup["seller"]

        await manager.start_all()

        assert knight.state == PersonaState.IDLE
        assert seller.state == PersonaState.IDLE
