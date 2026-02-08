# Test Coverage Improvement Plan

**Generated**: February 9, 2026  
**Current Overall Coverage**: 57%  
**Target Coverage**: 80%+

## Executive Summary

The codebase has strong coverage in core modules like `turn_controller.py` (100%) and the Wyoming TTS/STT providers (~90%), but critical interface modules between ASR, LLM, and TTS are severely undertested. This plan prioritizes testing the integration points that are most likely to cause production failures.

---

## Current Coverage by Module

### ✅ Well-Tested (>80%)

| Module | Coverage | Status |
|--------|----------|--------|
| `turn_controller.py` | 100% | Excellent |
| `wyoming_tts.py` | 93% | Good |
| `tool_formatter.py` | 92% | Good |
| `transcript_stabilizer.py` | 92% | Good |
| `tools/registry.py` | 91% | Good |
| `interfaces/events.py` | 90% | Good |
| `interfaces/tts.py` | 86% | Good |
| `interfaces/stt.py` | 82% | Good |
| `perf.py` | 81% | Good |
| `tools/base.py` | 80% | Good |

### ⚠️ Needs Improvement (40-80%)

| Module | Coverage | Priority |
|--------|----------|----------|
| `interfaces/llm.py` | 79% | Medium |
| `handlers.py` | 76% | High |
| `streaming_pipeline.py` | 76% | Medium |
| `latency.py` | 74% | Low |
| `wyoming_stt.py` | 73% | Medium |
| `phrase_chunker.py` | 61% | Low |
| `providers/__init__.py` | 60% | Low |
| `config.py` | 55% | Low |

### 🔴 Critical Gaps (<40%)

| Module | Coverage | Priority |
|--------|----------|----------|
| `streaming_asr.py` | 38% | High |
| `sherpa_nemotron.py` | 32% | Medium |
| `logging_config.py` | 23% | Low |
| `nemotron_stt.py` | 20% | Medium |
| `mumble_events.py` | 18% | High |
| `pipeline.py` | 17% | High |
| `openai_llm.py` | 13% | **Critical** |
| `tools/souls.py` | 0% | Medium |
| `tools/web_search.py` | 0% | Medium |
| `wyoming_tts_server.py` | 0% | Low |

---

## Priority 1: Critical Interface Testing

### 1.1 `openai_llm.py` (13% → 80%)

**Why Critical**: This is the primary LLM provider used in production. Tool calling, streaming, and error handling are untested.

**Test Cases Needed**:

```python
# tests/test_openai_llm.py

class TestOpenAIChatLLM:
    """Test OpenAI-compatible LLM provider."""

    # Constructor and configuration
    def test_init_with_all_parameters(self): ...
    def test_init_minimal_parameters(self): ...
    def test_tool_formatter_auto_selection_openai(self): ...
    def test_tool_formatter_auto_selection_lfm25(self): ...

    # Header and request building
    def test_build_headers_with_api_key(self): ...
    def test_build_headers_without_api_key(self): ...
    def test_build_messages_with_system_prompt(self): ...
    def test_build_messages_without_system_prompt(self): ...
    def test_build_messages_with_tools_openai_style(self): ...
    def test_build_messages_with_tools_lfm25_style(self): ...

    # Chat completion (non-streaming)
    def test_chat_simple_response(self): ...
    def test_chat_with_tools_no_tool_call(self): ...
    def test_chat_with_tool_call_openai_format(self): ...
    def test_chat_with_tool_call_lfm25_format(self): ...
    def test_chat_multiple_tool_calls(self): ...
    def test_chat_http_error_handling(self): ...
    def test_chat_timeout_handling(self): ...
    def test_chat_invalid_json_response(self): ...

    # Streaming
    def test_chat_stream_basic(self): ...
    def test_chat_stream_with_tool_calls(self): ...
    def test_chat_stream_connection_error(self): ...

    # Tool result formatting
    def test_format_tool_result_openai(self): ...
    def test_format_tool_result_lfm25(self): ...


class TestOpenAILLMIntegration:
    """Integration tests with mocked HTTP."""

    def test_full_conversation_with_tool_use(self): ...
    def test_conversation_history_maintained(self): ...
```

**Estimated Effort**: 4-6 hours

---

### 1.2 `mumble_events.py` (18% → 70%)

**Why Critical**: This is the interface between the bot and Mumble server. Event handling failures cause silent drops.

**Test Cases Needed**:

```python
# tests/test_mumble_events.py

class TestMumbleEventAdapter:
    """Test Mumble event adapter."""

    # Initialization
    def test_init_creates_event_mappings(self): ...
    def test_register_handler(self): ...
    def test_unregister_handler(self): ...

    # Event conversion
    def test_convert_user_connected_event(self): ...
    def test_convert_user_disconnected_event(self): ...
    def test_convert_text_message_event(self): ...
    def test_convert_audio_received_event(self): ...
    def test_convert_channel_joined_event(self): ...

    # Event dispatching
    def test_dispatch_to_single_handler(self): ...
    def test_dispatch_to_multiple_handlers(self): ...
    def test_dispatch_handler_exception_isolated(self): ...
    def test_dispatch_unknown_event_ignored(self): ...

    # Audio callbacks
    def test_audio_callback_invoked(self): ...
    def test_audio_callback_with_user_context(self): ...


class TestMumbleConnectionLifecycle:
    """Test connection state management."""

    def test_connect_success(self): ...
    def test_connect_failure_retry(self): ...
    def test_disconnect_cleanup(self): ...
    def test_reconnect_after_drop(self): ...
```

**Estimated Effort**: 3-4 hours

---

### 1.3 `pipeline.py` (17% → 60%)

**Why Critical**: Orchestrates the full ASR → LLM → TTS flow. Integration failures here break the entire bot.

**Test Cases Needed**:

```python
# tests/test_pipeline.py

class TestVoicePipeline:
    """Test the full voice pipeline."""

    # Pipeline initialization
    def test_init_with_all_providers(self): ...
    def test_init_missing_stt_raises(self): ...
    def test_init_missing_tts_raises(self): ...

    # Audio processing
    def test_process_audio_triggers_stt(self): ...
    def test_process_audio_silence_ignored(self): ...
    def test_process_audio_vad_detection(self): ...

    # Transcription handling
    def test_transcription_triggers_llm(self): ...
    def test_empty_transcription_ignored(self): ...
    def test_transcription_with_context(self): ...

    # LLM response handling
    def test_llm_response_triggers_tts(self): ...
    def test_llm_tool_call_executed(self): ...
    def test_llm_error_graceful_recovery(self): ...

    # TTS output
    def test_tts_audio_sent_to_mumble(self): ...
    def test_tts_chunked_for_streaming(self): ...

    # Interruption handling
    def test_barge_in_cancels_tts(self): ...
    def test_barge_in_restarts_listening(self): ...


class TestPipelineIntegration:
    """End-to-end pipeline tests with mocks."""

    def test_full_conversation_turn(self): ...
    def test_multi_turn_conversation(self): ...
    def test_tool_use_in_conversation(self): ...
```

**Estimated Effort**: 4-5 hours

---

## Priority 2: Provider Testing

### 2.1 `streaming_asr.py` (38% → 70%)

**Test Cases Needed**:

```python
class TestStreamingASR:
    def test_start_stream(self): ...
    def test_feed_audio_chunk(self): ...
    def test_get_partial_transcript(self): ...
    def test_get_final_transcript(self): ...
    def test_end_stream(self): ...
    def test_stream_timeout(self): ...
    def test_reconnect_on_error(self): ...
```

**Estimated Effort**: 2-3 hours

---

### 2.2 `sherpa_nemotron.py` (32% → 60%)

**Test Cases Needed**:

```python
class TestSherpaNemotron:
    def test_init_loads_model(self): ...
    def test_transcribe_audio(self): ...
    def test_streaming_transcription(self): ...
    def test_model_not_found_error(self): ...
```

**Estimated Effort**: 2 hours

---

### 2.3 `nemotron_stt.py` (20% → 60%)

**Test Cases Needed**:

```python
class TestNemotronSTT:
    def test_transcribe_audio_bytes(self): ...
    def test_transcribe_with_language(self): ...
    def test_streaming_mode(self): ...
    def test_connection_error_handling(self): ...
```

**Estimated Effort**: 2 hours

---

## Priority 3: Tool Testing

### 3.1 `tools/web_search.py` (0% → 80%)

**Test Cases Needed**:

```python
# tests/test_web_search.py

class TestWebSearchTool:
    def test_tool_definition(self): ...
    def test_execute_returns_results(self): ...
    def test_execute_empty_query(self): ...
    def test_execute_api_error(self): ...
    def test_execute_rate_limit(self): ...
    def test_result_formatting(self): ...


class TestWebSearchIntegration:
    def test_tool_registered_in_registry(self): ...
    def test_llm_can_invoke_tool(self): ...
```

**Estimated Effort**: 2 hours

---

### 3.2 `tools/souls.py` (0% → 80%)

**Test Cases Needed**:

```python
# tests/test_souls_tool.py

class TestSoulsTool:
    def test_tool_definition(self): ...
    def test_list_available_souls(self): ...
    def test_switch_soul_success(self): ...
    def test_switch_soul_not_found(self): ...
    def test_get_current_soul(self): ...


class TestSoulsToolIntegration:
    def test_tool_registered_in_registry(self): ...
    def test_personality_change_persists(self): ...
```

**Estimated Effort**: 2 hours

---

## Priority 4: Supporting Modules

### 4.1 `handlers.py` (76% → 90%)

**Missing Coverage** (lines 97-99, 132-135, 150-159, etc.):

```python
class TestTextCommandHandler:
    def test_command_parsing(self): ...
    def test_unknown_command_response(self): ...
    def test_help_command(self): ...

class TestVoiceHandler:
    def test_audio_received_processing(self): ...
    def test_silence_detection(self): ...
```

**Estimated Effort**: 1-2 hours

---

### 4.2 `config.py` (55% → 75%)

**Missing Coverage** (lines 257-280, 412-465, etc.):

```python
class TestConfigLoading:
    def test_load_from_yaml(self): ...
    def test_load_with_env_override(self): ...
    def test_missing_required_field(self): ...
    def test_default_values(self): ...

class TestSoulConfig:
    def test_load_soul_yaml(self): ...
    def test_soul_with_voice_weights(self): ...
```

**Estimated Effort**: 1-2 hours

---

## Dead Code Cleanup

The following unused imports should be removed:

```python
# mumble_voice_bot/handlers.py:27
- from somewhere import MumbleTTSBot  # unused

# mumble_voice_bot/providers/mumble_events.py:33
- from pymumble import Mumble  # unused

# mumble_voice_bot/providers/streaming_asr.py:46
- import websockets  # unused
```

**Estimated Effort**: 15 minutes

---

## Implementation Schedule

### Week 1: Critical Interfaces
- [ ] `openai_llm.py` tests (4-6 hours)
- [ ] `mumble_events.py` tests (3-4 hours)
- [ ] Dead code cleanup (15 min)

### Week 2: Pipeline & Providers
- [ ] `pipeline.py` tests (4-5 hours)
- [ ] `streaming_asr.py` tests (2-3 hours)

### Week 3: Tools & Polish
- [ ] `tools/web_search.py` tests (2 hours)
- [ ] `tools/souls.py` tests (2 hours)
- [ ] `handlers.py` gap filling (1-2 hours)
- [ ] `config.py` gap filling (1-2 hours)

### Week 4: Provider Completion
- [ ] `sherpa_nemotron.py` tests (2 hours)
- [ ] `nemotron_stt.py` tests (2 hours)

---

## Testing Patterns to Follow

### 1. Mock External Services

```python
@pytest.fixture
def mock_httpx_client():
    """Mock HTTP client for LLM/API tests."""
    with patch('httpx.AsyncClient') as mock:
        yield mock
```

### 2. Use Fixtures for Common Setup

```python
@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }
    ]
```

### 3. Test Both Success and Error Paths

```python
def test_chat_success(self, mock_client):
    """Test successful chat completion."""
    ...

def test_chat_http_error(self, mock_client):
    """Test HTTP error is handled gracefully."""
    mock_client.post.side_effect = httpx.HTTPError("Connection failed")
    ...

def test_chat_timeout(self, mock_client):
    """Test timeout is handled gracefully."""
    mock_client.post.side_effect = httpx.TimeoutException("Timeout")
    ...
```

### 4. Test Tool Formatting for Both Providers

```python
@pytest.mark.parametrize("model,formatter_type", [
    ("gpt-4", OpenAIToolFormatter),
    ("liquid/lfm-2.5-1.2b-instruct:free", LFM25ToolFormatter),
])
def test_tool_formatter_selection(self, model, formatter_type):
    llm = OpenAIChatLLM(endpoint="http://test", model=model)
    assert isinstance(llm.tool_formatter, formatter_type)
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Overall Coverage | 57% | 80% |
| Interface Modules | ~50% | 85% |
| Provider Modules | ~35% | 70% |
| Tool Modules | 45% | 80% |
| Critical Path (ASR→LLM→TTS) | ~20% | 75% |

---

## Running Coverage

```bash
# Full coverage report
uv run pytest --cov=mumble_voice_bot --cov-report=term-missing

# Coverage for specific module
uv run pytest --cov=mumble_voice_bot/providers/openai_llm --cov-report=term-missing tests/test_openai_llm.py

# HTML report
uv run pytest --cov=mumble_voice_bot --cov-report=html
open htmlcov/index.html
```

---

## Notes

- Skip integration tests that require real services with `@pytest.mark.skip` or environment checks
- Use `pytest-asyncio` for async test methods
- Consider adding `pytest-timeout` to catch hanging tests
- The `wyoming_tts_server.py` (0%) is low priority as it's a standalone server component
