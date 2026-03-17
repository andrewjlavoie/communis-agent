"""Unit tests for Gemini conversion functions (no API key needed).

Tests the pure conversion helpers in activities/llm_activities that transform
between our internal Anthropic-format messages and Gemini SDK types.

Run:
    uv sync --extra gemini --extra dev
    uv run pytest tests/test_gemini_conversion.py -v
"""
from __future__ import annotations

import pytest

# These imports will fail if google-genai is not installed
genai = pytest.importorskip("google.genai", reason="google-genai not installed (uv sync --extra gemini)")


class TestConvertMessagesToGemini:
    """Test _convert_messages_to_gemini."""

    def test_plain_text_user_message(self):
        from activities.llm_activities import _convert_messages_to_gemini

        messages = [{"role": "user", "content": "Hello"}]
        result = _convert_messages_to_gemini(messages)

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].parts[0].text == "Hello"

    def test_assistant_role_mapped_to_model(self):
        from activities.llm_activities import _convert_messages_to_gemini

        messages = [{"role": "assistant", "content": "Hi there"}]
        result = _convert_messages_to_gemini(messages)

        assert len(result) == 1
        assert result[0].role == "model"
        assert result[0].parts[0].text == "Hi there"

    def test_tool_use_block_to_function_call(self):
        from activities.llm_activities import _convert_messages_to_gemini

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search for that."},
                    {
                        "type": "tool_use",
                        "id": "tu_123",
                        "name": "search_flights",
                        "input": {"origin": "SFO", "destination": "JFK"},
                    },
                ],
            }
        ]
        result = _convert_messages_to_gemini(messages)

        assert len(result) == 1
        assert result[0].role == "model"
        parts = result[0].parts
        assert len(parts) == 2
        assert parts[0].text == "I'll search for that."
        assert parts[1].function_call.name == "search_flights"
        assert dict(parts[1].function_call.args) == {"origin": "SFO", "destination": "JFK"}

    def test_tool_result_block_to_function_response(self):
        from activities.llm_activities import _convert_messages_to_gemini

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_123",
                        "name": "search_flights",
                        "input": {"origin": "SFO"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_123",
                        "content": '{"flights": []}',
                    },
                ],
            },
        ]
        result = _convert_messages_to_gemini(messages)

        assert len(result) == 2
        # Second message should be a function response
        response_part = result[1].parts[0]
        assert response_part.function_response.name == "search_flights"
        assert response_part.function_response.response == {"result": '{"flights": []}'}

    def test_multi_turn_conversation(self):
        from activities.llm_activities import _convert_messages_to_gemini

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Search for flights"},
        ]
        result = _convert_messages_to_gemini(messages)

        assert len(result) == 3
        assert result[0].role == "user"
        assert result[1].role == "model"
        assert result[2].role == "user"


class TestConvertToolsToGemini:
    """Test _convert_tools_to_gemini."""

    def test_single_tool(self):
        from activities.llm_activities import _convert_tools_to_gemini

        tools = [
            {
                "name": "search_flights",
                "description": "Search for flights",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["origin", "destination"],
                },
            }
        ]
        result = _convert_tools_to_gemini(tools)

        assert len(result.function_declarations) == 1
        decl = result.function_declarations[0]
        assert decl.name == "search_flights"
        assert decl.description == "Search for flights"

    def test_multiple_tools(self):
        from activities.llm_activities import _convert_tools_to_gemini

        tools = [
            {"name": "tool_a", "description": "Tool A", "input_schema": {"type": "object", "properties": {}}},
            {"name": "tool_b", "description": "Tool B", "input_schema": {"type": "object", "properties": {}}},
        ]
        result = _convert_tools_to_gemini(tools)

        assert len(result.function_declarations) == 2
        names = [d.name for d in result.function_declarations]
        assert "tool_a" in names
        assert "tool_b" in names


class TestFindToolNameForId:
    """Test _find_tool_name_for_id."""

    def test_finds_name_in_previous_message(self):
        from activities.llm_activities import _find_tool_name_for_id

        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_abc", "name": "search_flights", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_abc", "content": "result"},
                ],
            },
        ]
        assert _find_tool_name_for_id(messages, "tu_abc") == "search_flights"

    def test_finds_correct_name_among_multiple(self):
        from activities.llm_activities import _find_tool_name_for_id

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "tool_a", "input": {}},
                    {"type": "tool_use", "id": "tu_2", "name": "tool_b", "input": {}},
                ],
            },
        ]
        assert _find_tool_name_for_id(messages, "tu_2") == "tool_b"

    def test_returns_unknown_when_not_found(self):
        from activities.llm_activities import _find_tool_name_for_id

        messages = [{"role": "user", "content": "hello"}]
        assert _find_tool_name_for_id(messages, "nonexistent") == "unknown"

    def test_scans_backwards(self):
        """Should find the most recent occurrence when IDs hypothetically repeat."""
        from activities.llm_activities import _find_tool_name_for_id

        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_x", "name": "old_tool", "input": {}},
                ],
            },
            {"role": "user", "content": "something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_x", "name": "new_tool", "input": {}},
                ],
            },
        ]
        # Scanning backwards should find "new_tool" first
        assert _find_tool_name_for_id(messages, "tu_x") == "new_tool"
