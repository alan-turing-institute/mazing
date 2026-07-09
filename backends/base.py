"""The single model-backend interface.

A backend takes OpenAI-style `messages` and `tools` and returns an
`LLMResponse`: the normalised tool calls plus the raw assistant message to
append back into the conversation (so history stays valid for hosted APIs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    tool_calls: list[ToolCall]
    # The assistant message exactly as it should be appended to the running
    # conversation (OpenAI schema, including the tool_calls array).
    assistant_message: dict
    text: str | None = None
    # The model's chain-of-thought, when the backend exposes one separately
    # (e.g. Ollama / reasoning models return it in a `reasoning` field).
    reasoning: str | None = None


@runtime_checkable
class LLMBackend(Protocol):
    def reset(self) -> None:
        """Called once at the start of each episode."""

    def step(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Produce the next action given the conversation so far."""
