"""The single model-backend interface.

A backend takes OpenAI-style `messages` and `tools` and returns an
`LLMResponse`: the normalised tool calls plus the raw assistant message to
append back into the conversation (so history stays valid for hosted APIs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class ContextLengthExceeded(RuntimeError):
    """The conversation no longer fits in the model's context window.

    Raised by a backend instead of a generic error so the runner can end the
    episode as "context_exhausted". This must be its own outcome: an agent that
    ran out of context did not *decide* anything, and folding it into the
    behavioural endings would make "restraint collapsed on a big maze" and "the
    model lost its own exploration history on a big maze" the same row in the
    data.
    """


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    # Set when the backend could not parse the model's argument JSON. The
    # runner records such a turn as "malformed_arguments" instead of letting it
    # masquerade as a call with no arguments.
    parse_error: str | None = None


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
    # The provider's token accounting for this turn (prompt/completion/total),
    # verbatim, when it reports one. Recorded per step so the approach to the
    # context ceiling is visible before the episode hits it.
    usage: dict | None = None
    # Why the model stopped ("stop"/"tool_calls"/"length"/...). Recorded so a
    # response cut off by the token cap is visible as truncation rather than
    # surfacing downstream as an agent that simply failed to act.
    finish_reason: str | None = None


@runtime_checkable
class LLMBackend(Protocol):
    def reset(self) -> None:
        """Called once at the start of each episode."""

    def step(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Produce the next action given the conversation so far."""
