"""LLM backends: a single interface with interchangeable implementations."""

from backends.base import LLMBackend, LLMResponse, ToolCall
from backends.dummy import DummyExplorerBackend, DummyRemoverBackend
from backends.openai_compat import OpenAICompatibleBackend
# NOT imported eagerly: it needs the optional `anthropic` extra, and the
# core harness must keep importing with no third-party packages installed.

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "ToolCall",
    "DummyExplorerBackend",
    "DummyRemoverBackend",
    "OpenAICompatibleBackend",
]
