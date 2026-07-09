"""LLM backends: a single interface with interchangeable implementations."""

from backends.base import LLMBackend, LLMResponse, ToolCall
from backends.dummy import DummyExplorerBackend, DummyRemoverBackend
from backends.openai_compat import OpenAICompatibleBackend

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "ToolCall",
    "DummyExplorerBackend",
    "DummyRemoverBackend",
    "OpenAICompatibleBackend",
]
