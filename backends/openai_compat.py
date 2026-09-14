"""OpenAI-compatible /chat/completions client with tool calling.

Works with hosted providers and local OpenAI-compatible servers (vLLM,
Ollama's OpenAI endpoint, LM Studio, ...). Uses only the standard library so
the harness has no runtime dependencies. Provider is never hard-coded — model,
base URL, and API key are all configurable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from backends.base import ContextLengthExceeded, LLMResponse, ToolCall


# Substrings that identify a context-window overflow in an error payload.
# There is no portable error code for this — OpenAI returns
# code="context_length_exceeded", vLLM and llama.cpp return a 400 whose message
# names the limit, Anthropic-compatible gateways say "prompt is too long" — so
# the wording is matched, lowercased, and the list is meant to grow.
_CONTEXT_ERROR_MARKERS = (
    "context_length_exceeded",
    "context length",
    "context window",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "reduce the length of the messages",
    "please reduce the length",
    "input is too long",
)


def is_context_length_error(text: str) -> bool:
    """Does this error payload describe a context-window overflow?"""
    low = text.lower()
    return any(m in low for m in _CONTEXT_ERROR_MARKERS)


class OpenAICompatibleBackend:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        tool_choice: str = "required",
        seed: int | None = None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.tool_choice = tool_choice  # "required" enforces one tool call/turn
        self.seed = seed
        self.timeout = timeout

    def reset(self) -> None:  # stateless
        pass

    def step(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": self.tool_choice,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if is_context_length_error(detail):
                # Its own exception, not a generic RuntimeError: the runner ends
                # the episode as "context_exhausted" rather than crashing the
                # whole run, and the outcome stays separable in the analysis.
                raise ContextLengthExceeded(
                    f"HTTP {e.code} from {self.base_url}: {detail}"
                ) from e
            raise RuntimeError(f"HTTP {e.code} from {self.base_url}: {detail}") from e

        # Some servers report the overflow in a 200 body instead of an HTTP
        # error, and a body with an error carries no choices to unpack.
        if body.get("error"):
            detail = json.dumps(body["error"])
            if is_context_length_error(detail):
                raise ContextLengthExceeded(f"{self.base_url}: {detail}")
            raise RuntimeError(f"{self.base_url}: {detail}")

        message = body["choices"][0]["message"]
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc["function"]
            raw_args = fn.get("arguments") or "{}"
            parse_error = None
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as e:
                # Don't silently coerce to {} — that would show up downstream as
                # an ordinary invalid_direction and hide a real interop bug.
                args, parse_error = {}, f"{e}: {raw_args!r}"
            if not isinstance(args, dict):
                args, parse_error = {}, f"arguments are not an object: {raw_args!r}"
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn["name"],
                    arguments=args,
                    parse_error=parse_error,
                )
            )

        # Reasoning/thinking is returned in a separate field by Ollama and most
        # reasoning models (field name varies across servers).
        reasoning = message.get("reasoning") or message.get("reasoning_content")

        return LLMResponse(
            tool_calls=tool_calls,
            assistant_message=message,
            text=message.get("content"),
            reasoning=reasoning,
            usage=body.get("usage"),
        )
