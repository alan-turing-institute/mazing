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

from backends.base import LLMResponse, ToolCall


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
            raise RuntimeError(f"HTTP {e.code} from {self.base_url}: {detail}") from e

        message = body["choices"][0]["message"]
        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc["function"]
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn["name"], arguments=args)
            )

        return LLMResponse(
            tool_calls=tool_calls,
            assistant_message=message,
            text=message.get("content"),
        )
