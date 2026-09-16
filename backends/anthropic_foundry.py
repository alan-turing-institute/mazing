"""Anthropic Messages API backend (first-party or Microsoft Foundry).

Foundry exposes Claude over the *Anthropic* Messages API, not an
OpenAI-compatible route, so `OpenAICompatibleBackend` cannot talk to it: the
wire format differs in the system prompt (a top-level parameter, not a message),
tool definitions (`input_schema`, not `function.parameters`), tool results
(`tool_result` content blocks inside a *user* message, not `role: "tool"`
messages) and the response shape (typed content blocks, not `message.content` +
`tool_calls`).

The harness's backend interface is OpenAI-shaped in both directions — the runner
builds OpenAI messages and appends whatever `assistant_message` we return — so
this backend translates at its own boundary and leaves the runner untouched.
That keeps the conversation the runner logs identical in structure across every
backend, which is what makes the measurement comparable at all.

Requires the optional `anthropic` extra:

    uv run --extra anthropic python run.py --backend anthropic ...
"""

from __future__ import annotations

import json

from backends.base import ContextLengthExceeded, LLMResponse, ToolCall
from backends.openai_compat import is_context_length_error

# Key under which the native Anthropic content blocks ride along on the
# OpenAI-shaped assistant message. The runner treats assistant messages as
# opaque and only appends them, so this survives the round trip and lets us
# replay the turn to the API byte-for-byte.
#
# Replaying verbatim matters: thinking blocks are bound to the model that
# produced them and must be echoed back unchanged on the next request. Rebuilding
# them from the OpenAI projection would drop the signatures and silently change
# what the model sees between turns — a confound in the middle of the thing being
# measured.
NATIVE_CONTENT_KEY = "_anthropic_content"


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """OpenAI function schemas -> Anthropic tool schemas."""
    out = []
    for t in tools:
        fn = t["function"] if t.get("type") == "function" else t
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object"}),
            }
        )
    return out


def _to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI conversation -> (system prompt, Anthropic messages).

    Two structural differences are handled here:

    * the system prompt is a top-level parameter, not the first message;
    * the runner appends one `role: "tool"` message per tool call, but the
      Messages API requires every `tool_result` for one assistant turn to arrive
      together in a single user message — so consecutive tool messages are
      coalesced. Splitting them across user messages is a 400, and worse, the
      shape that trains a model out of parallel calls.
    """
    system: str | None = None
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            # Only the leading system message is a system prompt; the harness
            # never sends another, but concatenate rather than silently drop if
            # that ever changes.
            system = m["content"] if system is None else f"{system}\n\n{m['content']}"
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content", ""),
            }
            # Append to the open tool_result user message if there is one.
            if (
                out
                and out[-1]["role"] == "user"
                and isinstance(out[-1]["content"], list)
                and out[-1]["content"]
                and out[-1]["content"][0].get("type") == "tool_result"
            ):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant":
            native = m.get(NATIVE_CONTENT_KEY)
            if native is not None:
                out.append({"role": "assistant", "content": native})
            else:
                # A backend-agnostic assistant turn (only reachable if the
                # runner ever synthesises one); text is all there is to send.
                out.append(
                    {"role": "assistant", "content": m.get("content") or ""}
                )
        else:
            out.append({"role": "user", "content": m.get("content", "")})
    return system, out


class AnthropicFoundryBackend:
    """Claude over the Anthropic Messages API, on Foundry or first-party.

    tool_choice defaults to "auto", not the "any" that mirrors the OpenAI
    backend's "required". Forcing a tool call suppresses extended thinking
    outright on Claude (measured: 0 thinking tokens per turn under "any", ~1000
    under "auto" on the same prompt), which would measure the model deliberating
    less than it does by default and leave no reasoning in the trajectory. The
    local-model baseline runs "auto" for the same reason — Ollama ignores
    "required" — so this also matches how that data was collected.

    The cost is that the model may answer in prose instead of calling a tool;
    the runner already records that as `no_action` and gives it three tries, so
    the failure mode is visible in the data rather than silent.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 8192,
        tool_choice: str = "auto",
        thinking: bool = True,
        effort: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 8,
    ):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                "the anthropic SDK is required for --backend anthropic: "
                "run with `uv run --extra anthropic python run.py ...`"
            ) from e

        self._anthropic = anthropic
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.tool_choice = tool_choice
        self.thinking = thinking
        self.effort = effort
        # The SDK retries 408/409/429/5xx with exponential backoff itself, so a
        # throttled hosted endpoint does not end the run and lose every episode
        # after the last completed one. Raised well above the default 2 because
        # a 20-episode run is hundreds of requests against a shared quota.
        self.max_retries = max_retries

        kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": max_retries}
        if base_url:
            kwargs["base_url"] = base_url
        # AnthropicFoundry knows the Foundry auth header and URL shape; the
        # first-party client is the fallback for a plain api.anthropic.com key.
        client_cls = getattr(anthropic, "AnthropicFoundry", None)
        if client_cls is None:  # pragma: no cover - very old SDK
            client_cls = anthropic.Anthropic
        self.client = client_cls(**kwargs)

    def reset(self) -> None:  # stateless
        pass

    def step(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        system, anthropic_messages = _to_anthropic_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
            "tools": _to_anthropic_tools(tools),
            "tool_choice": {"type": self.tool_choice},
        }
        if system is not None:
            kwargs["system"] = system
        if self.thinking:
            # display="summarized" only changes whether the reasoning text is
            # returned — the model thinks, and is billed, identically either
            # way. It is on because the trajectory records the agent's reasoning
            # for every backend, and the necessity justifications are the whole
            # point of the experiment.
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}

        try:
            response = self.client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as e:
            # Same reasoning as the OpenAI backend: a context overflow is a real
            # episode outcome the runner records as `context_exhausted`, not a
            # crash that costs every remaining episode.
            if is_context_length_error(str(e)):
                raise ContextLengthExceeded(f"{self.model}: {e}") from e
            raise

        # Keep the blocks as plain dicts: they are replayed verbatim on the next
        # turn, and dicts serialise if anything downstream ever logs them.
        blocks = [b.model_dump(exclude_none=True) for b in response.content]

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for b in blocks:
            kind = b.get("type")
            if kind == "tool_use":
                args = b.get("input")
                parse_error = None
                if not isinstance(args, dict):
                    # Don't coerce to {} silently — downstream that would look
                    # like an ordinary invalid_direction and hide an interop bug.
                    args, parse_error = {}, f"input is not an object: {args!r}"
                tool_calls.append(
                    ToolCall(
                        id=b.get("id", ""),
                        name=b.get("name", ""),
                        arguments=args,
                        parse_error=parse_error,
                    )
                )
            elif kind == "text":
                text_parts.append(b.get("text", ""))
            elif kind == "thinking":
                reasoning_parts.append(b.get("thinking", ""))

        # The OpenAI-shaped projection the runner appends and logs, carrying the
        # native blocks so the next request replays this turn exactly.
        assistant_message = {
            "role": "assistant",
            "content": "\n".join(p for p in text_parts if p) or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ],
            NATIVE_CONTENT_KEY: blocks,
        }

        usage = None
        if response.usage is not None:
            u = response.usage.model_dump(exclude_none=True)
            # The runner reads peak_prompt_tokens off `prompt_tokens`, so map
            # the Anthropic names onto the OpenAI ones rather than leaving the
            # token accounting silently empty for this backend. Cache reads
            # count as prompt tokens the model saw.
            prompt = (
                (u.get("input_tokens") or 0)
                + (u.get("cache_read_input_tokens") or 0)
                + (u.get("cache_creation_input_tokens") or 0)
            )
            completion = u.get("output_tokens") or 0
            usage = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "anthropic_usage": u,
            }

        return LLMResponse(
            tool_calls=tool_calls,
            assistant_message=assistant_message,
            text="\n".join(p for p in text_parts if p) or None,
            reasoning="\n".join(p for p in reasoning_parts if p) or None,
            usage=usage,
            finish_reason=getattr(response, "stop_reason", None),
        )
