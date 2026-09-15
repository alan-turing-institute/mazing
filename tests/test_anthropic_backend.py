"""Translation tests for the Anthropic Messages backend.

The harness speaks OpenAI shapes end to end; this backend is the only place
that translates, so a bug here is a silent change to what the model sees in the
middle of the thing being measured. The pure translation functions need no SDK
and no network, so they are tested directly.
"""

from __future__ import annotations

import json

import pytest

from backends.anthropic_foundry import (
    NATIVE_CONTENT_KEY,
    _to_anthropic_messages,
    _to_anthropic_tools,
)
from env.tools import TOOL_SCHEMAS


def test_tools_translate_to_input_schema():
    tools = _to_anthropic_tools(TOOL_SCHEMAS)
    assert len(tools) == len(TOOL_SCHEMAS)
    for src, out in zip(TOOL_SCHEMAS, tools):
        assert out["name"] == src["function"]["name"]
        assert out["input_schema"] == src["function"]["parameters"]
        assert "parameters" not in out and "function" not in out
    # The exception action still demands its justification — dropping a required
    # field here would let a removal through unjustified and quietly change what
    # removals_without_justification counts.
    remove = next(t for t in tools if t["name"] == "remove_wall")
    assert "justification" in remove["input_schema"]["required"]


def test_system_prompt_is_lifted_out_of_messages():
    system, msgs = _to_anthropic_messages(
        [
            {"role": "system", "content": "POLICY TEXT"},
            {"role": "user", "content": "obs"},
        ]
    )
    assert system == "POLICY TEXT"
    assert [m["role"] for m in msgs] == ["user"]


def test_tool_results_coalesce_into_one_user_message():
    """The runner appends one `role: tool` message per tool call, but the
    Messages API requires every tool_result for a turn in a single user
    message. Splitting them is a 400 — and the shape that trains a model out of
    parallel calls."""
    system, msgs = _to_anthropic_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": None, NATIVE_CONTENT_KEY: [{"type": "tool_use"}]},
            {"role": "tool", "tool_call_id": "a", "content": "res-a"},
            {"role": "tool", "tool_call_id": "b", "content": "res-b"},
            {"role": "tool", "tool_call_id": "c", "content": "res-c"},
        ]
    )
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    blocks = msgs[-1]["content"]
    assert len(blocks) == 3
    assert [b["tool_use_id"] for b in blocks] == ["a", "b", "c"]
    assert all(b["type"] == "tool_result" for b in blocks)


def test_separate_turns_do_not_coalesce():
    """Results from two different assistant turns must stay in their own user
    messages — merging them would reorder the conversation."""
    _, msgs = _to_anthropic_messages(
        [
            {"role": "user", "content": "obs"},
            {"role": "assistant", "content": None, NATIVE_CONTENT_KEY: [{"type": "tool_use"}]},
            {"role": "tool", "tool_call_id": "a", "content": "res-a"},
            {"role": "assistant", "content": None, NATIVE_CONTENT_KEY: [{"type": "tool_use"}]},
            {"role": "tool", "tool_call_id": "b", "content": "res-b"},
        ]
    )
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert len(msgs[2]["content"]) == 1
    assert len(msgs[4]["content"]) == 1


def test_assistant_turns_replay_native_blocks_verbatim():
    """Thinking blocks are bound to the model that produced them and must be
    echoed back unchanged. Rebuilding the turn from the OpenAI projection would
    drop the signature and silently change what the model sees."""
    native = [
        {"type": "thinking", "thinking": "all four faces are walled", "signature": "sig-abc"},
        {"type": "tool_use", "id": "t1", "name": "remove_wall", "input": {"direction": "N"}},
    ]
    _, msgs = _to_anthropic_messages(
        [
            {"role": "user", "content": "obs"},
            {
                "role": "assistant",
                "content": "text projection",
                "tool_calls": [{"id": "t1"}],
                NATIVE_CONTENT_KEY: native,
            },
        ]
    )
    assert msgs[-1]["content"] is native
    assert msgs[-1]["content"][0]["signature"] == "sig-abc"


def test_full_episode_conversation_round_trips():
    """A realistic runner conversation must alternate user/assistant cleanly —
    the Messages API rejects two assistant turns in a row."""
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "obs0"}]
    for i in range(5):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                NATIVE_CONTENT_KEY: [
                    {"type": "tool_use", "id": f"t{i}", "name": "move", "input": {"direction": "N"}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"obs{i + 1}"})

    system, msgs = _to_anthropic_messages(messages)
    assert system == "sys"
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user"
    for a, b in zip(roles, roles[1:]):
        assert a != b, f"consecutive {a} turns: {roles}"


# --- response translation (needs the SDK only for the exception type) ---

anthropic = pytest.importorskip(
    "anthropic", reason="needs the optional `anthropic` extra"
)


class _Block:
    def __init__(self, **kw):
        self._kw = kw

    def model_dump(self, exclude_none=False):
        return dict(self._kw)


class _Usage(_Block):
    pass


class _Response:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage


def _backend(monkeypatch, response):
    from backends import anthropic_foundry

    class FakeMessages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kw):
            self.kwargs = kw
            if isinstance(response, Exception):
                raise response
            return response

    class FakeClient:
        def __init__(self, **kw):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "AnthropicFoundry", FakeClient, raising=False)
    return anthropic_foundry.AnthropicFoundryBackend(model="claude-opus-5", api_key="k")


def test_response_translates_blocks_usage_and_reasoning(monkeypatch):
    response = _Response(
        content=[
            _Block(type="thinking", thinking="I have seen all four goal faces walled."),
            _Block(type="text", text="Removing the north wall."),
            _Block(
                type="tool_use",
                id="t1",
                name="remove_wall",
                input={"direction": "N", "justification": "all four faces walled"},
            ),
        ],
        usage=_Usage(input_tokens=1000, output_tokens=50, cache_read_input_tokens=200),
    )
    backend = _backend(monkeypatch, response)
    out = backend.step(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "obs"}],
        TOOL_SCHEMAS,
    )

    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert (tc.name, tc.id, tc.parse_error) == ("remove_wall", "t1", None)
    assert tc.arguments["justification"] == "all four faces walled"
    assert "four goal faces" in out.reasoning
    assert "Removing the north wall." in out.text
    # peak_prompt_tokens reads prompt_tokens; cached reads are tokens the model
    # still saw, so they belong in the prompt count.
    assert out.usage["prompt_tokens"] == 1200
    assert out.usage["total_tokens"] == 1250
    # The native blocks ride along for verbatim replay next turn.
    assert out.assistant_message[NATIVE_CONTENT_KEY][0]["type"] == "thinking"
    assert json.loads(
        out.assistant_message["tool_calls"][0]["function"]["arguments"]
    )["direction"] == "N"


def test_tool_choice_defaults_to_auto_so_thinking_survives(monkeypatch):
    """Forcing a tool call suppresses extended thinking outright on Claude
    (measured: 0 thinking tokens per turn under "any", ~1000 under "auto"), which
    would measure the model deliberating less than it does by default and leave
    no reasoning in the trajectory at all."""
    backend = _backend(monkeypatch, _Response(content=[]))
    backend.step([{"role": "system", "content": "s"}, {"role": "user", "content": "o"}],
                 TOOL_SCHEMAS)
    sent = backend.client.messages.kwargs
    assert sent["tool_choice"] == {"type": "auto"}
    assert sent["thinking"]["type"] == "adaptive"
    assert sent["system"] == "s"
    assert sent["max_tokens"] == 8192
    # effort omitted unless asked for, so the model default is what runs
    assert "output_config" not in sent


def test_context_overflow_becomes_its_own_outcome(monkeypatch):
    """context_exhausted is an episode outcome the runner records, not a crash
    that costs every remaining episode."""
    from backends.base import ContextLengthExceeded

    import httpx2

    request = httpx2.Request("POST", "https://example.invalid/v1/messages")
    err = anthropic.BadRequestError(
        "prompt is too long: 1050000 tokens > 1000000 maximum",
        response=httpx2.Response(400, request=request),
        body=None,
    )
    backend = _backend(monkeypatch, err)
    with pytest.raises(ContextLengthExceeded):
        backend.step([{"role": "user", "content": "o"}], TOOL_SCHEMAS)


def test_tool_choice_any_is_still_available(monkeypatch):
    """The forcing arm stays reachable for protocol parity with the vLLM runs."""
    from backends import anthropic_foundry

    backend = _backend(monkeypatch, _Response(content=[]))
    backend.tool_choice = "any"
    backend.step([{"role": "user", "content": "o"}], TOOL_SCHEMAS)
    assert backend.client.messages.kwargs["tool_choice"] == {"type": "any"}
