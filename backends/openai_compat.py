"""OpenAI-compatible /chat/completions client with tool calling.

Works with hosted providers and local OpenAI-compatible servers (vLLM,
Ollama's OpenAI endpoint, LM Studio, ...). Uses only the standard library so
the harness has no runtime dependencies. Provider is never hard-coded — model,
base URL, and API key are all configurable.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import http.client
import json
import sys
import time
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
        timeout: float = 600.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        max_rate_limit_retries: int = 8,
        max_retry_after: float = 300.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.tool_choice = tool_choice  # "required" enforces one tool call/turn
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Throttling gets its own, larger budget: a hosted endpoint can 429 many
        # times across a long run without anything being wrong, and those
        # retries must not eat the budget meant for a genuinely sick server.
        self.max_rate_limit_retries = max_rate_limit_retries
        self.max_retry_after = max_retry_after

    def reset(self) -> None:  # stateless
        pass

    # Transport errors that say "try again", not "this request is wrong": a
    # local server that stalled, was reloading a model, or dropped the socket.
    _TRANSIENT = (
        TimeoutError,
        ConnectionError,
        urllib.error.URLError,  # HTTPError is caught first — it subclasses this
        http.client.HTTPException,
    )

    def _retry_after_seconds(self, headers) -> float | None:
        """Seconds to wait per the server's Retry-After, if it sent a usable one.

        RFC 9110 allows either a delay in seconds or an HTTP date, and hosted
        endpoints send both spellings. A server that names a number knows its own
        throttle better than any backoff curve we could invent, so it wins — but
        it is capped, because an endpoint parked behind a multi-hour quota reset
        should surface as a failed run, not a process that looks alive for hours.
        """
        raw = headers.get("Retry-After") if headers else None
        if not raw:
            return None
        raw = raw.strip()
        try:
            seconds = float(raw)
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                return None
            if when is None:
                return None
            now = dt.datetime.now(dt.timezone.utc)
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            seconds = (when - now).total_seconds()
        if seconds != seconds or seconds < 0:  # NaN or a date already past
            return 0.0
        return min(seconds, self.max_retry_after)

    def _post(self, req) -> dict:
        """POST the request, retrying transient transport failures and throttling.

        urlopen raises TimeoutError when a server accepts the connection and
        then goes quiet, which is not an HTTPError and so used to escape step()
        and kill the whole run — losing every episode after the last completed
        one. A local Ollama stalling for two minutes is a blip to retry, not a
        reason to throw away an hour of episodes.

        A 429 or a 5xx is the hosted-endpoint version of the same thing. Both
        arrive as HTTPError, which was raised immediately, so a single throttle
        response part-way through a 20-episode run ended the run and lost every
        episode after the last completed one. They are retried with exponential
        backoff, honouring Retry-After when the server sends it, on a budget of
        their own: throttling is routine on a hosted endpoint and must not
        consume the retries meant for a genuinely sick connection.

        ContextLengthExceeded is deliberately NOT retried: it is a real episode
        outcome the runner records as `context_exhausted`, and retrying it would
        turn a finding into a hang. A 4xx that is not 429 is not retried either
        — the request is wrong, and sending it again just wastes quota.
        """
        transport_attempts = 0
        throttle_attempts = 0
        while True:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                if is_context_length_error(detail):
                    # Its own exception, not a generic RuntimeError: the runner
                    # ends the episode as "context_exhausted" rather than
                    # crashing the whole run, and the outcome stays separable in
                    # the analysis. Checked before the status test so a provider
                    # that reports an overflow as a 429 or 500 is still recorded
                    # as an episode outcome rather than retried into a hang.
                    raise ContextLengthExceeded(
                        f"HTTP {e.code} from {self.base_url}: {detail}"
                    ) from e
                if e.code != 429 and not (500 <= e.code < 600):
                    raise RuntimeError(
                        f"HTTP {e.code} from {self.base_url}: {detail}"
                    ) from e
                if throttle_attempts >= self.max_rate_limit_retries:
                    raise RuntimeError(
                        f"HTTP {e.code} from {self.base_url}: giving up after "
                        f"{self.max_rate_limit_retries} retries: {detail}"
                    ) from e
                delay = self._retry_after_seconds(getattr(e, "headers", None))
                if delay is None:
                    delay = self.retry_backoff * 2**throttle_attempts
                throttle_attempts += 1
                print(
                    f"  [backend] HTTP {e.code} - retry {throttle_attempts}/"
                    f"{self.max_rate_limit_retries} in {delay:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
            except self._TRANSIENT as e:
                if transport_attempts >= self.max_retries:
                    raise RuntimeError(
                        f"{self.base_url}: giving up after {self.max_retries} "
                        f"retries: {type(e).__name__}: {e}"
                    ) from e
                delay = self.retry_backoff * 2**transport_attempts
                transport_attempts += 1
                # stderr, not the trajectory: a retried blip is an operational
                # event, not something the agent did.
                print(
                    f"  [backend] {type(e).__name__}: {e} - retry "
                    f"{transport_attempts}/{self.max_retries} in {delay:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)

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
        body = self._post(req)

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
