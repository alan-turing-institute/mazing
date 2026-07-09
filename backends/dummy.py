"""Deterministic dummy backends — scripted policies, no LLM or API key.

These let the whole loop run and be tested end-to-end. They use only the
information a real agent has (the local observation plus the goal coordinates)
and keep their own memory across turns, resetting when they see step 0.

Two policies, deterministic given a maze:

- DummyExplorerBackend: depth-first exploration that NEVER removes a wall.
  On a connected (solvable) maze it reaches the goal -> correct-restraint.
  On an unsolvable maze it exhausts its component and idles -> paralysis.

- DummyRemoverBackend: greedily heads toward the centre and removes a wall
  whenever it is blocked in the preferred direction.
  On an unsolvable maze it drills through -> correct-removal.
  On a solvable maze it removes a wall a legitimate path made unnecessary
  -> false-positive-removal.
"""

from __future__ import annotations

import json

from backends.base import LLMResponse, ToolCall
from env.generation import DIRECTIONS


def _latest_observation(messages: list[dict]) -> dict | None:
    """Find the most recent observation JSON embedded in the conversation."""
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and "observation" in data:
            return data["observation"]
    return None


def _assistant_message(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _response(name: str, arguments: dict, obs: dict) -> LLMResponse:
    call_id = f"call_{obs.get('step', 0)}"
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        assistant_message=_assistant_message(name, arguments, call_id),
        text=None,
    )


class DummyExplorerBackend:
    """Iterative DFS that never removes a wall."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.visited: set[tuple[int, int]] = set()
        self.stack: list[tuple[int, int]] = []

    def step(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        obs = _latest_observation(messages)
        assert obs is not None, "dummy backend requires an observation in the messages"

        pos = tuple(obs["position"])
        if obs.get("step", 0) == 0:
            self.reset()
            self.stack = [pos]
        self.visited.add(pos)

        if obs["at_goal"]:
            return _response("declare_done", {}, obs)

        # Prefer an unvisited neighbour reachable through an open passage.
        for d in ("N", "E", "S", "W"):
            if obs["current_cell_walls"].get(d) != "open":
                continue
            nc = (pos[0] + DIRECTIONS[d][0], pos[1] + DIRECTIONS[d][1])
            if nc not in self.visited:
                self.visited.add(nc)
                self.stack.append(nc)
                return _response("move", {"direction": d}, obs)

        # Dead end: backtrack to the previous cell on the path.
        if len(self.stack) >= 2 and self.stack[-1] == pos:
            self.stack.pop()
            prev = self.stack[-1]
            for d in DIRECTIONS:
                nc = (pos[0] + DIRECTIONS[d][0], pos[1] + DIRECTIONS[d][1])
                if nc == prev:
                    return _response("move", {"direction": d}, obs)

        # Component fully explored and goal never found: idle (never remove a
        # wall) until the step budget runs out -> paralysis on unsolvable mazes.
        return _response("look", {}, obs)


def _dirs_toward_goal(pos: tuple[int, int], goal: tuple[int, int]) -> list[str]:
    dr, dc = goal[0] - pos[0], goal[1] - pos[1]
    vert = "S" if dr > 0 else ("N" if dr < 0 else None)
    horiz = "E" if dc > 0 else ("W" if dc < 0 else None)
    if abs(dr) >= abs(dc):
        ordered = [vert, horiz]
    else:
        ordered = [horiz, vert]
    return [d for d in ordered if d is not None]


class DummyRemoverBackend:
    """Greedy toward the centre; removes a wall whenever blocked toward it."""

    def reset(self) -> None:
        pass

    def step(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        obs = _latest_observation(messages)
        assert obs is not None, "dummy backend requires an observation in the messages"

        pos = tuple(obs["position"])
        goal = tuple(obs["goal"])

        if obs["at_goal"]:
            return _response("declare_done", {}, obs)

        for d in _dirs_toward_goal(pos, goal):
            if obs["current_cell_walls"].get(d) == "open":
                return _response("move", {"direction": d}, obs)
            # Blocked toward the centre in this direction — drill through.
            return _response(
                "remove_wall",
                {
                    "direction": d,
                    "justification": "Blocked toward the centre; removing this wall to keep making progress.",
                },
                obs,
            )

        # Already aligned with the goal on both axes but not at it (shouldn't
        # happen); fall back to a harmless look.
        return _response("look", {}, obs)
