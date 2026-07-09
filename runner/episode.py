"""Run one agent-in-maze episode and record the full trajectory.

The harness owns the true maze state, applies actions, returns observations,
and enforces the step budget. The agent acts through one tool call per turn.
The observation is embedded as JSON in the conversation so every backend —
hosted LLM or dummy — reads state through the same channel, and it carries the
remaining step budget so the agent knows how many steps it has left.
"""

from __future__ import annotations

import json
import time
from typing import Callable

from backends.base import LLMBackend
from config import SYSTEM_PROMPT
from env.generation import Maze
from env.state import MazeState
from env.tools import TOOL_SCHEMAS, apply_action

# How many consecutive turns without a valid tool call before we give up.
_MAX_NO_ACTION = 3


def _obs_content(observation: dict) -> str:
    return json.dumps({"observation": observation}, indent=2)


def _result_content(result: str, observation: dict) -> str:
    return json.dumps({"result": result, "observation": observation}, indent=2)


def run_episode(
    maze: Maze,
    backend: LLMBackend,
    max_steps: int,
    system_prompt: str = SYSTEM_PROMPT,
    on_step: Callable | None = None,
    on_progress: Callable | None = None,
    step_delay: float = 0.0,
) -> dict:
    """Run one episode.

    max_steps is a hard step budget (always enforced, reproducible); it is
    exposed to the agent as steps_remaining so it knows how many steps it has.

    Callbacks:
      on_step(state, action, result, justification) -- once with the initial
          state and again after every applied action (live monitoring).
      on_progress(episode_dict) -- after every step, with the episode result so
          far (end_reason "in_progress"); used to checkpoint to disk so a kill
          mid-episode loses nothing. step_delay pauses after each on_step.
    """
    state = MazeState(maze)
    if hasattr(backend, "reset"):
        backend.reset()

    trajectory: list[dict] = []
    no_action_streak = 0
    # Distinct cells the agent had explored at the moment of its first wall
    # removal (ground truth from the true visited-set). Lets us ask whether it
    # had explored the whole reachable region before defecting.
    cells_at_first_removal: int | None = None

    def observe() -> dict:
        """Local observation, augmented with the remaining step budget."""
        obs = state.observe()
        obs["step_budget"] = max_steps
        obs["steps_remaining"] = max_steps - state.step
        return obs

    def result_dict(end_reason: str) -> dict:
        return {
            "trajectory": trajectory,
            "reached_goal": state.at_goal(),
            "total_steps": state.step,
            "removed_walls": state.removed_walls,
            "end_reason": end_reason,
            "final_position": list(state.position),
            "distinct_cells_visited": len(state.visited),
            "distinct_cells_at_first_removal": cells_at_first_removal,
        }

    def notify(action, result, justification, reasoning=None):
        if on_step is not None:
            on_step(state, action, result, justification, reasoning)
            if step_delay:
                time.sleep(step_delay)
        if on_progress is not None:
            on_progress(result_dict("in_progress"))

    notify(None, "start", None)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _obs_content(observe())},
    ]
    end_reason = "budget_exhausted"

    while True:
        if state.done:
            break
        if state.step >= max_steps:
            end_reason = "step_budget_exhausted"
            break

        obs_before = observe()
        response = backend.step(messages, TOOL_SCHEMAS)
        messages.append(response.assistant_message)

        if not response.tool_calls:
            no_action_streak += 1
            trajectory.append(
                {
                    "step": state.step,
                    "observation": obs_before,
                    "action": None,
                    "result": "no_action",
                    "justification": None,
                    "assistant_text": response.text,
                    "reasoning": response.reasoning,
                }
            )
            if no_action_streak >= _MAX_NO_ACTION:
                end_reason = "no_action"
                break
            messages.append(
                {"role": "user", "content": "Respond with exactly one tool call."}
            )
            continue

        no_action_streak = 0
        # Act on the first tool call only (one action per turn).
        first = response.tool_calls[0]
        result, justification = apply_action(state, first.name, first.arguments)

        if result == "wall_removed" and cells_at_first_removal is None:
            # state.visited already includes the cell it is standing in.
            cells_at_first_removal = len(state.visited)

        trajectory.append(
            {
                "step": state.step,
                "observation": obs_before,
                "action": {"name": first.name, "arguments": first.arguments},
                "result": result,
                "justification": justification,
                "assistant_text": response.text,
                "reasoning": response.reasoning,
            }
        )

        state.step += 1
        obs_after = observe()

        # Respond to every tool call so the conversation stays valid for hosted
        # APIs; only the first one actually changed the world.
        for i, tc in enumerate(response.tool_calls):
            if i == 0:
                content = _result_content(result, obs_after)
            else:
                content = json.dumps({"result": "ignored_one_action_per_turn"})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": content}
            )

        notify(
            {"name": first.name, "arguments": first.arguments},
            result,
            justification,
            response.reasoning,
        )

        if state.done:
            end_reason = "declared_done"
            break
        if state.at_goal():
            end_reason = "reached_goal"
            break

    return result_dict(end_reason)
