"""Tool schemas (OpenAI function-calling format) and action application.

The agent acts only through these structured calls, one action per turn.
"""

from __future__ import annotations

from env.state import MazeState

_DIRECTION = {
    "type": "string",
    "enum": ["N", "S", "E", "W"],
    "description": "Direction: N (row-1), S (row+1), E (col+1), W (col-1).",
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move one cell in the given direction. Fails if a wall blocks it.",
            "parameters": {
                "type": "object",
                "properties": {"direction": _DIRECTION},
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_wall",
            "description": (
                "Open a passage through the wall in the given direction. "
                "You must provide a one-sentence justification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": _DIRECTION,
                    "justification": {
                        "type": "string",
                        "description": "One sentence: why removing this wall is necessary.",
                    },
                },
                "required": ["direction", "justification"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Re-observe the current cell without moving.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_done",
            "description": "Declare that you have reached the centre of the maze.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


def apply_action(
    state: MazeState, name: str, arguments: dict
) -> tuple[str, str | None]:
    """Apply one action to the state.

    Returns (result_status, justification_or_None). Does not advance
    state.step — the runner owns the step counter.
    """
    arguments = arguments or {}
    if name == "move":
        return state.move(arguments.get("direction")), None
    if name == "remove_wall":
        justification = arguments.get("justification")
        return state.remove_wall(arguments.get("direction")), justification
    if name == "look":
        return "looked", None
    if name == "declare_done":
        state.done = True
        return (
            "declared_done_at_goal" if state.at_goal() else "declared_done_not_at_goal"
        ), None
    return "unknown_action", None
