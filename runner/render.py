"""Live ASCII renderer for watching an agent explore a maze.

Draws the *true* maze (the observer's view — the agent still only ever gets its
local observation). The agent is `A`, the goal `G`, the start `S`. Walls the
agent has removed are drawn in colour so defection is visible at a glance.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
import unicodedata

from env.generation import edge
from env.state import MazeState
from metrics.compute import removal_justified

# ANSI helpers (used only when writing to a real terminal).
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_CLEAR = "\033[2J\033[H"


# Cell glyphs. ASCII by default so the maze renders in any terminal, in a log,
# or piped to a file; emoji are opt-in because they need a font and a terminal
# that agrees with unicodedata about their width.
ASCII_GLYPHS = {"agent": "A", "goal": "G", "start": "S"}
EMOJI_GLYPHS = {"agent": "\U0001F916", "goal": "\U0001F3C1", "start": "\U0001F3E0"}

_CELL_WIDTH = 3


def _c(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


def _display_width(text: str) -> int:
    """Terminal columns a glyph occupies. Emoji are East-Asian 'wide' and take
    two, so padding computed with len() shears the grid one column per emoji."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(glyph: str) -> str:
    """Centre a glyph in a cell, measured in columns rather than characters."""
    slack = _CELL_WIDTH - _display_width(glyph)
    left = max(slack, 0) // 2
    return " " * left + glyph + " " * max(slack - left, 0)


def _removed_edges(state: MazeState) -> set:
    return {
        edge(tuple(r["from"]), tuple(r["to"])) for r in state.removed_walls
    }


def render_maze(
    state: MazeState,
    color: bool = True,
    removal_color: str = _RED,
    glyphs: dict[str, str] | None = None,
) -> str:
    maze = state.maze
    removed = _removed_edges(state)
    pos, goal, start = state.position, maze.goal, maze.start
    g = glyphs or ASCII_GLYPHS

    def cell_glyph(cell) -> str:
        # Pad first, colour second: ANSI codes are zero-width but len() counts
        # them, so measuring a coloured string gets the padding wrong.
        if cell == pos:
            return _c(_pad(g["agent"]), _BOLD + _CYAN, color)
        if cell == goal:
            return _c(_pad(g["goal"]), _BOLD + _GREEN, color)
        if cell == start:
            return _c(_pad(g["start"]), _DIM, color)
        return " " * _CELL_WIDTH

    lines = ["+" + "---+" * maze.cols]
    for r in range(maze.rows):
        # Row of cells + east walls.
        row = "|"
        for c in range(maze.cols):
            row += cell_glyph((r, c))
            east = (r, c + 1)
            if not state.has_wall((r, c), "E"):
                opened = edge((r, c), east) in removed
                row += _c(":", removal_color, color) if opened else " "
            else:
                row += "|"
        lines.append(row)
        # Row of south walls.
        sep = "+"
        for c in range(maze.cols):
            south = (r + 1, c)
            if not state.has_wall((r, c), "S"):
                opened = edge((r, c), south) in removed
                sep += _c("...", removal_color, color) if opened else "   "
            else:
                sep += "---"
            sep += "+"
        lines.append(sep)
    return "\n".join(lines)


class MazeWatcher:
    """Step callback that clears the screen and redraws each turn."""

    def __init__(self, title: str, stream=None, show_reasoning: bool = True,
                 reasoning_lines: int = 12, glyphs: dict[str, str] | None = None):
        self.title = title
        self.glyphs = glyphs or ASCII_GLYPHS
        self.stream = stream or sys.stdout
        self.color = getattr(self.stream, "isatty", lambda: False)()
        self.show_reasoning = show_reasoning
        self.reasoning_lines = reasoning_lines
        # Colour for removed walls, decided at the moment of the first removal:
        # green if opening a wall was the right call (sealed maze, agent had
        # explored the whole wall-free region first), red otherwise.
        self.removal_color = _RED
        self._removal_resolved = False

    def _resolve_removal_color(self, state: MazeState) -> None:
        """Judge the first removal the same way the metrics do: the visited-set
        snapshot at this instant is what `explored_all_before_removal` measures."""
        maze = state.maze
        explored_all = (
            bool(maze.reachable_component_size)
            and len(state.visited) >= maze.reachable_component_size
        )
        if removal_justified(maze.reachable, explored_all):
            self.removal_color = _GREEN

    def __call__(self, state: MazeState, action, result, justification, reasoning=None):
        if result == "wall_removed" and not self._removal_resolved:
            self._resolve_removal_color(state)
            self._removal_resolved = True
        frame = [
            _CLEAR if self.color else "",
            _c(self.title, _BOLD, self.color),
            render_maze(
                state,
                color=self.color,
                removal_color=self.removal_color,
                glyphs=self.glyphs,
            ),
            "",
            _c("legend: ", _DIM, self.color)
            + f"{self.glyphs['agent']}=agent  {self.glyphs['goal']}=goal  "
            + f"{self.glyphs['start']}=start  "
            + _c(":/…=removed wall", _RED, self.color)
            + _c(" (green=justified removal)", _GREEN, self.color),
            f"step {state.step}  pos {list(state.position)} -> goal {list(state.maze.goal)}",
        ]
        act = "(none)" if action is None else f"{action['name']}({action.get('arguments', {})})"
        line = f"action: {act}  ->  {result}"
        if result == "wall_removed":
            line = _c(line, self.removal_color, self.color)
        frame.append(line)
        if justification:
            frame.append(_c(f"justification: {justification}", _DIM, self.color))
        if self.show_reasoning and reasoning:
            frame.append(_c("reasoning:", _BOLD, self.color))
            frame.append(_c(self._wrap_reasoning(reasoning), _DIM, self.color))
        self.stream.write("\n".join(frame) + "\n")
        self.stream.flush()

    def _wrap_reasoning(self, reasoning: str) -> str:
        width = max(40, (shutil.get_terminal_size((100, 24)).columns) - 2)
        wrapped: list[str] = []
        for para in reasoning.strip().splitlines():
            wrapped.extend(textwrap.wrap(para, width=width) or [""])
        if len(wrapped) > self.reasoning_lines:
            # Keep the tail — the conclusion is usually the most relevant part.
            wrapped = ["  …"] + wrapped[-self.reasoning_lines :]
        return "\n".join(wrapped)
