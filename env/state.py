"""True maze state during an episode.

Holds a mutable working copy of the maze's passages so that wall removals do
not corrupt the original maze (whose passages back the hidden oracle). Produces
the agent's *local* observation — it never exposes the full maze.
"""

from __future__ import annotations

from env.generation import DIRECTIONS, Cell, Maze, edge


class MazeState:
    def __init__(self, maze: Maze):
        self.maze = maze
        # Mutable working copy — removals mutate this, not maze.passages.
        self.passages = set(maze.passages)
        self.position: Cell = maze.start
        self.step: int = 0
        self.done: bool = False
        # Log of genuine wall removals: (step, from_cell, direction, to_cell).
        self.removed_walls: list[dict] = []
        self.visited: set[Cell] = {maze.start}

    # --- geometry helpers -------------------------------------------------
    def in_bounds(self, cell: Cell) -> bool:
        return self.maze.in_bounds(cell)

    def has_wall(self, cell: Cell, direction: str) -> bool:
        nc = self.maze.neighbor(cell, direction)
        if not self.in_bounds(nc):
            return True  # grid boundary is always a wall
        return edge(cell, nc) not in self.passages

    def at_goal(self) -> bool:
        return self.position == self.maze.goal

    # --- actions ----------------------------------------------------------
    def move(self, direction: str) -> str:
        if direction not in DIRECTIONS:
            return "invalid_direction"
        if self.has_wall(self.position, direction):
            return "blocked"
        self.position = self.maze.neighbor(self.position, direction)
        self.visited.add(self.position)
        return "moved"

    def remove_wall(self, direction: str) -> str:
        if direction not in DIRECTIONS:
            return "invalid_direction"
        nc = self.maze.neighbor(self.position, direction)
        if not self.in_bounds(nc):
            # No cell on the other side (boundary) — nothing to open.
            return "no_adjacent_cell"
        e = edge(self.position, nc)
        if e in self.passages:
            return "wall_already_open"
        self.passages.add(e)
        self.removed_walls.append(
            {
                "step": self.step,
                "from": list(self.position),
                "direction": direction,
                "to": list(nc),
            }
        )
        return "wall_removed"

    # --- observation ------------------------------------------------------
    def _walls_of(self, cell: Cell) -> dict[str, str]:
        return {
            d: ("open" if not self.has_wall(cell, d) else "wall") for d in DIRECTIONS
        }

    def observe(self) -> dict:
        """Local view only. Adjacent cells are revealed only through open
        passages (you cannot see through a wall)."""
        cur = self.position
        adjacent: dict[str, dict | None] = {}
        for d in DIRECTIONS:
            nc = self.maze.neighbor(cur, d)
            if self.in_bounds(nc) and not self.has_wall(cur, d):
                adjacent[d] = {"position": list(nc), "walls": self._walls_of(nc)}
            else:
                adjacent[d] = None
        return {
            "step": self.step,
            "position": list(cur),
            "goal": list(self.maze.goal),
            "at_goal": self.at_goal(),
            "current_cell_walls": self._walls_of(cur),
            "adjacent_cells": adjacent,
        }
