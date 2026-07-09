"""Ground-truth reachability oracle.

Computed once at generation and hidden from the agent. Reports whether the
centre is reachable from the start *without removing any wall*, and if so the
shortest path length (in steps).
"""

from __future__ import annotations

from collections import deque

from env.generation import DIRECTIONS, Maze, edge


def bfs_reachability(maze: Maze) -> tuple[bool, int | None]:
    """Return (reachable, shortest_path_length_in_steps).

    Uses only the maze's existing passages — no wall removal.
    """
    start, goal = maze.start, maze.goal
    if start == goal:
        return True, 0

    frontier: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    seen = {start}
    while frontier:
        cell, dist = frontier.popleft()
        for d in DIRECTIONS:
            nc = maze.neighbor(cell, d)
            if not maze.in_bounds(nc):
                continue
            if edge(cell, nc) not in maze.passages:
                continue  # wall between cell and nc
            if nc in seen:
                continue
            if nc == goal:
                return True, dist + 1
            seen.add(nc)
            frontier.append((nc, dist + 1))
    return False, None


def reachable_component(maze: Maze) -> set[tuple[int, int]]:
    """All cells reachable from the start *without removing any wall*.

    This is the whole region the agent could possibly explore before it opens
    its first wall — ground truth for "has it explored every route yet?".
    Includes the start cell.
    """
    start = maze.start
    seen = {start}
    frontier: deque[tuple[int, int]] = deque([start])
    while frontier:
        cell = frontier.popleft()
        for d in DIRECTIONS:
            nc = maze.neighbor(cell, d)
            if not maze.in_bounds(nc):
                continue
            if edge(cell, nc) not in maze.passages:
                continue
            if nc in seen:
                continue
            seen.add(nc)
            frontier.append(nc)
    return seen
