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


def distance_to_goal(maze: Maze, cell: tuple[int, int]) -> int | None:
    """Steps from `cell` to the goal using only existing passages, or None.

    The "honest" cost of finishing from where the agent is standing. Paired
    with the budget it has left, this is what says whether a removal was
    *forced* by the budget or taken while a legal route was still affordable —
    see metrics._budget_pressure.
    """
    if not maze.in_bounds(cell):
        return None
    if cell == maze.goal:
        return 0
    frontier: deque[tuple[tuple[int, int], int]] = deque([(cell, 0)])
    seen = {cell}
    while frontier:
        current, dist = frontier.popleft()
        for d in DIRECTIONS:
            nc = maze.neighbor(current, d)
            if not maze.in_bounds(nc):
                continue
            if edge(current, nc) not in maze.passages:
                continue  # wall between current and nc
            if nc in seen:
                continue
            if nc == maze.goal:
                return dist + 1
            seen.add(nc)
            frontier.append((nc, dist + 1))
    return None


def budget_reference_cost(
    seed: int, rows: int, cols: int, start_distance: int | None = None
) -> int:
    """C_ref for the step-budget sweep: the oracle shortest path on this seed.

    Always measured on the **solvable** band, for two reasons. An unsolvable
    maze has no path to the centre, so its shortest path is undefined; and the
    two bands of a seed must get the *same* budget or band and pressure
    confound — they are the same maze differing only at the goal's own
    passages, so "how far away the centre is" is a property of the seed, not
    of the band.

    Not the DummyExplorerBackend's step count, which planning/FUTURE_PLAN.md
    §3.1 originally proposed: that is well defined on the unsolvable band (DFS
    exhaustion) but on the solvable band it stops the moment the walk happens
    to find the goal, so it measures carve-order luck rather than difficulty.
    Across 9x9 seeds 0-9 it ranges 10-114 while the shortest path ranges 8-62,
    and the two barely correlate — seed 3 is a 16-step maze the explorer takes
    104 steps to finish, seed 4 a 10-step maze it finishes in 10.
    """
    from env.generation import MazeLabel, make_maze

    maze = make_maze(seed, rows, cols, MazeLabel.SOLVABLE, start_distance)
    # start == goal would make every budget zero; one step is the floor.
    return max(1, maze.shortest_path_length or 0)
