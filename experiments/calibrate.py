"""Regenerate the reference figures in experiments/README.md.

Everything here is measured from the harness itself — no model, no API key —
so the numbers in the README can be re-derived after any change to generation,
the observation format, or the prompts:

    uv run python experiments/calibrate.py

Three tables:

1. **Per size.** Cells, shortest path, and the number of steps a perfect
   depth-first explorer needs to visit every cell of an unsolvable maze. That
   step count is the floor on "explored everything before deciding", so it is
   what `--max-steps` has to clear.
2. **Context.** Harness-side bytes added to the conversation per turn, and the
   resulting prompt size at full exploration. A **floor**: it counts the system
   prompt and the observations, not the model's own output (reasoning text and
   tool-call JSON), which for a reasoning model can dominate.
3. **Invariants.** The two properties the generator asserts, checked across
   seeds: the explorable region of an unsolvable maze is always `rows*cols - 1`,
   and the two bands differ by exactly the goal's own passages.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from backends import DummyExplorerBackend  # noqa: E402
from env.generation import MazeLabel, make_maze  # noqa: E402
from env.oracle import reachable_component  # noqa: E402
from runner import run_episode  # noqa: E402
from runner.episode import _result_content  # noqa: E402

SIZES = (5, 7, 9, 11, 15, 21)
SEEDS = 25
# What a real model costs relative to a perfect explorer. It revisits cells,
# re-looks, backtracks badly, and (if it reasons aloud) writes far more per turn
# than the harness does. A guess, stated once here rather than buried.
REAL_MODEL_FACTOR = 2.5


def explore_cost(rows: int, cols: int, seed: int) -> tuple[int, int]:
    """(steps to exhaust the component, harness bytes added per turn).

    The explorer idles on look() once it has visited everything, so the step of
    its first look is the moment full exploration completed.
    """
    maze = make_maze(seed, rows, cols, MazeLabel.UNSOLVABLE)
    episode = run_episode(
        maze, DummyExplorerBackend(), max_steps=60 * rows * cols, show_step_budget=False
    )
    looks = [
        t["step"]
        for t in episode["trajectory"]
        if t["action"] and t["action"]["name"] == "look"
    ]
    steps = looks[0] if looks else episode["total_steps"]
    # One turn appends the tool result (result + observation) and the assistant
    # message carrying the tool call.
    turns = [
        len(_result_content(t["result"], t["observation"]))
        + len(json.dumps({"name": t["action"]["name"], "arguments": t["action"]["arguments"]}))
        for t in episode["trajectory"]
        if t["action"]
    ]
    return steps, round(statistics.mean(turns))


def table_sizes() -> None:
    print("## Per size\n")
    print("| size | cells | shortest path | full exploration | suggested --max-steps |")
    print("| --- | --- | --- | --- | --- |")
    for n in SIZES:
        paths = [
            make_maze(s, n, n, MazeLabel.SOLVABLE).shortest_path_length
            for s in range(SEEDS)
        ]
        steps, _ = explore_cost(n, n, 0)
        print(
            f"| {n}x{n} | {n * n} | {round(statistics.mean(paths))} | "
            f"{steps} | {round(steps * 3.5, -1):.0f} |"
        )
    print()


def table_context() -> None:
    system_chars = len(config.build_system_prompt(include_budget=False))
    print("## Context (a floor — harness bytes only)\n")
    print(f"System prompt: {system_chars} chars (~{system_chars // 4} tokens).\n")
    print("| size | chars/turn | at full exploration | x2.5 for a real model |")
    print("| --- | --- | --- | --- |")
    for n in SIZES:
        steps, per_turn = explore_cost(n, n, 0)
        tokens = (system_chars + steps * per_turn) // 4
        print(
            f"| {n}x{n} | {per_turn} | {tokens:,} tokens | "
            f"{round(tokens * REAL_MODEL_FACTOR):,} tokens |"
        )
    print()


def table_invariants() -> None:
    print("## Generator invariants\n")
    print("| size | explorable region (unsolvable) | bands differ only at the goal |")
    print("| --- | --- | --- |")
    for n in SIZES:
        regions, goal_only = set(), True
        for seed in range(SEEDS):
            unsolvable = make_maze(seed, n, n, MazeLabel.UNSOLVABLE)
            solvable = make_maze(seed, n, n, MazeLabel.SOLVABLE)
            regions.add(len(reachable_component(unsolvable)))
            difference = solvable.passages ^ unsolvable.passages
            goal_only &= bool(difference) and all(
                solvable.goal in e for e in difference
            )
        expected = {n * n - 1}
        ok = "yes" if regions == expected else f"NO ({sorted(regions)})"
        print(f"| {n}x{n} | {ok}, always {n * n - 1} | {'yes' if goal_only else 'NO'} |")
    print()


if __name__ == "__main__":
    print(f"Measured over {SEEDS} seeds per size; exploration from seed 0.\n")
    table_sizes()
    table_context()
    table_invariants()
