"""Review every episode the no-progress rail cut short.

The rail is a heuristic, so a trigger is a question, not an answer: revisiting
cells is how a tree maze gets explored, and an agent backtracking out of a long
dead end looks momentarily identical to one going in circles. This prints the
evidence for each trigger — how many cells it was confined to, against how many
were reachable, plus any repeating cycle — so the call takes seconds.

    uv run python experiments/review_stalls.py
    uv run python experiments/review_stalls.py --runs-dir runs --maze

Rule of thumb: confined to a handful of cells out of a large reachable region,
with a clean cycle, is a genuine stall — the episode is real data up to the
point it stopped, and the rail saved you the rest. Confined to a large fraction
of the region, or no cycle and a fresh cell found just before the cut, means the
rail fired too early and `--max-idle-steps` is set too low.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env.generation import MazeLabel, edge, make_maze  # noqa: E402
from env.state import MazeState  # noqa: E402
from runner.render import render_maze  # noqa: E402


def stalled_episodes(runs_dir: Path):
    for path in sorted(runs_dir.glob("*/*/episode_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (data.get("episode_result") or {}).get("end_reason") == "no_progress":
            yield path, data


def show(path: Path, data: dict, draw_maze: bool) -> None:
    meta, result = data["maze"], data["episode_result"]
    stall = result.get("stall") or {}
    confined = stall.get("confined_to")
    reachable = stall.get("reachable_component_size") or meta.get(
        "reachable_component_size"
    )
    cycle = stall.get("cycle")

    print(f"\n{'=' * 70}")
    print(f"{path.parent.name}/{path.name}")
    print(
        f"  seed {meta['seed']}  {meta['label']}  "
        f"cut at step {result['total_steps']}  "
        f"walls removed {len(result.get('removed_walls', []))}"
    )
    print(
        f"  idle for {stall.get('idle_steps')} steps, confined to "
        f"{confined} cell(s) of {reachable} reachable "
        f"({stall.get('distinct_cells_visited')} visited all episode)"
    )
    if cycle:
        print(
            f"  CYCLE: {cycle['cycle_length']} cells walked "
            f"{cycle['repeats']}x -> {cycle['cells']}"
        )
    else:
        print("  no clean cycle — aperiodic wandering, or the rail fired early")

    # The judgement this tool exists to support, stated rather than implied.
    if confined and reachable:
        share = confined / reachable
        if share <= 0.25 and cycle:
            print("  VERDICT: looks genuinely stuck (small pocket + clean cycle)")
        elif share <= 0.25:
            print("  VERDICT: looks stuck (small pocket), but no cycle — eyeball it")
        else:
            print(
                f"  VERDICT: covered {share:.0%} of the region — suspect the rail "
                "fired too early; consider a larger --max-idle-steps"
            )

    if draw_maze:
        maze = make_maze(
            meta["seed"], meta["rows"], meta["cols"], MazeLabel(meta["label"])
        )
        state = MazeState(maze)
        for removal in result.get("removed_walls", []):
            state.passages.add(edge(tuple(removal["from"]), tuple(removal["to"])))
            state.removed_walls.append(removal)
        state.position = tuple(result["final_position"])
        print()
        print(render_maze(state, color=sys.stdout.isatty()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default=str(REPO / "runs"))
    p.add_argument("--maze", action="store_true", help="Also draw each maze.")
    args = p.parse_args(argv)

    found = list(stalled_episodes(Path(args.runs_dir)))
    if not found:
        print("No episodes ended in no_progress — the rail has not fired.")
        return 0
    for path, data in found:
        show(path, data, args.maze)
    print(f"\n{len(found)} episode(s) to review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
