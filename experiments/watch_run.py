"""Attach to a run already in flight and redraw its live episode.

`run.py --watch` renders from inside the loop, so it has to be decided before
the run starts. This attaches afterwards: episode checkpoints are rewritten
after every step, so tailing them gives the same picture for a run that is
already going — including one started in another terminal.

    uv run python experiments/watch_run.py            # newest run
    uv run python experiments/watch_run.py runs/<policy>/<run-id>

Read-only: it never writes to the run directory, and Ctrl-C cannot disturb the
run it is watching. The one thing it cannot show is the model's reasoning,
which `--watch` prints but the checkpoints do not carry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env.generation import MazeLabel, edge, make_maze  # noqa: E402
from env.state import MazeState  # noqa: E402
from runner.render import ASCII_GLYPHS, EMOJI_GLYPHS, render_maze  # noqa: E402

CLEAR = "\033[2J\033[H"


def newest_run(runs_dir: Path) -> Path:
    """The most recently touched directory holding a run_config.json."""
    dirs = [p for p in runs_dir.glob("*/*") if (p / "run_config.json").exists()]
    if not dirs:
        sys.exit(f"no runs found under {runs_dir}/")
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _load(path: Path):
    """Checkpoints are written atomically, but a run dir can still hold a
    half-written file from a killed process — skip those rather than crash."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def live_episode(run: Path):
    """The episode currently being written, else the last one finished."""
    eps = sorted(run.glob("episode_*.json"))
    for path in eps:
        data = _load(path)
        if data and not data.get("complete"):
            return data
    return _load(eps[-1]) if eps else None


def frame(run: Path, glyphs=None) -> str:
    episode = live_episode(run)
    if episode is None:
        return f"{run.name}\n\nwaiting for the first checkpoint..."

    meta, result = episode["maze"], episode["episode_result"]
    trajectory = episode["trajectory"]

    # The maze is not stored in the checkpoint, but generation is deterministic
    # in (seed, size, band), so it can be rebuilt exactly.
    maze = make_maze(meta["seed"], meta["rows"], meta["cols"], MazeLabel(meta["label"]))
    state = MazeState(maze)
    for removal in result.get("removed_walls", []):
        state.passages.add(edge(tuple(removal["from"]), tuple(removal["to"])))
        state.removed_walls.append(removal)
    if result.get("final_position"):
        state.position = tuple(result["final_position"])
    state.step = result.get("total_steps", 0)

    seen = {tuple(s["observation"]["position"]) for s in trajectory if s.get("observation")}
    config = _load(run / "run_config.json") or {}
    last = trajectory[-1] if trajectory else {}
    action = last.get("action") or {}
    action_str = (
        f"{action.get('name')}({action.get('arguments', {})})" if action else "(none)"
    )

    return "\n".join(
        [
            f"{run.name}   episode {episode['episode_index']} of {config.get('n_mazes', '?')}",
            render_maze(state, color=True, glyphs=glyphs),
            "",
            f"{meta['label']}  seed {meta['seed']}  "
            f"reachable region {meta.get('reachable_component_size')} cells",
            f"step {result.get('total_steps')}   cells seen {len(seen)}   "
            f"removals {len(result.get('removed_walls', []))}",
            f"peak_prompt_tokens {result.get('peak_prompt_tokens')}   "
            f"end_reason {result.get('end_reason')}",
            f"last: {action_str} -> {last.get('result')}",
        ]
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", nargs="?", default=None, help="Run directory (default: newest).")
    p.add_argument("--runs-dir", default=str(REPO / "runs"))
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between redraws.")
    p.add_argument(
        "--emoji",
        action="store_true",
        help="Draw the agent, goal and start as emoji instead of A/G/S.",
    )
    args = p.parse_args(argv)

    run = Path(args.run) if args.run else newest_run(Path(args.runs_dir))
    glyphs = EMOJI_GLYPHS if args.emoji else ASCII_GLYPHS
    try:
        while True:
            sys.stdout.write(CLEAR + frame(run, glyphs) + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
