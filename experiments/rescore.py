"""Recompute the stored metrics of completed episodes, in place.

Episode files keep their metrics as computed at run time, so a metric added
later is missing from everything already collected. The trajectory is stored in
full, though, and the maze is reproducible from (seed, size, band,
start_distance) — so a new metric can be backfilled without re-running a single
episode.

    uv run python experiments/rescore.py --dry-run
    uv run python experiments/rescore.py

Only the `metrics` block is rewritten; the trajectory and episode_result are
left exactly as recorded. Incomplete episodes are skipped (their metrics are
null by design).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env.generation import MazeLabel, make_maze  # noqa: E402
from metrics import compute_metrics  # noqa: E402


def rescore(path: Path, dry_run: bool) -> str | None:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("complete"):
        return None
    meta, config = record["maze"], record.get("config", {})
    maze = make_maze(
        meta["seed"],
        rows=meta["rows"],
        cols=meta["cols"],
        label=MazeLabel(meta["label"]),
        start_distance=config.get("start_distance"),
    )
    # The maze must be the one the episode actually ran on, or every metric
    # derived from it is quietly wrong.
    if list(maze.start) != list(meta["start"]) or list(maze.goal) != list(meta["goal"]):
        return f"SKIPPED {path}: regenerated maze does not match the recorded one"

    episode = dict(record["episode_result"])
    episode["trajectory"] = record["trajectory"]
    fresh = compute_metrics(episode, maze)
    old = record.get("metrics") or {}
    added = sorted(set(fresh) - set(old))
    changed = sorted(k for k in old if k in fresh and old[k] != fresh[k])
    if not added and not changed:
        return None
    if not dry_run:
        record["metrics"] = fresh
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    note = f"{path.parent.name}/{path.name}: +{len(added)} field(s)"
    if changed:
        note += f", {len(changed)} changed: {changed}"
    return note


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default=str(REPO / "runs"))
    p.add_argument("--dry-run", action="store_true", help="Report, change nothing.")
    args = p.parse_args(argv)

    notes = [
        note
        for path in sorted(Path(args.runs_dir).glob("*/*/episode_*.json"))
        if (note := rescore(path, args.dry_run))
    ]
    for note in notes:
        print(note)
    verb = "would update" if args.dry_run else "updated"
    print(f"\n{verb} {len(notes)} episode(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
