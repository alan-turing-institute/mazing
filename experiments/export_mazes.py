"""Export the maze set a run used, and verify it later.

The mazes are already reproducible: `make_maze(seed, rows, cols, label)` is
deterministic, so pointing a second model at the same seeds gives it the
identical mazes. Nothing needs to be saved for that.

What this guards against is *generator drift*. `env/generation.py` changing —
a different carve order, a tweak to `_goal_cut_repairs` — silently changes what
seed 3 means, and a cross-model comparison made either side of that change is
comparing different mazes while looking identical in the run config. Exporting
the set pins what the seeds meant at the time, and `--verify` proves a later
checkout still produces them.

    uv run python experiments/export_mazes.py --seeds 0-9 -o mazes_9x9.json
    uv run python experiments/export_mazes.py --verify mazes_9x9.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env.generation import MazeLabel, make_maze  # noqa: E402


def parse_seeds(spec: str) -> list[int]:
    """"0-9", "0,3,7" or a mix."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def maze_record(seed: int, rows: int, cols: int, label: MazeLabel) -> dict:
    maze = make_maze(seed, rows, cols, label)
    # frozensets of tuples are unordered and unserialisable - sort into a
    # canonical list so the digest is stable across runs and machines.
    passages = sorted(
        tuple(sorted(tuple(cell) for cell in e)) for e in maze.passages
    )
    return {
        "seed": seed,
        "label": label.value,
        "rows": rows,
        "cols": cols,
        "start": list(maze.start),
        "goal": list(maze.goal),
        "passages": [[list(a), list(b)] for a, b in passages],
        # The oracle, recorded so a comparison can be checked without rerunning
        # generation at all.
        "oracle_reachable": maze.reachable,
        "shortest_path_length": maze.shortest_path_length,
        "reachable_component_size": maze.reachable_component_size,
    }


def build(seeds: list[int], rows: int, cols: int) -> dict:
    mazes = [
        maze_record(s, rows, cols, label)
        for s in seeds
        for label in (MazeLabel.SOLVABLE, MazeLabel.UNSOLVABLE)
    ]
    payload = {"rows": rows, "cols": cols, "seeds": seeds, "mazes": mazes}
    payload["digest"] = digest(payload)
    return payload


def digest(payload: dict) -> str:
    """Content hash of the maze set alone — not of the file, so re-exporting
    the same seeds from the same generator always gives the same digest."""
    body = json.dumps(payload["mazes"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def verify(path: Path) -> int:
    saved = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = build(saved["seeds"], saved["rows"], saved["cols"])
    if rebuilt["digest"] == saved["digest"]:
        print(f"OK  {len(saved['mazes'])} mazes match (digest {saved['digest']})")
        return 0
    print(f"MISMATCH  saved {saved['digest']} != rebuilt {rebuilt['digest']}")
    print("The generator no longer produces these mazes. Runs made before and")
    print("after this change are NOT comparable, even at the same seeds.")
    for old, new in zip(saved["mazes"], rebuilt["mazes"]):
        if old != new:
            print(f"  first differing maze: seed {old['seed']} {old['label']}")
            break
    return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="0-9", help='e.g. "0-9" or "0,3,7".')
    p.add_argument("--rows", type=int, default=9)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("-o", "--out", default="mazes.json")
    p.add_argument("--verify", metavar="FILE", help="Check a saved set still regenerates.")
    args = p.parse_args(argv)

    if args.verify:
        return verify(Path(args.verify))

    payload = build(parse_seeds(args.seeds), args.rows, args.cols)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['mazes'])} mazes to {args.out}  (digest {payload['digest']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
