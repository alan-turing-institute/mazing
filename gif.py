"""Render saved runs into a single animated GIF, one episode after another.

Because mazes are deterministic from their seed and every step is saved to the
episode JSON, the animation is reconstructed *after* the session — nothing is
captured during the run, and even killed (incomplete) episodes animate fine.

Requires the optional 'viz' extra (Pillow):
    uv run --extra viz python gif.py                 # newest run folder -> session.gif
    uv run --extra viz python gif.py --runs-dir runs/policy_43728752/<run> --out out.gif
    uv run --extra viz python gif.py --runs-dir runs   # every run under runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from env.generation import DIRECTIONS, MazeLabel, edge, make_maze

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover
    sys.exit(
        "Pillow is required for gif.py. Install the optional extra, e.g.:\n"
        "    uv run --extra viz python gif.py ..."
    )

# Dark palette, matching the terminal --watch aesthetic.
BG = (30, 30, 30)
WALL = (200, 200, 200)
REMOVED = (220, 60, 60)
AGENT = (60, 190, 220)
GOAL = (70, 200, 120)
START = (120, 120, 120)
TEXT = (230, 230, 230)
TEXT_DIM = (150, 150, 150)


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _has_wall(cell, d, rows, cols, passages) -> bool:
    dr, dc = DIRECTIONS[d]
    nc = (cell[0] + dr, cell[1] + dc)
    if not (0 <= nc[0] < rows and 0 <= nc[1] < cols):
        return True
    return edge(cell, nc) not in passages


def _draw_frame(maze, passages, position, caption, sub, cell, font, sub_font):
    rows, cols = maze.rows, maze.cols
    margin = cell
    cap_h = cell + 20
    w = cols * cell + 2 * margin
    h = rows * cell + 2 * margin + cap_h
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    tw = max(2, cell // 14)  # wall thickness

    def px(r, c):
        return margin + c * cell, margin + r * cell

    # Cell fills for start / goal / agent.
    def fill_cell(rc, color, inset):
        x0, y0 = px(*rc)
        d.rectangle(
            [x0 + inset, y0 + inset, x0 + cell - inset, y0 + cell - inset], fill=color
        )

    fill_cell(maze.start, START, cell // 4)
    fill_cell(maze.goal, GOAL, cell // 5)
    x0, y0 = px(*position)
    d.ellipse(
        [x0 + cell // 5, y0 + cell // 5, x0 + cell - cell // 5, y0 + cell - cell // 5],
        fill=AGENT,
    )

    # Walls: draw N and W of every cell (covers all interior walls), plus the
    # far right / bottom boundaries.
    for r in range(rows):
        for c in range(cols):
            x0, y0 = px(r, c)
            if _has_wall((r, c), "N", rows, cols, passages):
                d.line([x0, y0, x0 + cell, y0], fill=WALL, width=tw)
            if _has_wall((r, c), "W", rows, cols, passages):
                d.line([x0, y0, x0, y0 + cell], fill=WALL, width=tw)
    right = margin + cols * cell
    bottom = margin + rows * cell
    d.line([right, margin, right, bottom], fill=WALL, width=tw)
    d.line([margin, bottom, right, bottom], fill=WALL, width=tw)

    # Mark removed walls in red where they used to be.
    for e in passages:
        cells = sorted(e)
        if len(cells) != 2:
            continue
        (r1, c1), (r2, c2) = cells
        if edge((r1, c1), (r2, c2)) in maze.passages:
            continue  # was always open, not a removal
        if r1 == r2:  # horizontal neighbours -> removed wall was vertical
            x = margin + max(c1, c2) * cell
            y = margin + r1 * cell
            d.line([x, y + tw, x, y + cell - tw], fill=REMOVED, width=tw + 1)
        else:  # vertical neighbours -> removed wall was horizontal
            x = margin + c1 * cell
            y = margin + max(r1, r2) * cell
            d.line([x + tw, y, x + cell - tw, y], fill=REMOVED, width=tw + 1)

    # Caption.
    cy = margin + rows * cell + margin // 2
    color = REMOVED if "remove_wall" in caption else TEXT
    d.text((margin, cy), caption, fill=color, font=font)
    if sub:
        d.text((margin, cy + cell // 2 + 2), sub, fill=TEXT_DIM, font=sub_font)
    return img


def frames_for_episode(record, cell, font, sub_font):
    """Reconstruct the maze from its seed and replay the trajectory into frames."""
    m = record["maze"]
    maze = make_maze(m["seed"], rows=m["rows"], cols=m["cols"], label=MazeLabel(m["label"]))
    base = set(maze.passages)
    removed = record.get("episode_result", {}).get("removed_walls", [])

    def removed_up_to(step):
        passages = set(base)
        for rw in removed:
            if rw["step"] <= step:
                passages.add(edge(tuple(rw["from"]), tuple(rw["to"])))
        return passages

    ep_i = record.get("episode_index", 0)
    model = record.get("config", {}).get("model") or record.get("config", {}).get(
        "backend", "?"
    )
    header = f"episode {ep_i}  [{m['label']}]  goal {m['goal']}  model={model}"
    metrics = record.get("metrics") or {}
    label = metrics.get("label", "(incomplete)")

    frames, durations = [], []
    traj = record["trajectory"]
    for t in traj:
        pos = tuple(t["observation"]["position"])
        act = t["action"]
        act_str = (
            "(no action)"
            if act is None
            else f"{act['name']}({', '.join(f'{k}={v}' for k, v in act['arguments'].items())})"
        )
        caption = f"step {t['step']}: {act_str} -> {t['result']}"
        frames.append(
            _draw_frame(maze, removed_up_to(t["step"]), pos, caption, header, cell, font, sub_font)
        )
        durations.append(700 if t["result"] == "wall_removed" else 260)

    # Final resting frame.
    final_pos = tuple(record.get("episode_result", {}).get("final_position", maze.start))
    passages = removed_up_to(traj[-1]["step"] if traj else 0)
    caption = f"result: {label}  |  steps {metrics.get('total_steps', len(traj))}  |  reached_goal={metrics.get('reached_goal')}"
    frames.append(_draw_frame(maze, passages, final_pos, caption, header, cell, font, sub_font))
    durations.append(1400)
    return frames, durations


def find_episode_files(runs_dir: Path) -> list[Path]:
    return sorted(runs_dir.rglob("episode_*.json"))


def newest_run_dir(root: Path) -> Path | None:
    """The most recently modified folder that directly contains episode files."""
    run_dirs = {p.parent for p in root.rglob("episode_*.json")}
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def build_gif(episode_files, out_path: Path, cell: int = 40) -> int:
    font, sub_font = _font(max(12, cell // 3)), _font(max(11, cell // 4))
    all_frames, all_durations = [], []
    for path in episode_files:
        try:
            record = json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            continue
        if not record.get("trajectory"):
            continue
        frames, durations = frames_for_episode(record, cell, font, sub_font)
        all_frames.extend(frames)
        all_durations.extend(durations)
    if not all_frames:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_frames[0].save(
        out_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=all_durations,
        loop=0,
        disposal=2,
    )
    return len(all_frames)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        default=None,
        help="Run folder (or a tree of them) to animate. Default: newest run under runs/.",
    )
    p.add_argument("--out", default=None, help="Output GIF path. Default: <runs-dir>/session.gif.")
    p.add_argument("--cell", type=int, default=40, help="Cell size in pixels.")
    args = p.parse_args(argv)

    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
    else:
        runs_dir = newest_run_dir(Path("runs"))
        if runs_dir is None:
            sys.exit("No episodes found under runs/. Run run.py first.")
        print(f"Animating newest run: {runs_dir}")

    files = find_episode_files(runs_dir)
    if not files:
        sys.exit(f"No episode_*.json found under {runs_dir}/")

    out_path = Path(args.out) if args.out else runs_dir / "session.gif"
    n = build_gif(files, out_path, cell=args.cell)
    if n:
        print(f"Wrote {n} frames from {len(files)} episode(s) to {out_path}")
    else:
        print("No frames produced (no trajectories found).")


if __name__ == "__main__":
    main()
