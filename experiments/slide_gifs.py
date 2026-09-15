"""Render one episode as a pair of GIFs: the true maze, and what the agent knew.

Made for talks. The point of the pair is that the two views answer different
questions about the same episode:

* `_truth.gif`  — the maze as it really is, with the agent's path. This is the
  view `experiments/watch_run.py --emoji` draws, and it is the only view in
  which "a wall-free route existed" is even visible.
* `_knowledge.gif` — the same episode drawn from the agent's *observations
  only*: a cell is drawn once the agent has seen it, and a wall once the agent
  has observed it. Everything else is fog.

The second view is reconstructed from the recorded observations, never from the
true maze, so it cannot accidentally show the agent something it did not see.
That is what makes it honest as an illustration: a removal that looks obviously
premature against the full maze often looks locally reasonable against the fog,
and the gap between the two is the thing worth putting on a slide.

Requires the optional 'viz' extra (Pillow):

    uv run --extra viz python experiments/slide_gifs.py \
        --episode runs/<policy>/<run>/episode_000.json --out-prefix /tmp/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env.generation import DIRECTIONS, MazeLabel, edge, make_maze  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover
    sys.exit(
        "Pillow is required. Install the optional extra, e.g.:\n"
        "    uv run --extra viz python experiments/slide_gifs.py ..."
    )

# Palette matches gif.py so the two renderers look like one project.
BG = (30, 30, 30)
WALL = (200, 200, 200)
REMOVED = (220, 60, 60)
TEXT = (230, 230, 230)
TEXT_DIM = (150, 150, 150)
# Knowledge view only: unknown territory, and cells the agent has walked.
FOG = (18, 18, 18)
KNOWN = (44, 44, 44)
TRAIL = (38, 62, 74)

EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
# Apple Color Emoji is a bitmap font with a native strike at 160px; rendering at
# that size and downscaling is far cleaner than asking for an arbitrary size.
EMOJI_NATIVE = 160
AGENT_EMOJI, GOAL_EMOJI, START_EMOJI = "🤖", "🏁", "🏠"


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
    return ImageFont.load_default(size)


def _emoji_tile(char: str, size: int):
    """One emoji as an RGBA tile, or None if colour emoji is unavailable."""
    try:
        font = ImageFont.truetype(EMOJI_FONT, EMOJI_NATIVE)
    except OSError:
        return None
    img = Image.new("RGBA", (EMOJI_NATIVE * 2, EMOJI_NATIVE * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        d.text((0, 0), char, font=font, embedded_color=True)
    except TypeError:  # Pillow too old for embedded_color
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    return img.crop(bbox).resize((size, size), Image.LANCZOS)


def _wall(walls: dict | None, d: str) -> bool:
    return bool(walls) and walls.get(d) == "wall"


def _true_has_wall(cell, d, rows, cols, passages) -> bool:
    dr, dc = DIRECTIONS[d]
    nc = (cell[0] + dr, cell[1] + dc)
    if not (0 <= nc[0] < rows and 0 <= nc[1] < cols):
        return True
    return edge(cell, nc) not in passages


class Renderer:
    def __init__(self, maze, cell: int, captions=()):
        self.maze = maze
        self.cell = cell
        self.margin = cell
        self.cap_h = cell + 26
        self.h = maze.rows * cell + 2 * self.margin + self.cap_h
        self.font = _font(max(13, cell // 2))
        self.sub_font = _font(max(11, cell // 5 * 2 - 4))
        # Widen the canvas to whatever the longest caption needs. Sizing to the
        # maze alone clips the step text, which is the half of the frame that
        # says what the agent actually did.
        board = maze.cols * cell + 2 * self.margin
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        text_w = 0
        for text, font in captions:
            text_w = max(text_w, probe.textlength(text, font=font))
        self.w = int(max(board, text_w + 2 * self.margin))
        # Centre the board when the captions made the canvas wider than it.
        self.board_x = (self.w - maze.cols * cell) // 2
        tile = max(10, cell - max(4, cell // 6))
        self.tiles = {
            "agent": _emoji_tile(AGENT_EMOJI, tile),
            "goal": _emoji_tile(GOAL_EMOJI, tile),
            "start": _emoji_tile(START_EMOJI, tile),
        }
        self.emoji_ok = all(t is not None for t in self.tiles.values())

    def px(self, r, c):
        return self.board_x + c * self.cell, self.margin + r * self.cell

    def _blit(self, img, kind, rc):
        x0, y0 = self.px(*rc)
        tile = self.tiles[kind]
        if tile is None:  # fallback: a plain coloured marker
            colour = {"agent": (60, 190, 220), "goal": (70, 200, 120),
                      "start": (120, 120, 120)}[kind]
            ImageDraw.Draw(img).ellipse(
                [x0 + self.cell // 5, y0 + self.cell // 5,
                 x0 + self.cell - self.cell // 5, y0 + self.cell - self.cell // 5],
                fill=colour,
            )
            return
        off = (self.cell - tile.width) // 2
        img.paste(tile, (x0 + off, y0 + off), tile)

    def frame(self, *, passages, position, caption, header, visited,
              known=None, removed_edges=()):
        """One frame. `known` = None renders the true maze; a dict of
        cell -> observed walls renders the agent's knowledge instead."""
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        tw = max(2, self.cell // 14)
        rows, cols = self.maze.rows, self.maze.cols

        # Cell fills: fog vs seen (knowledge view), trail in both.
        for r in range(rows):
            for c in range(cols):
                x0, y0 = self.px(r, c)
                if known is not None and (r, c) not in known:
                    fill = FOG
                elif (r, c) in visited:
                    fill = TRAIL
                elif known is not None:
                    fill = KNOWN
                else:
                    continue
                d.rectangle([x0, y0, x0 + self.cell, y0 + self.cell], fill=fill)

        self._blit(img, "start", self.maze.start)
        # The goal's coordinates are in every observation, so the agent always
        # knows where the centre is — it is drawn even while fogged.
        self._blit(img, "goal", self.maze.goal)
        self._blit(img, "agent", position)

        # Walls.
        if known is None:
            for r in range(rows):
                for c in range(cols):
                    x0, y0 = self.px(r, c)
                    if _true_has_wall((r, c), "N", rows, cols, passages):
                        d.line([x0, y0, x0 + self.cell, y0], fill=WALL, width=tw)
                    if _true_has_wall((r, c), "W", rows, cols, passages):
                        d.line([x0, y0, x0, y0 + self.cell], fill=WALL, width=tw)
            right = self.board_x + cols * self.cell
            bottom = self.margin + rows * self.cell
            d.line([right, self.margin, right, bottom], fill=WALL, width=tw)
            d.line([self.board_x, bottom, right, bottom], fill=WALL, width=tw)
        else:
            # Only walls the agent has actually observed, drawn per known cell.
            for (r, c), walls in known.items():
                x0, y0 = self.px(r, c)
                if _wall(walls, "N"):
                    d.line([x0, y0, x0 + self.cell, y0], fill=WALL, width=tw)
                if _wall(walls, "W"):
                    d.line([x0, y0, x0, y0 + self.cell], fill=WALL, width=tw)
                if _wall(walls, "S"):
                    d.line([x0, y0 + self.cell, x0 + self.cell, y0 + self.cell],
                           fill=WALL, width=tw)
                if _wall(walls, "E"):
                    d.line([x0 + self.cell, y0, x0 + self.cell, y0 + self.cell],
                           fill=WALL, width=tw)

        # Removed walls, in red, where the wall used to be.
        for a, b in removed_edges:
            (r1, c1), (r2, c2) = sorted((a, b))
            if r1 == r2:
                x = self.board_x + max(c1, c2) * self.cell
                y = self.margin + r1 * self.cell
                d.line([x, y + tw, x, y + self.cell - tw], fill=REMOVED, width=tw + 1)
            else:
                x = self.board_x + c1 * self.cell
                y = self.margin + max(r1, r2) * self.cell
                d.line([x + tw, y, x + self.cell - tw, y], fill=REMOVED, width=tw + 1)

        cy = self.margin + rows * self.cell + self.margin // 2
        d.text((self.margin, cy), header, fill=TEXT_DIM, font=self.sub_font)
        colour = REMOVED if "remove_wall" in caption else TEXT
        d.text((self.margin, cy + self.cell // 2), caption, fill=colour, font=self.font)
        return img


def build(record: dict, cell: int):
    m = record["maze"]
    maze = make_maze(m["seed"], rows=m["rows"], cols=m["cols"],
                     label=MazeLabel(m["label"]))
    base = set(maze.passages)
    removals = record.get("episode_result", {}).get("removed_walls", [])
    metrics = record.get("metrics") or {}
    model = record.get("config", {}).get("model", "?")
    # On a sealed maze there is no wall-free path at all, and "None steps"
    # reads as a missing value rather than as the point of the condition.
    route = (
        f"shortest wall-free path: {m['shortest_path_length']} steps"
        if m.get("shortest_path_length") is not None
        else "goal sealed: no wall-free path exists"
    )
    header = (
        f"{model}  |  seed {m['seed']}  |  {m['label']}  |  {route}  |  "
        f"{metrics.get('label')}"
    )

    def caption_for(t):
        act = t["action"]
        act_str = ("(no action)" if act is None else
                   f"{act['name']}(" +
                   ", ".join(f"{k}={v}" for k, v in act["arguments"].items()
                             if k != "justification") + ")")
        return f"step {t['step']}: {act_str} -> {t['result']}"

    final_caption = (
        f"{metrics.get('label')}  |  {metrics.get('total_steps')} steps  |  "
        f"reached_goal={metrics.get('reached_goal')}"
    )
    probe_font = _font(max(13, cell // 2))
    probe_sub = _font(max(11, cell // 5 * 2 - 4))
    measured = [(header, probe_sub), (final_caption, probe_font)]
    measured += [(caption_for(t), probe_font) for t in record["trajectory"]]
    rend = Renderer(maze, cell, captions=measured)

    def state_at(step):
        passages, removed_edges = set(base), []
        for rw in removals:
            if rw["step"] <= step:
                a, b = tuple(rw["from"]), tuple(rw["to"])
                passages.add(edge(a, b))
                removed_edges.append((a, b))
        return passages, removed_edges

    truth, know, durations = [], [], []
    known: dict[tuple, dict] = {}
    visited: set = set()

    for t in record["trajectory"]:
        obs = t["observation"]
        pos = tuple(obs["position"])
        visited.add(pos)
        # Absorb this observation into the agent's knowledge BEFORE drawing, so
        # the frame shows what it knew when it chose this step's action.
        known[pos] = obs["current_cell_walls"]
        for _d, info in (obs.get("adjacent_cells") or {}).items():
            if info:
                known[tuple(info["position"])] = info["walls"]

        caption = caption_for(t)
        passages, removed_edges = state_at(t["step"])

        truth.append(rend.frame(passages=passages, position=pos, caption=caption,
                                header=header, visited=set(visited),
                                removed_edges=removed_edges))
        know.append(rend.frame(passages=passages, position=pos, caption=caption,
                               header=header, visited=set(visited),
                               known=dict(known), removed_edges=removed_edges))
        durations.append(900 if t["result"] == "wall_removed" else 300)

    final = tuple(record["episode_result"]["final_position"])
    passages, removed_edges = state_at(10**9)
    cap = final_caption
    truth.append(rend.frame(passages=passages, position=final, caption=cap,
                            header=header, visited=visited,
                            removed_edges=removed_edges))
    know.append(rend.frame(passages=passages, position=final, caption=cap,
                           header=header, visited=visited, known=known,
                           removed_edges=removed_edges))
    durations.append(2500)
    return truth, know, durations, rend.emoji_ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", required=True, type=Path,
                   help="Path to an episode_NNN.json")
    p.add_argument("--out-prefix", required=True,
                   help="Output prefix; writes <prefix>_truth.gif and <prefix>_knowledge.gif")
    p.add_argument("--cell", type=int, default=56, help="Pixels per maze cell.")
    args = p.parse_args(argv)

    record = json.loads(args.episode.read_text())
    truth, know, durations, emoji_ok = build(record, args.cell)
    for name, frames in (("truth", truth), ("knowledge", know)):
        out = Path(f"{args.out_prefix}_{name}.gif")
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True)
        print(f"wrote {out}  ({len(frames)} frames, {out.stat().st_size / 1024:.0f} KB)")
    if not emoji_ok:
        print("note: colour emoji unavailable — drew coloured markers instead",
              file=sys.stderr)


if __name__ == "__main__":
    main()
