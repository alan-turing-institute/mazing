"""Entry point: run the calibrated-restraint experiment.

Example (no model or API key needed):
    uv run python run.py --backend dummy --n-mazes 2

Against any OpenAI-compatible endpoint:
    uv run python run.py --backend openai \\
        --model gpt-4o-mini --base-url https://api.openai.com/v1 \\
        --n-mazes 2 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import config
from backends import (
    DummyExplorerBackend,
    DummyRemoverBackend,
    OpenAICompatibleBackend,
)
from env.generation import MazeLabel, make_maze
from metrics import compute_metrics, summary_row
from runner import run_episode
from runner.render import MazeWatcher


def build_maze_specs(n: int, seed: int) -> list[tuple[int, MazeLabel]]:
    """Deterministic mix; guarantees >=1 solvable and >=1 unsolvable for n>=2."""
    specs = []
    for i in range(n):
        label = MazeLabel.SOLVABLE if i % 2 == 0 else MazeLabel.UNSOLVABLE
        specs.append((seed + i, label))
    return specs


def make_backend(args):
    if args.backend == "dummy":
        return (
            DummyRemoverBackend()
            if args.dummy_policy == "remover"
            else DummyExplorerBackend()
        )
    if args.backend == "ollama":
        # Ollama's OpenAI-compatible endpoint. No real API key needed, and it
        # doesn't honour tool_choice="required", so ask nicely with "auto".
        base_url = args.base_url or config.OLLAMA_BASE_URL
        model = args.model or config.DEFAULT_OLLAMA_MODEL
        return OpenAICompatibleBackend(
            model=model,
            base_url=base_url,
            api_key=args.api_key or "ollama",
            seed=args.seed,
            tool_choice="auto",
        )
    if args.backend == "openai":
        if not args.model:
            sys.exit("--model is required for --backend openai")
        base_url = args.base_url or config.DEFAULT_BASE_URL
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAICompatibleBackend(
            model=args.model,
            base_url=base_url,
            api_key=api_key,
            seed=args.seed,
        )
    sys.exit(f"unknown backend: {args.backend}")


def maze_meta(maze) -> dict:
    return {
        "label": maze.label.value,
        "seed": maze.seed,
        "rows": maze.rows,
        "cols": maze.cols,
        "start": list(maze.start),
        "goal": list(maze.goal),
        "oracle_reachable": maze.reachable,
        "shortest_path_length": maze.shortest_path_length,
        "reachable_component_size": maze.reachable_component_size,
    }


def print_summary(rows: list[dict]) -> None:
    cols = [
        "episode",
        "maze_label",
        "oracle_reachable",
        "shortest_path",
        "wall_removed",
        "label",
        "reached_goal",
        "total_steps",
        "steps_before_removal",
        "cells_before_removal",
        "reachable_cells",
        "explored_frac_before_removal",
    ]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="dummy", choices=["dummy", "ollama", "openai"])
    p.add_argument(
        "--dummy-policy",
        default="explorer",
        choices=["explorer", "remover"],
        help="Which scripted policy to use when --backend dummy.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name (e.g. 'qwen3:4b' for ollama). Defaults to "
        f"'{config.DEFAULT_OLLAMA_MODEL}' for --backend ollama.",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults per backend "
        f"(ollama: {config.OLLAMA_BASE_URL}).",
    )
    p.add_argument("--api-key", default=None, help="Overrides $OPENAI_API_KEY.")
    p.add_argument("--n-mazes", type=int, default=config.DEFAULT_N_MAZES)
    p.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p.add_argument("--max-steps", type=int, default=config.DEFAULT_MAX_STEPS)
    p.add_argument("--rows", type=int, default=config.DEFAULT_ROWS)
    p.add_argument("--cols", type=int, default=config.DEFAULT_COLS)
    p.add_argument("--out-dir", default="runs")
    p.add_argument(
        "--task-prompt",
        default=config.TASK_PROMPT_FILE,
        help="Path to the task/mechanics document (default: prompts/task.md).",
    )
    p.add_argument(
        "--policy",
        default=config.POLICY_FILE,
        help="Path to the policy document under test (default: prompts/policy.md).",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Live-render the maze after every step so you can watch the agent explore.",
    )
    p.add_argument(
        "--watch-delay",
        type=float,
        default=0.4,
        help="Seconds to pause after each step when --watch is set.",
    )
    p.add_argument(
        "--no-reasoning",
        action="store_true",
        help="With --watch, do NOT print the model's reasoning under the maze.",
    )
    p.add_argument(
        "--reasoning-lines",
        type=int,
        default=12,
        help="Max lines of reasoning to show under the maze when watching.",
    )
    p.add_argument(
        "--gif",
        action="store_true",
        help="At the end, render all episodes of this run into one GIF "
        "(<run-dir>/session.gif). Requires the 'viz' extra: uv run --extra viz.",
    )
    args = p.parse_args(argv)

    backend = make_backend(args)
    specs = build_maze_specs(args.n_mazes, args.seed)

    # Compose the system prompt from the (independently swappable) documents.
    task_prompt = config.load_prompt(args.task_prompt)
    policy = config.load_prompt(args.policy)
    system_prompt = config.build_system_prompt(args.task_prompt, args.policy)

    # Group runs by policy: same policy text -> same folder, an edited policy
    # -> a new folder (via a short content hash). A copy of the policy is
    # dropped in so the hash stays decodable.
    policy_hash = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:8]
    policy_dir = Path(args.out_dir) / f"{Path(args.policy).stem}_{policy_hash}"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "policy.md").write_text(policy + "\n")

    model_tag = getattr(backend, "model", None) or args.dummy_policy
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "-", model_tag)
    run_id = f"{args.backend}_{safe_model}_seed{args.seed}_{int(time.time())}"
    out_dir = policy_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "backend": args.backend,
        "dummy_policy": args.dummy_policy if args.backend == "dummy" else None,
        "model": getattr(backend, "model", None),
        "base_url": getattr(backend, "base_url", None),
        "n_mazes": args.n_mazes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "rows": args.rows,
        "cols": args.cols,
        "task_prompt_file": str(args.task_prompt),
        "policy_file": str(args.policy),
        "task_prompt": task_prompt,
        "policy": policy,
        "policy_hash": policy_hash,
    }
    # Write the run config once, upfront, so a run is identifiable even if it is
    # killed before the first episode finishes.
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    def assemble_record(i, maze, episode, complete):
        """Build the on-disk episode record. Metrics are only meaningful once
        the episode has finished, so partial checkpoints leave them null."""
        return {
            "config": run_config,
            "episode_index": i,
            "complete": complete,
            "maze": maze_meta(maze),
            "trajectory": episode["trajectory"],
            "episode_result": {
                k: episode[k]
                for k in (
                    "reached_goal",
                    "total_steps",
                    "removed_walls",
                    "end_reason",
                    "final_position",
                )
            },
            "metrics": compute_metrics(episode, maze) if complete else None,
        }

    print(f"Writing results to {out_dir}/ (saved after every step)")
    summary_rows = []
    for i, (mseed, label) in enumerate(specs):
        maze = make_maze(mseed, rows=args.rows, cols=args.cols, label=label)
        episode_path = out_dir / f"episode_{i:03d}.json"

        watcher = None
        if args.watch:
            title = f"episode {i}  [{label.value}]  seed={mseed}  model={getattr(backend, 'model', args.backend)}"
            watcher = MazeWatcher(
                title,
                show_reasoning=not args.no_reasoning,
                reasoning_lines=args.reasoning_lines,
            )

        def checkpoint(partial, i=i, maze=maze, path=episode_path):
            # Flush the episode to disk after every step so a kill loses nothing.
            path.write_text(json.dumps(assemble_record(i, maze, partial, False), indent=2))

        episode = run_episode(
            maze,
            backend,
            max_steps=args.max_steps,
            system_prompt=system_prompt,
            on_step=watcher,
            on_progress=checkpoint,
            step_delay=args.watch_delay if args.watch else 0.0,
        )
        metrics = compute_metrics(episode, maze)

        # Final, complete record (overwrites the last checkpoint).
        episode_path.write_text(
            json.dumps(assemble_record(i, maze, episode, True), indent=2)
        )
        summary_rows.append(summary_row(i, maze, metrics))

        # Rewrite the aggregate summary after each episode so it is always current.
        with (out_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nSaved {len(summary_rows)} episode(s) to {out_dir}/")
    print_summary(summary_rows)

    if args.gif:
        try:
            from gif import build_gif, find_episode_files

            gif_path = out_dir / "session.gif"
            n = build_gif(find_episode_files(out_dir), gif_path)
            print(f"\nWrote GIF ({n} frames) to {gif_path}")
        except SystemExit as e:
            print(f"\nCould not build GIF: {e}")


if __name__ == "__main__":
    main()
