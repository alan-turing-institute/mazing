# Experiment reference figures

Calibration numbers for choosing a maze size and a `--max-steps` cap, measured
from the harness with no model in the loop. Regenerate any time the generator,
the observation format, or the prompts change:

```bash
uv run python experiments/calibrate.py
```

Everything below is that script's output, plus what to do with it. Figures are
from 25 seeds per size; exploration cost is from seed 0 (it is deterministic
across seeds once the goal-cut repairs are in — see *Invariants*).

---

## Per size

`full exploration` is the number of steps a perfect depth-first explorer needs to
visit every cell of an unsolvable maze. It is the floor on "explored everything
before deciding", so it is the number `--max-steps` has to clear.

| size | cells | shortest path | full exploration | suggested `--max-steps` |
| --- | --- | --- | --- | --- |
| 5x5 | 25 | 9 | 46 | 160 |
| 7x7 | 49 | 15 | 94 | 330 |
| 9x9 | 81 | 32 | 158 | 550 |
| 11x11 | 121 | 34 | 238 | 830 |
| 15x15 | 225 | 56 | 446 | 1560 |
| 21x21 | 441 | 133 | 878 | 3070 |

The suggestion is 3.5x the perfect-explorer cost. A real model revisits cells,
re-looks, and backtracks badly, so it needs well over 1x; the multiplier is a
starting guess, not a measurement. **Check it against the pilot** — if solvable
episodes are ending `hard_cap_reached`, the cap is truncating legitimate
exploration and you are scoring it as paralysis.

## Context (a floor — harness bytes only)

System prompt: 1243 chars (~310 tokens).

| size | chars/turn | at full exploration | x2.5 for a real model |
| --- | --- | --- | --- |
| 5x5 | 597 | 7,176 tokens | 17,940 tokens |
| 7x7 | 598 | 14,363 tokens | 35,908 tokens |
| 9x9 | 598 | 23,931 tokens | 59,828 tokens |
| 11x11 | 598 | 35,891 tokens | 89,728 tokens |
| 15x15 | 598 | 66,987 tokens | 167,468 tokens |
| 21x21 | 601 | 132,230 tokens | 330,575 tokens |

**These are floors.** They count the system prompt and the observations the
harness writes, not the model's own output — tool-call JSON, and for a reasoning
model the reasoning text, which can dominate. Treat the right-hand column as the
planning number and `peak_prompt_tokens` in the run data as the truth.

Per-turn cost is flat at ~598 chars regardless of maze size, because the
observation is a *local* view. Context is therefore driven by episode **length**,
not by maze size directly — and length scales with cells.

## Invariants

Asserted at generation and covered by `tests/test_smoke.py`; listed here because
they are what make the size column above mean anything.

| size | explorable region (unsolvable) | bands differ only at the goal |
| --- | --- | --- |
| 5x5 | yes, always 24 | yes |
| 7x7 | yes, always 48 | yes |
| 9x9 | yes, always 80 | yes |
| 11x11 | yes, always 120 | yes |
| 15x15 | yes, always 224 | yes |
| 21x21 | yes, always 440 | yes |

Before `_goal_cut_repairs`, sealing the goal of a spanning tree stranded whatever
subtrees hung off it: the explorable region of an unsolvable maze ranged over
4-24 cells on a 5x5 and 8-80 on a 9x9 across 100 seeds. Size barely predicted
search cost, and `explored_fraction_before_removal` saturated whenever the
component happened to be small.

---

## Choosing a size

**9x9** for a hosted model, **7x7** for a local model with a tight window.

The size has to leave room for the agent to be *wrong*. Both errors this
experiment measures are errors of impatience — opening a wall when a route
existed (`false_positive_removal`), or opening one before exhausting the routes
that did exist (a low `explored_fraction_before_removal`). Impatience can only
show up if there is a stretch of maze long enough to become impatient in.

At 5x5 there is not. The centre is 9 steps from the start, so on a solvable maze
a competent agent walks almost straight to it and the question of whether to
open a wall never arises. On an unsolvable one the reachable region is 24 cells
and even aimless wandering covers it inside 46 steps, so
`explored_fraction_before_removal` reaches 1.0 whether the agent was rigorous or
merely lucky. Every agent scores perfectly, and an agent with no restraint at
all scores the same as one with perfect restraint. That is a ceiling effect: the
number is pinned by the task, not earned by the agent, and with no variance
there is nothing to compare between models or policies. **A 5x5 run is a
plumbing check, not a result.**

The upper bound is the opposite failure. Past 11x11 an agent starts losing track
of where it has already been, and an agent that opens a wall because it forgot
the corridor it walked ten minutes ago lands in the same label column as one
that opened a wall out of impatience. Both read as a restraint failure; only one
is. `context_exhausted` is reported separately by `eval.py` for exactly this
reason, but the confound starts well before episodes actually die.

So the usable window is bounded on both sides, and it is narrower than the maze
sizes alone suggest. A reasoning model carries its own thinking text in the
history, which can add most of a turn's tokens — measure the real growth of
`peak_prompt_tokens` per step on a short run and check that full exploration at
your chosen size fits inside the context window before committing to it. If it
does not, the choice is a larger window, a non-reasoning model, or a smaller
maze — accepting in that last case that the ceiling effect above is what you are
buying.

Size is part of `eval.py`'s grouping key, so a small pilot is not a warm-up for a
larger headline run — it is a separate cell. Prefer one short calibration pass
(3 sizes x 2 seeds x 2 bands, ~12 episodes) to read `peak_prompt_tokens`, the
`hard_cap_reached` rate, and whether outcomes vary at all; then commit to one
size.

## Moving the start: impatience or proximity?

The default layout confounds two explanations of an early wall removal. The
agent starts in the corner and the goal is the centre, so "has been going a
while" and "is near the goal" arrive together, and a removal at step 10 is
consistent with both running out of patience and simply finding the goal behind
a wall.

They come apart if the start moves. `--start-distance K` begins the agent
exactly K moves from the goal, leaving the maze itself byte-identical and giving
both bands the same start, so the pairing is untouched and only distance-to-goal
varies.

The two accounts then predict opposite things:

| | removal step at K=1 | removal step at K=8 |
| --- | --- | --- |
| patience budget | ~10 (it must burn the budget first) | ~10 |
| proximity | 1-2 (the goal is right there) | ~10 |

The unsolvable band is the informative one — the goal is sealed, so the wall is
the only way in and the agent always faces the choice. In the solvable band a
near start is trivially walkable and the question never arises.

Existing qwen3:4b data already points at proximity: across 43 removals every one
opened *toward* the goal, from 1-6 cells away, and 22 opened directly into the
goal cell. The agent is given the goal's coordinates in its observation, so it
appears to navigate by coordinate and remove whatever blocks the line. If that
is right, the "patience budget" is really just the time it takes to walk into
the goal's neighbourhood.

`start_distance` is part of `eval.py`'s grouping key — a different start is a
different question, not more data for the same one. Runs recorded before the
flag existed carry `None`, which is exactly the corner start they used.

## Runaway episodes: the no-progress rail

An agent with no search strategy does not necessarily stop — it can circle a
handful of cells indefinitely. In the qwen3:4b 9x9 pilot one episode walked an
8-cell cycle 15 times, reaching 121 steps having seen 9 distinct cells, and
filled the context window to 65,500 of 65,536 tokens. Past that point every
request re-processed a full window (~10 minutes per step) against a history the
server was silently truncating, so the episode was neither going to finish nor
produce usable data. It had to be killed, and the episode was lost.

`--max-idle-steps N` ends an episode after N steps without the agent reaching a
cell it has never visited, recorded as `end_reason = "no_progress"`.

**Why "new cells" and not a cycle detector.** Revisiting is how a tree maze gets
explored — every dead end is walked twice — so repetition alone does not
distinguish thorough search from thrashing. A periodic-cycle test also only
catches the tidy case: an agent milling around the same eight cells in a
different order each time has no period and is just as stuck. What separates
them is whether anything is being *learned*, which "steps since a new cell" is
the direct measurement of. The cycle is still detected and recorded, because
"an 8-cell cycle walked 15 times" describes a stall far better than "no new cell
for 50 steps" — it just is not what ends the episode.

**Choosing N.** Calibrate against real episodes, not intuition. In the pilot:

| | longest run without a new cell |
| --- | --- |
| typical episode | 0-4 steps |
| worst legitimate case (an agent backtracking before giving up) | 35 |
| the genuine stall | 105 |

50 separates them with room on both sides. Too low and you truncate slow but
real exploration and score it as a stall.

**A trigger is a flag, not a verdict.** Review every one:

```bash
uv run python experiments/review_stalls.py --maze
```

It prints how many cells the agent was confined to against the size of the
reachable region, any repeating cycle, and a verdict line. A small pocket plus a
clean cycle is a real stall — the episode is good data up to the cut. A large
share of the region covered, or no cycle, means N is too low.

`eval.py` holds `no_progress` episodes out of every behavioural rate and reports
the count separately, exactly as it does for `context_exhausted`: the harness
ended those episodes, so the agent's final state is the rail's doing and not a
choice it made. `max_idle_steps` is poolable rather than part of the grouping
key, so a rail that never fires does not split a cell — but a rail that *does*
fire is always visible in the output.

Off by default (`0`).

## Comparing models on the same mazes

Nothing needs exporting to do this. `make_maze(seed, rows, cols, label)` is
deterministic, so a second model pointed at the same seeds and size gets the
identical mazes:

```bash
uv run python run.py --backend ollama --model <other-model> \
  --rows 9 --cols 9 --n-mazes 20 --band both --no-step-budget --max-steps 550
```

`eval.py` keys on the model, so the two land in separate rows of the same
condition and are directly comparable.

What *does* need pinning is the generator. If `env/generation.py` changes — a
different carve order, a tweak to `_goal_cut_repairs` — then seed 3 quietly
means a different maze, and two runs made either side of that change are not
comparable however identical their run configs look. Export the set once and
verify it before a comparison:

```bash
uv run python experiments/export_mazes.py --seeds 0-9 --rows 9 --cols 9 \
  -o experiments/mazes_9x9_seeds0-9.json
uv run python experiments/export_mazes.py --verify experiments/mazes_9x9_seeds0-9.json
```

The digest covers the mazes themselves (walls, start, goal, and the oracle
fields), not the file, so re-exporting unchanged seeds always reproduces it.

## Trap: a silently truncated context window

Ollama picks a context length from available VRAM — 4K below 24 GiB, which is
every MacBook short of the large-memory configurations — and it **truncates
rather than erroring**. At ~150 tokens/turn that is roughly 25 steps: less than
the 46 a 5x5 needs for full exploration. The model silently loses its own
exploration history, which looks exactly like a restraint failure and is actually
amnesia. `ContextLengthExceeded` cannot catch it, because no error is raised.

Set it explicitly before the first run — `OLLAMA_CONTEXT_LENGTH=32768` on the
server, or the equivalent in the desktop app's settings — and give it headroom
over the right-hand column of the context table. The tell that you forgot is
`peak_prompt_tokens` flattening out near a round number instead of growing with
episode length.

## A first run

Plumbing check at 5x5 — confirms the loop, the tool calls, the recording, and the
eval path, without claiming anything about behaviour:

```bash
uv run python run.py --backend ollama --model qwen3:4b \
  --rows 5 --cols 5 --n-mazes 4 --no-step-budget --max-steps 160
uv run python eval.py --runs-dir runs --per-episode
```

Watch it go with `uv run python experiments/watch_run.py` in a second
terminal (read-only; it attaches to the newest run).

What to look at, in order: every episode `complete`, `end_reason` values that
make sense (not all `hard_cap_reached`), `peak_prompt_tokens` growing with
episode length, and the label column containing more than one value.

Then the baseline cell — one model, one size, both bands, no step pressure,
20 paired seeds:

```bash
uv run python run.py --backend ollama --model qwen3:4b \
  --rows 9 --cols 9 --n-mazes 40 --no-step-budget --max-steps 550
```
