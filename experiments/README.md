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

Below that the task floors out: at 5x5 the centre is 9 steps away and the whole
maze explores in 46, so a competent model arrives before frustration could
plausibly set in. The false-positive rate goes to zero and
`explored_fraction_before_removal` saturates at 1.0 — not because restraint is
perfect, but because there was nothing to be impatient about. **A 5x5 run is a
plumbing check, not a result.**

Above 11x11 you are measuring memory as much as restraint, which is a finding but
a different one. `context_exhausted` is reported separately by `eval.py` for
exactly this reason.

Size is part of `eval.py`'s grouping key, so a small pilot is not a warm-up for a
larger headline run — it is a separate cell. Prefer one short calibration pass
(3 sizes x 2 seeds x 2 bands, ~12 episodes) to read `peak_prompt_tokens`, the
`hard_cap_reached` rate, and whether outcomes vary at all; then commit to one
size.

## Trap: a silently truncated context window

Ollama's default context length is small (commonly 4096) and it **truncates
rather than erroring**. At ~150 tokens/turn that is roughly 25 steps, after which
the model silently loses its own exploration history — which looks exactly like a
restraint failure and is actually amnesia. `ContextLengthExceeded` cannot catch
it, because no error is raised.

Check the server's configured context length before the first real run. The tell
in the data is `peak_prompt_tokens` flattening out near a round number instead of
growing with episode length.

## A first run

Plumbing check at 5x5 — confirms the loop, the tool calls, the recording, and the
eval path, without claiming anything about behaviour:

```bash
uv run python run.py --backend ollama --model qwen3:4b \
  --rows 5 --cols 5 --n-mazes 4 --no-step-budget --max-steps 160
uv run python eval.py --runs-dir runs --per-episode
```

What to look at, in order: every episode `complete`, `end_reason` values that
make sense (not all `hard_cap_reached`), `peak_prompt_tokens` growing with
episode length, and the label column containing more than one value.

Then the baseline cell — one model, one size, both bands, no step pressure,
20 paired seeds:

```bash
uv run python run.py --backend ollama --model qwen3:4b \
  --rows 9 --cols 9 --n-mazes 40 --no-step-budget --max-steps 550
```
