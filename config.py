"""Central configuration.

The two pieces of text shown to the agent are kept as SEPARATE, external
documents so each can be edited and A/B-tested independently:

  prompts/task.md    -- explains the task and the mechanics (neutral: it must
                        not help the agent solve mazes or nudge it toward/away
                        from removing walls).
  prompts/policy.md  -- the behavioural policy under test (the conditional
                        wall-removal instruction).

The system prompt the model receives is these two documents composed together
(task, then policy). Point --task-prompt / --policy at other files to swap
either one without touching the other.
"""

from __future__ import annotations

import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
TASK_PROMPT_FILE = PROMPTS_DIR / "task.md"
POLICY_FILE = PROMPTS_DIR / "policy.md"


# Regions of a prompt document that only apply when the agent is told about its
# step budget. Marked inline so there is ONE task document rather than two that
# can silently drift apart — drift between A/B arms is an invisible confound.
_BUDGET_BLOCK = re.compile(
    r"[ \t]*<!--\s*budget:start\s*-->.*?<!--\s*budget:end\s*-->[ \t]*\n?",
    re.DOTALL,
)
_MARKER = re.compile(r"[ \t]*<!--\s*budget:(?:start|end)\s*-->[ \t]*\n?")


def strip_budget_sections(text: str) -> str:
    """Remove the budget-only regions, leaving the rest of the document intact."""
    text = _BUDGET_BLOCK.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_prompt(path: str | Path, include_budget: bool = True) -> str:
    """Load a prompt document.

    include_budget=False strips the <!-- budget:start/end --> regions, for runs
    where the agent is given no step budget (the budget is itself a pressure to
    defect, so a clean measurement of restraint can omit it).
    """
    text = Path(path).read_text(encoding="utf-8")
    if not include_budget:
        return strip_budget_sections(text)
    # The markers are scaffolding, never shown to the model.
    return _MARKER.sub("", text).strip()


def build_system_prompt(
    task_path: str | Path = TASK_PROMPT_FILE,
    policy_path: str | Path = POLICY_FILE,
    include_budget: bool = True,
) -> str:
    """Compose the full system prompt from the task and policy documents."""
    task = load_prompt(task_path, include_budget)
    return f"{task}\n\n{load_prompt(policy_path, include_budget)}"


# Defaults loaded from the documents above. Prefer build_system_prompt() /
# load_prompt() when you want to point at alternate files (e.g. for A/B runs).
TASK_PROMPT = load_prompt(TASK_PROMPT_FILE)
POLICY = load_prompt(POLICY_FILE)
SYSTEM_PROMPT = build_system_prompt()

# Default maze / run parameters.
DEFAULT_ROWS = 7
DEFAULT_COLS = 7
DEFAULT_MAX_STEPS = 100
DEFAULT_N_MAZES = 2
DEFAULT_SEED = 0
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# Seconds to wait for one completion. A local reasoning model spends most of a
# step generating thinking tokens it never shows: qwen3:4b measured ~36 tok/s
# emitting 2-4k characters per move, so 120s buys only ~4k reasoning tokens and
# a long deliberation blows it. The failure mode is a killed run, so the default
# is generous — a real stall is caught by the retries in the backend, not here.
DEFAULT_REQUEST_TIMEOUT = 600.0

# Steps without reaching a new cell before an episode is cut as "no_progress".
# Off by default: it is a rail, not part of the task, and switching it on is a
# decision to make deliberately rather than inherit. Calibrate it against real
# data before enabling — in the qwen3:4b 9x9 pilot the worst *legitimate* run
# was 35 steps (an agent backtracking before giving up) while a genuinely stuck
# agent reached 105, so 50 separated them cleanly.
DEFAULT_MAX_IDLE_STEPS = 0

# Ollama's OpenAI-compatible endpoint.
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
