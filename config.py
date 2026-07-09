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

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
TASK_PROMPT_FILE = PROMPTS_DIR / "task.md"
POLICY_FILE = PROMPTS_DIR / "policy.md"


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def build_system_prompt(
    task_path: str | Path = TASK_PROMPT_FILE,
    policy_path: str | Path = POLICY_FILE,
) -> str:
    """Compose the full system prompt from the task and policy documents."""
    return f"{load_prompt(task_path)}\n\n{load_prompt(policy_path)}"


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

# Ollama's OpenAI-compatible endpoint.
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
