from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess

from train import CONFIG_PATH, train


ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "artifacts" / "baseline_metrics.json"


def mutate_config(config: dict) -> dict:
    mutated = dict(config)
    candidates = {
        "learning_rate": (0.12, 0.35),
        "discount_factor": (0.90, 0.99),
        "epsilon_decay": (0.995, 0.9995),
        "episodes": (2200, 4200),
    }
    key = random.choice(list(candidates.keys()))
    low, high = candidates[key]
    value = random.uniform(low, high)
    mutated[key] = round(value, 4) if key != "episodes" else int(value)
    return mutated


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)


def repo_ready() -> bool:
    return git("rev-parse", "--is-inside-work-tree").returncode == 0


def commit_improvement(config: dict, metrics: dict) -> str:
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    git("add", str(CONFIG_PATH), "artifacts/metrics.json", "artifacts/q_table.npy", "artifacts/baseline_metrics.json")
    message = f"Improve training eval reward to {metrics['eval_mean_reward']:.2f}"
    git("commit", "-m", message)
    return message


def main() -> dict:
    if not repo_ready():
        raise SystemExit("git repository not initialized")

    config = json.loads(CONFIG_PATH.read_text())
    baseline = train(config=config, save=True)
    BASELINE_PATH.write_text(json.dumps({k: v for k, v in baseline.items() if k not in {"q_table", "training_rewards"}}, indent=2))

    candidate = mutate_config(config)
    candidate_metrics = train(config=candidate, save=True)

    improved = candidate_metrics["eval_mean_reward"] > baseline["eval_mean_reward"]
    if improved:
        commit_message = commit_improvement(candidate, candidate_metrics)
    else:
        train(config=config, save=True)
        commit_message = "discarded candidate config"

    summary = {
        "baseline_eval_mean_reward": baseline["eval_mean_reward"],
        "candidate_eval_mean_reward": candidate_metrics["eval_mean_reward"],
        "improved": improved,
        "commit_message": commit_message,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
