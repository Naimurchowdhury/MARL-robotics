from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np

from mukti_env import MuktiProductionEnv


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "training_config.json"
ARTIFACTS_DIR = ROOT / "artifacts"
QTABLE_PATH = ARTIFACTS_DIR / "q_table.npy"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, float]:
    return json.loads(path.read_text())


def train(config: Dict[str, float] | None = None, save: bool = True) -> Dict[str, object]:
    cfg = config or load_config()
    env = MuktiProductionEnv(seed=int(cfg["seed"]))
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    q_table: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
    rewards: List[float] = []
    epsilon = float(cfg["epsilon_start"])

    for _ in range(int(cfg["episodes"])):
        state = env.reset()
        total_reward = 0.0
        for _step in range(int(cfg["max_steps_per_episode"])):
            if random.random() < epsilon:
                action = env.sample_action()
            else:
                action = int(np.argmax(q_table.setdefault(state, np.zeros(len(env.actions)))))

            next_state, reward, done, _info = env.step(action)
            q_table.setdefault(state, np.zeros(len(env.actions)))
            q_table.setdefault(next_state, np.zeros(len(env.actions)))

            old_value = q_table[state][action]
            next_best = float(np.max(q_table[next_state]))
            q_table[state][action] = old_value + float(cfg["learning_rate"]) * (
                reward + float(cfg["discount_factor"]) * next_best - old_value
            )
            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        epsilon = max(float(cfg["epsilon_end"]), epsilon * float(cfg["epsilon_decay"]))

    evaluation = evaluate(q_table, int(cfg["eval_episodes"]), int(cfg["seed"]))
    result = {
        "mean_train_reward_last_200": float(np.mean(rewards[-200:])),
        "best_train_reward": float(np.max(rewards)),
        "eval_mean_reward": evaluation["mean_reward"],
        "eval_mean_deliveries": evaluation["mean_deliveries"],
        "episodes": int(cfg["episodes"]),
        "q_table": serialize_q_table(q_table),
        "training_rewards": rewards,
    }

    if save:
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        np.save(QTABLE_PATH, result["q_table"], allow_pickle=True)
        METRICS_PATH.write_text(json.dumps({k: v for k, v in result.items() if k != "q_table"}, indent=2))
    return result


def evaluate(q_table_data: Dict[str, List[float]] | Dict[Tuple[int, int, int, int, int], np.ndarray], episodes: int, seed: int) -> Dict[str, float]:
    env = MuktiProductionEnv(seed=seed + 99)
    q_table = deserialize_q_table(q_table_data) if isinstance(next(iter(q_table_data.keys())), str) else q_table_data
    rewards = []
    deliveries = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for _ in range(180):
            action = int(np.argmax(q_table.get(state, np.zeros(len(env.actions)))))
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                deliveries.append(info["delivered"])
                break
        rewards.append(total_reward)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_deliveries": float(np.mean(deliveries) if deliveries else 0.0),
    }


def serialize_q_table(q_table: Dict[Tuple[int, int, int, int, int], np.ndarray]) -> Dict[str, List[float]]:
    return {"|".join(map(str, state)): values.tolist() for state, values in q_table.items()}


def deserialize_q_table(q_table_data: Dict[str, List[float]]) -> Dict[Tuple[int, int, int, int, int], np.ndarray]:
    return {
        tuple(int(part) for part in state.split("|")): np.array(values, dtype=float)
        for state, values in q_table_data.items()
    }


if __name__ == "__main__":
    result = train()
    print(json.dumps({k: v for k, v in result.items() if k != "q_table" and k != "training_rewards"}, indent=2))
