from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mukti_env import MuktiProductionEnv, save_gif
from train import METRICS_PATH, QTABLE_PATH, deserialize_q_table, train


ROOT = Path(__file__).resolve().parent.parent
GIF_PATH = ROOT / "artifacts" / "simulation.gif"
PNG_PATH = ROOT / "artifacts" / "simulation_last_frame.png"


def ensure_trained() -> dict:
    if not QTABLE_PATH.exists():
        train()
    return np.load(QTABLE_PATH, allow_pickle=True).item()


def rollout_episode(q_table: dict, seed: int, max_steps: int) -> dict:
    env = MuktiProductionEnv(seed=seed)
    state = env.reset()
    frames = [env.render_frame()]
    total_reward = 0.0
    info = {"delivered": 0}

    for _ in range(max_steps):
        action = int(np.argmax(q_table.get(state, np.zeros(len(env.actions)))))
        state, reward, done, info = env.step(action)
        total_reward += reward
        frames.append(env.render_frame())
        if done:
            break
    return {"frames": frames, "reward": total_reward, "deliveries": info["delivered"]}


def render_episode(max_steps: int = 100) -> dict:
    q_table_data = ensure_trained()
    q_table = deserialize_q_table(q_table_data)
    candidates = [rollout_episode(q_table, seed=seed, max_steps=max_steps) for seed in range(100, 110)]
    best = max(candidates, key=lambda item: (item["deliveries"], item["reward"]))
    frames = best["frames"]

    save_gif(frames, GIF_PATH)
    frames[-1].save(PNG_PATH)
    summary = {
        "frames": len(frames),
        "reward": best["reward"],
        "deliveries": best["deliveries"],
        "gif_path": str(GIF_PATH),
        "png_path": str(PNG_PATH),
    }
    (ROOT / "artifacts" / "simulation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(render_episode(), indent=2))
