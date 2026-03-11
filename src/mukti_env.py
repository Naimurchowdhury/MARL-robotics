from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import random

from PIL import Image, ImageDraw


Position = Tuple[int, int]


@dataclass(frozen=True)
class ProductTask:
    pickup: Position
    dropoff: Position
    label: str


class MuktiProductionEnv:
    width = 8
    height = 6
    actions = {
        0: (0, -1),   # up
        1: (0, 1),    # down
        2: (-1, 0),   # left
        3: (1, 0),    # right
        4: (0, 0),    # interact
    }

    def __init__(self, seed: int = 7) -> None:
        self.random = random.Random(seed)
        self.obstacles = {
            (3, 1), (3, 2), (3, 3),
            (5, 2), (5, 3), (1, 4),
        }
        self.chargers = {(0, 5)}
        self.tasks = [
            ProductTask((0, 0), (7, 0), "A"),
            ProductTask((0, 2), (7, 2), "B"),
            ProductTask((0, 4), (7, 5), "C"),
        ]
        self.reset()

    def reset(self) -> Tuple[int, int, int, int, int]:
        self.agent = (1, 5)
        self.battery = 40
        self.carrying = -1
        self.task_index = self.random.randrange(len(self.tasks))
        self.delivered = 0
        self.steps = 0
        return self.state

    @property
    def state(self) -> Tuple[int, int, int, int, int]:
        return (
            self.agent[0],
            self.agent[1],
            self.carrying,
            self.task_index,
            min(self.battery // 6, 6),
        )

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int, int], float, bool, dict]:
        self.steps += 1
        reward = -0.08
        done = False

        if action != 4:
            dx, dy = self.actions[action]
            nx = self.agent[0] + dx
            ny = self.agent[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.obstacles:
                self.agent = (nx, ny)
            else:
                reward -= 0.9
            self.battery -= 1
        else:
            reward += self._interact()

        if self.agent in self.chargers and action == 4:
            self.battery = min(40, self.battery + 10)
            reward += 0.5

        if self.battery <= 0:
            reward -= 5.0
            done = True

        if self.delivered >= 4:
            reward += 10.0
            done = True

        if self.steps >= 180:
            done = True

        return self.state, reward, done, {"delivered": self.delivered}

    def _interact(self) -> float:
        task = self.tasks[self.task_index]
        if self.carrying == -1 and self.agent == task.pickup:
            self.carrying = self.task_index
            return 7.0
        if self.carrying == self.task_index and self.agent == task.dropoff:
            self.carrying = -1
            self.delivered += 1
            self.task_index = (self.task_index + 1) % len(self.tasks)
            return 15.0
        return -0.6

    def sample_action(self) -> int:
        return self.random.randrange(len(self.actions))

    def render_frame(self, scale: int = 80) -> Image.Image:
        image = Image.new("RGB", (self.width * scale, self.height * scale), "#f3efe6")
        draw = ImageDraw.Draw(image)

        colors = {
            "floor": "#f3efe6",
            "grid": "#d7cdbd",
            "obstacle": "#565449",
            "charger": "#7fb069",
            "pickup": "#2a9d8f",
            "dropoff": "#e76f51",
            "agent": "#264653",
            "carry": "#e9c46a",
            "task": "#1d3557",
        }

        for x in range(self.width):
            for y in range(self.height):
                cell = (x * scale, y * scale, (x + 1) * scale, (y + 1) * scale)
                draw.rectangle(cell, outline=colors["grid"], width=2)
                pos = (x, y)
                if pos in self.obstacles:
                    draw.rounded_rectangle(cell, radius=10, fill=colors["obstacle"])
                elif pos in self.chargers:
                    draw.rounded_rectangle(cell, radius=10, fill=colors["charger"])

        for idx, task in enumerate(self.tasks):
            pcell = self._cell_box(task.pickup, scale)
            dcell = self._cell_box(task.dropoff, scale)
            draw.ellipse(self._inset(pcell, 16), fill=colors["pickup"])
            draw.rectangle(self._inset(dcell, 16), fill=colors["dropoff"])
            draw.text((pcell[0] + 8, pcell[1] + 6), task.label, fill="white")
            draw.text((dcell[0] + 8, dcell[1] + 6), task.label, fill="white")
            if idx == self.task_index:
                draw.text((dcell[0] + 42, dcell[1] + 8), "target", fill=colors["task"])

        ax0, ay0, ax1, ay1 = self._cell_box(self.agent, scale)
        draw.rounded_rectangle(self._inset((ax0, ay0, ax1, ay1), 12), radius=20, fill=colors["agent"])
        if self.carrying != -1:
            draw.rectangle((ax0 + 28, ay0 + 20, ax0 + 52, ay0 + 44), fill=colors["carry"])

        hud_y = self.height * scale - 28
        draw.rectangle((0, hud_y, self.width * scale, self.height * scale), fill="#ffffff")
        draw.text((12, hud_y + 4), f"deliveries={self.delivered}  battery={self.battery}  carrying={self.carrying}", fill="#111111")
        return image

    @staticmethod
    def _cell_box(pos: Position, scale: int) -> Tuple[int, int, int, int]:
        x, y = pos
        return (x * scale, y * scale, (x + 1) * scale, (y + 1) * scale)

    @staticmethod
    def _inset(box: Tuple[int, int, int, int], amount: int) -> Tuple[int, int, int, int]:
        return (box[0] + amount, box[1] + amount, box[2] - amount, box[3] - amount)


def save_gif(frames: List[Image.Image], path: Path, duration_ms: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
