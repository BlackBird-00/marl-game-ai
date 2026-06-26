from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


ACTIONS = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}


DEFAULT_LEVELS = {
    "basic": [
        "########",
        "#A.P#..#",
        "#...D.G#",
        "#B.K#..#",
        "#...#..#",
        "#..T#..#",
        "#......#",
        "########",
    ],
    "key_door": [
        "##########",
        "#A..P#...#",
        "#....D.G.#",
        "#B..K#...#",
        "#....#...#",
        "#..T.#...#",
        "#........#",
        "##########",
    ],
    "trap": [
        "##########",
        "#A.P.#...#",
        "#..T.D.G.#",
        "#B.K.#...#",
        "#..T.#...#",
        "#....#...#",
        "#........#",
        "##########",
    ],
}


Position = Tuple[int, int]


@dataclass
class StepStats:
    collision: bool = False
    invalid: int = 0
    opened_now: bool = False
    key_taken_now: bool = False
    success: bool = False


class CoopPuzzleEnv:
    """A small ParallelEnv-style cooperative puzzle game.

    Two agents share one team reward. One agent must hold a pressure plate to
    open a door while the team collects a key and reaches the goal.
    """

    metadata = {"name": "coop_puzzle_v0"}

    def __init__(
        self,
        level: str = "basic",
        max_steps: int = 120,
        seed: Optional[int] = None,
        shared_reward: bool = True,
    ) -> None:
        if level not in DEFAULT_LEVELS:
            raise ValueError(f"Unknown level '{level}'. Choices: {sorted(DEFAULT_LEVELS)}")
        self.level_name = level
        self.level_lines = DEFAULT_LEVELS[level]
        self.max_steps = max_steps
        self.shared_reward = shared_reward
        self.rng = np.random.default_rng(seed)
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self.n_agents = len(self.agents)
        self.action_dim = len(ACTIONS)

        self.height = len(self.level_lines)
        self.width = max(len(row) for row in self.level_lines)
        self.walls: set[Position] = set()
        self.traps: set[Position] = set()
        self.pressure_plates: set[Position] = set()
        self.doors: set[Position] = set()
        self.goals: set[Position] = set()
        self.key_pos: Optional[Position] = None
        self.start_positions: Dict[str, Position] = {}
        self._parse_level()

        self.agent_positions: Dict[str, Position] = {}
        self.key_taken = False
        self.door_open = False
        self.door_ever_opened = False
        self.step_count = 0
        self.last_stats = StepStats()
        self.trajectory: List[dict] = []

    @property
    def obs_dim(self) -> int:
        return len(self.observe(self.possible_agents[0]))

    @property
    def state_dim(self) -> int:
        return len(self.state())

    def _parse_level(self) -> None:
        for r, row in enumerate(self.level_lines):
            for c, cell in enumerate(row):
                pos = (r, c)
                if cell == "#":
                    self.walls.add(pos)
                elif cell == "T":
                    self.traps.add(pos)
                elif cell == "P":
                    self.pressure_plates.add(pos)
                elif cell == "D":
                    self.doors.add(pos)
                elif cell == "G":
                    self.goals.add(pos)
                elif cell == "K":
                    self.key_pos = pos
                elif cell == "A":
                    self.start_positions["agent_0"] = pos
                elif cell == "B":
                    self.start_positions["agent_1"] = pos
        missing = [a for a in self.possible_agents if a not in self.start_positions]
        if missing or not self.pressure_plates or not self.doors or not self.goals:
            raise ValueError(f"Level '{self.level_name}' is missing required symbols.")

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.agent_positions = dict(self.start_positions)
        self.key_taken = False
        self.door_open = self._compute_door_open()
        self.door_ever_opened = self.door_open
        self.step_count = 0
        self.last_stats = StepStats()
        self.trajectory = []
        self._record_frame(reward=0.0, actions={a: 0 for a in self.agents})
        return self._observations(), {a: {} for a in self.agents}

    def step(self, actions: Dict[str, int]):
        self.step_count += 1
        old_positions = dict(self.agent_positions)
        old_door_open = self.door_open
        old_progress = self._team_progress_score()
        proposed: Dict[str, Position] = {}
        stats = StepStats()

        for agent in self.agents:
            action = int(actions.get(agent, 0))
            dr, dc = ACTIONS.get(action, (0, 0))
            nr = self.agent_positions[agent][0] + dr
            nc = self.agent_positions[agent][1] + dc
            target = (nr, nc)
            if not self._is_passable(target):
                proposed[agent] = self.agent_positions[agent]
                stats.invalid += 1
            else:
                proposed[agent] = target

        counts: Dict[Position, int] = {}
        for pos in proposed.values():
            counts[pos] = counts.get(pos, 0) + 1
        for agent, pos in proposed.items():
            if counts[pos] > 1 and pos not in self.goals:
                proposed[agent] = old_positions[agent]
                stats.collision = True

        self.agent_positions = proposed
        self.door_open = self._compute_door_open()
        stats.opened_now = (not old_door_open) and self.door_open and not self.door_ever_opened
        self.door_ever_opened = self.door_ever_opened or self.door_open

        if self.key_pos is not None and not self.key_taken:
            if self.agent_positions["agent_1"] == self.key_pos:
                self.key_taken = True
                stats.key_taken_now = True

        stats.success = self._is_success()
        reward = self._reward(stats, old_progress)
        terminations = {a: stats.success for a in self.agents}
        truncations = {a: self.step_count >= self.max_steps and not stats.success for a in self.agents}
        infos = {
            a: {
                "success": stats.success,
                "collision": stats.collision,
                "invalid": stats.invalid,
                "door_open": self.door_open,
                "key_taken": self.key_taken,
            }
            for a in self.agents
        }
        rewards = {a: reward for a in self.agents}
        self.last_stats = stats
        self._record_frame(reward=reward, actions=actions)
        return self._observations(), rewards, terminations, truncations, infos

    def observe(self, agent: str) -> np.ndarray:
        ar, ac = self.agent_positions.get(agent, self.start_positions[agent])
        other = [a for a in self.possible_agents if a != agent][0]
        agent_index = self.possible_agents.index(agent)
        orow, ocol = self.agent_positions.get(other, self.start_positions[other])
        plate = self._nearest(self.pressure_plates, (ar, ac))
        door = self._nearest(self.doors, (ar, ac))
        goal = self._nearest(self.goals, (ar, ac))
        key = self.key_pos if self.key_pos is not None else goal
        features = [
            ar / max(1, self.height - 1),
            ac / max(1, self.width - 1),
            orow / max(1, self.height - 1),
            ocol / max(1, self.width - 1),
            plate[0] / max(1, self.height - 1),
            plate[1] / max(1, self.width - 1),
            door[0] / max(1, self.height - 1),
            door[1] / max(1, self.width - 1),
            key[0] / max(1, self.height - 1),
            key[1] / max(1, self.width - 1),
            goal[0] / max(1, self.height - 1),
            goal[1] / max(1, self.width - 1),
            float(self.door_open),
            float(self.key_taken),
            self.step_count / max(1, self.max_steps),
            float(agent_index == 0),
            float(agent_index == 1),
        ]
        return np.asarray(features, dtype=np.float32)

    def state(self) -> np.ndarray:
        return np.concatenate([self.observe(agent) for agent in self.possible_agents]).astype(np.float32)

    def render_ascii(self) -> str:
        grid = [list(row) for row in self.level_lines]
        for r, c in self.doors:
            grid[r][c] = "d" if self.door_open else "D"
        if self.key_pos and self.key_taken:
            r, c = self.key_pos
            grid[r][c] = "."
        for agent, (r, c) in self.agent_positions.items():
            grid[r][c] = "A" if agent == "agent_0" else "B"
        return "\n".join("".join(row) for row in grid)

    def export_trajectory(self) -> List[dict]:
        return list(self.trajectory)

    def shortest_action_towards(self, start: Position, targets: Iterable[Position]) -> int:
        targets = set(targets)
        if start in targets:
            return 0
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, first_action = queue.popleft()
            for action, delta in ACTIONS.items():
                if action == 0:
                    continue
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if nxt in seen or not self._is_passable(nxt):
                    continue
                next_first = action if first_action == 0 else first_action
                if nxt in targets:
                    return next_first
                seen.add(nxt)
                queue.append((nxt, next_first))
        return 0

    def _observations(self) -> Dict[str, np.ndarray]:
        return {a: self.observe(a) for a in self.agents}

    def _is_passable(self, pos: Position) -> bool:
        r, c = pos
        if r < 0 or c < 0 or r >= self.height or c >= len(self.level_lines[r]):
            return False
        if pos in self.walls:
            return False
        if pos in self.doors and not self.door_open:
            return False
        return True

    def _compute_door_open(self) -> bool:
        return self.agent_positions.get("agent_0") in self.pressure_plates

    def _is_success(self) -> bool:
        return self.key_taken and self.agent_positions.get("agent_1") in self.goals

    def _reward(self, stats: StepStats, old_progress: float) -> float:
        reward = -0.10
        new_progress = self._team_progress_score()
        reward += 0.50 * (new_progress - old_progress)
        if stats.opened_now:
            reward += 1.0
        if stats.key_taken_now:
            reward += 2.0
        if any(pos in self.traps for pos in self.agent_positions.values()):
            reward -= 1.0
        reward -= 0.05 * stats.invalid
        if stats.collision:
            reward -= 0.2
        if stats.success:
            speed_bonus = 8.0 * (1.0 - self.step_count / max(1, self.max_steps))
            reward += 8.0 + max(0.0, speed_bonus)
        return float(reward)

    def _team_progress_score(self) -> float:
        agent0 = self.agent_positions.get("agent_0", self.start_positions["agent_0"])
        agent1 = self.agent_positions.get("agent_1", self.start_positions["agent_1"])
        plate_dist = self._shortest_distance(agent0, self.pressure_plates, doors_passable=True)
        key_target = [self.key_pos] if self.key_pos and not self.key_taken else self.goals
        agent1_dist = self._shortest_distance(agent1, key_target, doors_passable=True)
        return -float(plate_dist + agent1_dist)

    def _record_frame(self, reward: float, actions: Dict[str, int]) -> None:
        self.trajectory.append(
            {
                "step": self.step_count,
                "agents": {k: list(v) for k, v in self.agent_positions.items()},
                "actions": {k: int(v) for k, v in actions.items()},
                "door_open": self.door_open,
                "door_ever_opened": self.door_ever_opened,
                "key_taken": self.key_taken,
                "reward": float(reward),
                "success": self._is_success(),
                "level": self.level_lines,
                "pressure_plates": [list(p) for p in sorted(self.pressure_plates)],
                "doors": [list(p) for p in sorted(self.doors)],
                "goals": [list(p) for p in sorted(self.goals)],
                "key": list(self.key_pos) if self.key_pos else None,
                "traps": [list(p) for p in sorted(self.traps)],
            }
        )

    @staticmethod
    def _nearest(points: Iterable[Position], origin: Position) -> Position:
        pts = list(points)
        return min(pts, key=lambda p: abs(p[0] - origin[0]) + abs(p[1] - origin[1]))

    @staticmethod
    def _manhattan_to_any(pos: Position, points: Iterable[Position]) -> int:
        pts = list(points)
        if not pts:
            return 0
        return min(abs(pos[0] - p[0]) + abs(pos[1] - p[1]) for p in pts)

    def _shortest_distance(
        self,
        start: Position,
        targets: Iterable[Position],
        doors_passable: bool = False,
    ) -> int:
        target_set = set(targets)
        if not target_set or start in target_set:
            return 0
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, distance = queue.popleft()
            for action, (dr, dc) in ACTIONS.items():
                if action == 0:
                    continue
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt in seen:
                    continue
                r, c = nxt
                if r < 0 or c < 0 or r >= self.height or c >= len(self.level_lines[r]):
                    continue
                if nxt in self.walls:
                    continue
                if nxt in self.doors and not (doors_passable or self.door_open):
                    continue
                if nxt in target_set:
                    return distance + 1
                seen.add(nxt)
                queue.append((nxt, distance + 1))
        return self.height * self.width
