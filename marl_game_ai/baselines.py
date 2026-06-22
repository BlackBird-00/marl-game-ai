from __future__ import annotations

from typing import Dict

import numpy as np

from .envs import CoopPuzzleEnv


class RandomPolicy:
    def __init__(self, action_dim: int, seed: int | None = None) -> None:
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def act(self, env: CoopPuzzleEnv, observations=None) -> Dict[str, int]:
        return {agent: int(self.rng.integers(0, self.action_dim)) for agent in env.agents}


class RuleBasedPolicy:
    """Hand-written baseline for the default puzzle mechanics."""

    def act(self, env: CoopPuzzleEnv, observations=None) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        plate_targets = env.pressure_plates
        key_targets = [env.key_pos] if env.key_pos and not env.key_taken else env.goals
        for agent, pos in env.agent_positions.items():
            if agent == "agent_0":
                actions[agent] = env.shortest_action_towards(pos, plate_targets)
            else:
                actions[agent] = env.shortest_action_towards(pos, key_targets)
        return actions

