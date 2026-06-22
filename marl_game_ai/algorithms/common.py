from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, action_dim, hidden_dim)
        self.critic = mlp(obs_dim, 1, hidden_dim)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class MAPPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = mlp(obs_dim, action_dim, hidden_dim)
        self.critic = mlp(state_dim, 1, hidden_dim)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(obs))

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    device: str = "cpu"


def to_tensor(array, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def random_actions(agents: List[str], action_dim: int, rng: np.random.Generator) -> Dict[str, int]:
    return {agent: int(rng.integers(0, action_dim)) for agent in agents}

