from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from .common import MAPPOActorCritic, PPOConfig, compute_gae, to_tensor


class MAPPOTrainer:
    """Shared actor with a centralized critic over the global state."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        agents: list[str],
        config: PPOConfig | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.agents = agents
        self.action_dim = action_dim
        self.model = MAPPOActorCritic(obs_dim, state_dim, action_dim, self.config.hidden_dim).to(self.config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def act(self, observations: Dict[str, np.ndarray], state: np.ndarray, deterministic: bool = False):
        actions, log_probs = {}, {}
        with torch.no_grad():
            state_t = to_tensor(state, self.config.device).unsqueeze(0)
            value = float(self.model.value(state_t).item())
            for agent, obs in observations.items():
                obs_t = to_tensor(obs, self.config.device).unsqueeze(0)
                dist = self.model.dist(obs_t)
                action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
                actions[agent] = int(action.item())
                log_probs[agent] = float(dist.log_prob(action).item())
        return actions, log_probs, value

    def update(self, data: list[dict]) -> Dict[str, float]:
        obs = np.asarray([x["obs"] for x in data], dtype=np.float32)
        states = np.asarray([x["state"] for x in data], dtype=np.float32)
        actions = np.asarray([x["action"] for x in data], dtype=np.int64)
        old_log_probs = np.asarray([x["log_prob"] for x in data], dtype=np.float32)
        rewards = np.asarray([x["reward"] for x in data], dtype=np.float32)
        dones = np.asarray([x["done"] for x in data], dtype=np.float32)
        values = np.asarray([x["value"] for x in data], dtype=np.float32)
        advantages, returns = compute_gae(rewards, dones, values, self.config.gamma, self.config.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = to_tensor(obs, self.config.device)
        states_t = to_tensor(states, self.config.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.config.device)
        old_log_probs_t = to_tensor(old_log_probs, self.config.device)
        advantages_t = to_tensor(advantages, self.config.device)
        returns_t = to_tensor(returns, self.config.device)
        n = len(actions)
        idxs = np.arange(n)
        policy_loss_value = value_loss_value = entropy_value = 0.0

        for _ in range(self.config.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.config.minibatch_size):
                batch = idxs[start : start + self.config.minibatch_size]
                dist = self.model.dist(obs_t[batch])
                new_log_probs = dist.log_prob(actions_t[batch])
                entropy = dist.entropy().mean()
                new_values = self.model.value(states_t[batch])
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch])
                pg1 = ratio * advantages_t[batch]
                pg2 = torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef) * advantages_t[batch]
                policy_loss = -torch.min(pg1, pg2).mean()
                value_loss = nn.functional.mse_loss(new_values, returns_t[batch])
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())
                entropy_value = float(entropy.item())

        return {
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
        }

    def save(self, path: str) -> None:
        torch.save({"type": "mappo", "agents": self.agents, "model": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt["model"])

