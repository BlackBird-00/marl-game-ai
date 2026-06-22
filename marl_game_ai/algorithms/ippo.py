from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import nn

from .common import ActorCritic, PPOConfig, compute_gae, to_tensor


class IPPOTrainer:
    """Independent PPO: one actor-critic per agent with local observations."""

    def __init__(self, obs_dim: int, action_dim: int, agents: list[str], config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()
        self.agents = agents
        self.action_dim = action_dim
        self.models = {
            agent: ActorCritic(obs_dim, action_dim, self.config.hidden_dim).to(self.config.device)
            for agent in agents
        }
        self.optimizers = {
            agent: torch.optim.Adam(self.models[agent].parameters(), lr=self.config.lr)
            for agent in agents
        }

    def act(self, observations: Dict[str, np.ndarray], deterministic: bool = False):
        actions, log_probs, values = {}, {}, {}
        with torch.no_grad():
            for agent, obs in observations.items():
                obs_t = to_tensor(obs, self.config.device).unsqueeze(0)
                dist = self.models[agent].dist(obs_t)
                action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
                actions[agent] = int(action.item())
                log_probs[agent] = float(dist.log_prob(action).item())
                values[agent] = float(self.models[agent].value(obs_t).item())
        return actions, log_probs, values

    def update(self, rollouts: Dict[str, list[dict]]) -> Dict[str, float]:
        metrics = {}
        for agent in self.agents:
            data = rollouts[agent]
            obs = np.asarray([x["obs"] for x in data], dtype=np.float32)
            actions = np.asarray([x["action"] for x in data], dtype=np.int64)
            old_log_probs = np.asarray([x["log_prob"] for x in data], dtype=np.float32)
            rewards = np.asarray([x["reward"] for x in data], dtype=np.float32)
            dones = np.asarray([x["done"] for x in data], dtype=np.float32)
            values = np.asarray([x["value"] for x in data], dtype=np.float32)
            advantages, returns = compute_gae(rewards, dones, values, self.config.gamma, self.config.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_t = to_tensor(obs, self.config.device)
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
                    dist = self.models[agent].dist(obs_t[batch])
                    new_log_probs = dist.log_prob(actions_t[batch])
                    entropy = dist.entropy().mean()
                    new_values = self.models[agent].value(obs_t[batch])
                    ratio = torch.exp(new_log_probs - old_log_probs_t[batch])
                    pg1 = ratio * advantages_t[batch]
                    pg2 = torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef) * advantages_t[batch]
                    policy_loss = -torch.min(pg1, pg2).mean()
                    value_loss = nn.functional.mse_loss(new_values, returns_t[batch])
                    loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                    self.optimizers[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.models[agent].parameters(), self.config.max_grad_norm)
                    self.optimizers[agent].step()
                    policy_loss_value = float(policy_loss.item())
                    value_loss_value = float(value_loss.item())
                    entropy_value = float(entropy.item())

            metrics[f"{agent}/policy_loss"] = policy_loss_value
            metrics[f"{agent}/value_loss"] = value_loss_value
            metrics[f"{agent}/entropy"] = entropy_value
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "type": "ippo",
                "agents": self.agents,
                "models": {agent: model.state_dict() for agent, model in self.models.items()},
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.config.device)
        for agent, state in ckpt["models"].items():
            self.models[agent].load_state_dict(state)

