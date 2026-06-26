from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from marl_game_ai.algorithms.common import PPOConfig
from marl_game_ai.algorithms.ippo import IPPOTrainer
from marl_game_ai.algorithms.mappo import MAPPOTrainer
from marl_game_ai.envs import CoopPuzzleEnv
from marl_game_ai.utils import ensure_dir, load_config, plot_training, save_json, save_metrics_csv


def build_ppo_config(config: dict) -> PPOConfig:
    return PPOConfig(
        gamma=float(config.get("gamma", 0.99)),
        gae_lambda=float(config.get("gae_lambda", 0.95)),
        clip_coef=float(config.get("clip_coef", 0.2)),
        value_coef=float(config.get("value_coef", 0.5)),
        entropy_coef=float(config.get("entropy_coef", 0.01)),
        lr=float(config.get("lr", 3e-4)),
        update_epochs=int(config.get("update_epochs", 4)),
        minibatch_size=int(config.get("minibatch_size", 256)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        device=str(config.get("device", "cpu")),
    )


def train_ippo(args, env: CoopPuzzleEnv, out_dir: Path, config: PPOConfig):
    trainer = IPPOTrainer(env.obs_dim, env.action_dim, env.possible_agents, config)
    metrics = []
    rng = np.random.default_rng(args.seed)
    pending_rollouts = {agent: [] for agent in env.possible_agents}
    last_update_metrics = {}
    best_score = float("-inf")

    def evaluation_score() -> tuple[float, float]:
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(args.seed + 20_000)
        successes = []
        returns = []
        try:
            for eval_episode in range(args.selection_episodes):
                eval_env = CoopPuzzleEnv(level=env.level_name, max_steps=env.max_steps, seed=args.seed)
                eval_obs, _ = eval_env.reset(seed=args.seed + eval_episode)
                eval_return = 0.0
                success = False
                for _ in range(eval_env.max_steps):
                    eval_actions, _, _ = trainer.act(eval_obs, deterministic=False)
                    eval_obs, eval_rewards, terminations, truncations, infos = eval_env.step(eval_actions)
                    eval_return += float(np.mean(list(eval_rewards.values())))
                    success = any(info["success"] for info in infos.values())
                    if any(terminations.values()) or any(truncations.values()):
                        break
                successes.append(float(success))
                returns.append(eval_return)
        finally:
            torch.random.set_rng_state(torch_rng_state)
        success_rate = float(np.mean(successes))
        score = 1000.0 * success_rate + float(np.mean(returns))
        return score, success_rate

    for episode in range(1, args.episodes + 1):
        observations, _ = env.reset(seed=int(rng.integers(1_000_000)))
        total_return = 0.0
        success = False
        collisions = 0
        for _ in range(args.max_steps):
            actions, log_probs, values = trainer.act(observations)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = any(terminations.values()) or any(truncations.values())
            reward = float(np.mean(list(rewards.values())))
            total_return += reward
            success = success or any(info["success"] for info in infos.values())
            collisions += int(any(info["collision"] for info in infos.values()))
            for agent in env.possible_agents:
                pending_rollouts[agent].append(
                    {
                        "obs": observations[agent],
                        "action": actions[agent],
                        "log_prob": log_probs[agent],
                        "value": values[agent],
                        "reward": reward,
                        "done": float(done),
                    }
                )
            observations = next_obs
            if done:
                break

        should_update = episode % args.rollout_episodes == 0 or episode == args.episodes
        evaluation_success_rate = 0.0
        if should_update:
            last_update_metrics = trainer.update(pending_rollouts)
            pending_rollouts = {agent: [] for agent in env.possible_agents}
            score, evaluation_success_rate = evaluation_score()
            if score > best_score:
                best_score = score
                trainer.save(str(out_dir / "ippo_best.pt"))

        row = {
            "episode": episode,
            "return": total_return,
            "success": int(success),
            "steps": env.step_count,
            "collisions": collisions,
            "evaluation_success_rate": evaluation_success_rate,
        }
        row.update(last_update_metrics)
        metrics.append(row)
        if episode % args.log_interval == 0:
            recent = metrics[-args.log_interval :]
            sr = sum(r["success"] for r in recent) / len(recent)
            print(f"[IPPO] episode={episode} return={total_return:.2f} success_rate={sr:.2f}")
    trainer.save(str(out_dir / "ippo.pt"))
    return metrics, trainer


def train_mappo(args, env: CoopPuzzleEnv, out_dir: Path, config: PPOConfig):
    trainer = MAPPOTrainer(env.obs_dim, env.state_dim, env.action_dim, env.possible_agents, config)
    metrics = []
    rng = np.random.default_rng(args.seed)
    pending_rollout = []
    last_update_metrics = {}
    best_score = float("-inf")

    def evaluation_score() -> tuple[float, float]:
        # PPO learns a stochastic policy. Compare checkpoints with the same
        # sampling sequence, then restore RNG so evaluation does not alter
        # subsequent training.
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(args.seed + 10_000)
        successes = []
        returns = []
        try:
            for eval_episode in range(args.selection_episodes):
                eval_env = CoopPuzzleEnv(level=env.level_name, max_steps=env.max_steps, seed=args.seed)
                eval_obs, _ = eval_env.reset(seed=args.seed + eval_episode)
                eval_return = 0.0
                success = False
                for _ in range(eval_env.max_steps):
                    eval_actions, _, _ = trainer.act(eval_obs, eval_env.state(), deterministic=False)
                    eval_obs, eval_rewards, terminations, truncations, infos = eval_env.step(eval_actions)
                    eval_return += float(np.mean(list(eval_rewards.values())))
                    success = any(info["success"] for info in infos.values())
                    if any(terminations.values()) or any(truncations.values()):
                        break
                successes.append(float(success))
                returns.append(eval_return)
        finally:
            torch.random.set_rng_state(torch_rng_state)
        success_rate = float(np.mean(successes))
        score = 1000.0 * success_rate + float(np.mean(returns))
        return score, success_rate

    for episode in range(1, args.episodes + 1):
        observations, _ = env.reset(seed=int(rng.integers(1_000_000)))
        total_return = 0.0
        success = False
        collisions = 0
        for _ in range(args.max_steps):
            state = env.state()
            actions, log_probs, value = trainer.act(observations, state)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = any(terminations.values()) or any(truncations.values())
            reward = float(np.mean(list(rewards.values())))
            total_return += reward
            success = success or any(info["success"] for info in infos.values())
            collisions += int(any(info["collision"] for info in infos.values()))
            pending_rollout.append(
                {
                    "observations": {agent: observations[agent] for agent in env.possible_agents},
                    "state": state,
                    "actions": dict(actions),
                    "log_probs": dict(log_probs),
                    "value": value,
                    "reward": reward,
                    "done": float(done),
                }
            )
            observations = next_obs
            if done:
                break

        should_update = episode % args.rollout_episodes == 0 or episode == args.episodes
        evaluation_success_rate = 0.0
        if should_update:
            last_update_metrics = trainer.update(pending_rollout)
            pending_rollout = []
            score, evaluation_success_rate = evaluation_score()
            if score > best_score:
                best_score = score
                trainer.save(str(out_dir / "mappo_best.pt"))

        row = {
            "episode": episode,
            "return": total_return,
            "success": int(success),
            "steps": env.step_count,
            "collisions": collisions,
            "evaluation_success_rate": evaluation_success_rate,
        }
        row.update(last_update_metrics)
        metrics.append(row)
        if episode % args.log_interval == 0:
            recent = metrics[-args.log_interval :]
            sr = sum(r["success"] for r in recent) / len(recent)
            print(f"[MAPPO] episode={episode} return={total_return:.2f} success_rate={sr:.2f}")
    trainer.save(str(out_dir / "mappo.pt"))
    return metrics, trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IPPO or MAPPO on the cooperative puzzle game.")
    parser.add_argument("--algorithm", choices=["ippo", "mappo"], default="mappo")
    parser.add_argument("--config", default="marl_game_ai/configs/puzzle.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--level", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()

    raw_config = load_config(args.config)
    env_config = raw_config.get("env", {})
    train_config = raw_config.get("train", {})
    algo_config = raw_config.get("algorithm", {})
    args.episodes = args.episodes or int(train_config.get("episodes", 300))
    args.rollout_episodes = int(train_config.get("rollout_episodes", 4))
    args.selection_episodes = int(train_config.get("selection_episodes", 5))
    args.max_steps = int(env_config.get("max_steps", 120))
    level = args.level or env_config.get("level", "basic")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = CoopPuzzleEnv(level=level, max_steps=args.max_steps, seed=args.seed)
    out_dir = ensure_dir(Path(args.out_dir) / args.algorithm)
    config = build_ppo_config(algo_config)
    if args.algorithm == "ippo":
        metrics, _ = train_ippo(args, env, out_dir, config)
    else:
        metrics, _ = train_mappo(args, env, out_dir, config)
    save_metrics_csv(metrics, out_dir / "training_metrics.csv")
    plot_training(metrics, out_dir / "training_curve.png")
    save_json(env.export_trajectory(), out_dir / "last_training_episode.json")
    print(f"Saved model, metrics, curve, and replay to: {out_dir}")


if __name__ == "__main__":
    main()
