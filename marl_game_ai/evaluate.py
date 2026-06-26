from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from marl_game_ai.baselines import RandomPolicy, RuleBasedPolicy
from marl_game_ai.envs import CoopPuzzleEnv
from marl_game_ai.utils import ensure_dir, save_json


def run_episode(
    env: CoopPuzzleEnv,
    policy,
    algorithm: str,
    deterministic: bool = True,
    temperature: float = 1.0,
):
    observations, _ = env.reset()
    total_return = 0.0
    success = False
    collisions = 0
    for _ in range(env.max_steps):
        if algorithm == "random" or algorithm == "rule":
            actions = policy.act(env, observations)
        elif algorithm == "ippo":
            actions, _, _ = policy.act(
                observations,
                deterministic=deterministic,
                temperature=temperature,
            )
        elif algorithm == "mappo":
            actions, _, _ = policy.act(
                observations,
                env.state(),
                deterministic=deterministic,
                temperature=temperature,
            )
        else:
            raise ValueError(algorithm)
        observations, rewards, terminations, truncations, infos = env.step(actions)
        total_return += float(np.mean(list(rewards.values())))
        success = success or any(info["success"] for info in infos.values())
        collisions += int(any(info["collision"] for info in infos.values()))
        if any(terminations.values()) or any(truncations.values()):
            break
    return {
        "return": total_return,
        "success": int(success),
        "steps": env.step_count,
        "collisions": collisions,
        "trajectory": env.export_trajectory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy or baseline.")
    parser.add_argument("--algorithm", choices=["random", "rule", "ippo", "mappo"], default="rule")
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--level", default="basic")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the learned PPO policy instead of taking argmax.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature used with --stochastic; lower values reduce wandering.",
    )
    args = parser.parse_args()

    env = CoopPuzzleEnv(level=args.level, max_steps=args.max_steps, seed=args.seed)
    if args.algorithm == "random":
        policy = RandomPolicy(env.action_dim, seed=args.seed)
    elif args.algorithm == "rule":
        policy = RuleBasedPolicy()
    elif args.algorithm == "ippo":
        if not args.model:
            raise ValueError("--model is required for ippo")
        from marl_game_ai.algorithms.common import PPOConfig
        from marl_game_ai.algorithms.ippo import IPPOTrainer

        policy = IPPOTrainer(env.obs_dim, env.action_dim, env.possible_agents, PPOConfig())
        policy.load(args.model)
    else:
        if not args.model:
            raise ValueError("--model is required for mappo")
        from marl_game_ai.algorithms.common import PPOConfig
        from marl_game_ai.algorithms.mappo import MAPPOTrainer

        policy = MAPPOTrainer(env.obs_dim, env.state_dim, env.action_dim, env.possible_agents, PPOConfig())
        policy.load(args.model)

    results = []
    last_trajectory = []
    for episode in range(args.episodes):
        result = run_episode(
            env,
            policy,
            args.algorithm,
            deterministic=not args.stochastic,
            temperature=args.temperature,
        )
        last_trajectory = result.pop("trajectory")
        result["episode"] = episode + 1
        results.append(result)
    out_dir = ensure_dir(args.out_dir)
    save_json(results, Path(out_dir) / f"{args.algorithm}_summary.json")
    save_json(last_trajectory, Path(out_dir) / f"{args.algorithm}_replay.json")
    successful_steps = [r["steps"] for r in results if r["success"]]
    success_steps_text = f"{np.mean(successful_steps):.1f}" if successful_steps else "n/a"
    print(
        f"{args.algorithm} ({'stochastic' if args.stochastic else 'deterministic'}): "
        f"success_rate={np.mean([r['success'] for r in results]):.2f}, "
        f"avg_return={np.mean([r['return'] for r in results]):.2f}, "
        f"avg_steps={np.mean([r['steps'] for r in results]):.1f}, "
        f"avg_success_steps={success_steps_text}"
    )
    print(f"Saved summary and replay to: {out_dir}")


if __name__ == "__main__":
    main()
