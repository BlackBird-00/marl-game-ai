# 面向合作解谜游戏的多智能体强化学习方法研究与可视化实现

这是一个课程大作业风格的多智能体强化学习项目。项目实现了一个 2D 合作解谜小游戏：两个 agent 需要通过踩压力板、打开门、取得钥匙、到达终点来完成任务，并提供随机策略、规则策略、IPPO 和 MAPPO 的训练与评估代码。

## 项目结构

```text
marl_game_ai/
├─ envs/coop_puzzle_env.py      # 2D 合作解谜环境
├─ algorithms/ippo.py           # Independent PPO
├─ algorithms/mappo.py          # MAPPO: 集中 critic + 分散 actor
├─ visualizer/pygame_viewer.py  # pygame 回放可视化
├─ configs/puzzle.yaml          # 默认训练配置
├─ train.py                     # 训练入口
├─ evaluate.py                  # 评估与轨迹导出
└─ requirements.txt
```

## 快速开始

在工作区根目录运行：

```bash
python -m pip install -r requirements.txt
python -m marl_game_ai.evaluate --algorithm rule --episodes 3 --out-dir outputs/eval
python -m marl_game_ai.visualizer.pygame_viewer --replay outputs/eval/rule_replay.json
```

训练 MAPPO：

```bash
python -m marl_game_ai.train --algorithm mappo --episodes 300 --out-dir outputs
python -m marl_game_ai.evaluate --algorithm mappo --model outputs/mappo/mappo.pt --episodes 20 --out-dir outputs/eval
```

训练 IPPO：

```bash
python -m marl_game_ai.train --algorithm ippo --episodes 300 --out-dir outputs
python -m marl_game_ai.evaluate --algorithm ippo --model outputs/ippo/ippo.pt --episodes 20 --out-dir outputs/eval
```

如果 Windows/Anaconda 环境出现 `torch` DLL 加载错误，随机策略、规则策略和 pygame 回放仍可运行；训练 IPPO/MAPPO 前需要先修复 PyTorch 安装。若遇到 `NumPy 2.x` 与 `matplotlib` 或 `torch` 的兼容报错，先按 `requirements.txt` 安装 `numpy<2`。

## 上传到 GitHub

本目录已经适合初始化为 Git 仓库，`outputs/`、模型权重和缓存不会被提交。首次关联远程仓库时，在工作区根目录执行：

```bash
git remote add origin https://github.com/<your-name>/<repo-name>.git
git branch -M main
git push -u origin main
```

## 环境设定

默认地图包含：

- `A`：agent_0，推荐学习或执行“踩压力板”角色。
- `B`：agent_1，推荐学习或执行“取钥匙/到终点”角色。
- `P`：压力板，踩住后门打开。
- `D`：门，关闭时不可通过，打开后可通过。
- `K`：钥匙。
- `G`：终点。
- `T`：陷阱。
- `#`：墙。

动作空间：

```text
0 stay
1 up
2 down
3 left
4 right
```

观测包含自身位置、队友位置、压力板、门、钥匙、终点位置，以及门和钥匙状态。MAPPO 使用全局状态作为 centralized critic 输入，actor 仍只看局部观测。

## 实验指标

训练后会生成：

- `training_metrics.csv`：每个 episode 的回报、成功率、步数、碰撞次数、loss。
- `training_curve.png`：回报曲线和滑动成功率曲线。
- `*.pt`：模型权重。
- `*_replay.json`：可用于 pygame 播放的轨迹。

建议报告中对比：

- 随机策略 vs 规则策略 vs IPPO vs MAPPO。
- `basic`、`key_door`、`trap` 三种地图难度。
- 成功率、平均通关步数、平均回报、碰撞次数、训练收敛速度。

## 参考资料

- PPO: https://arxiv.org/abs/1707.06347
- MAPPO: https://arxiv.org/abs/2103.01955
- MADDPG: https://arxiv.org/abs/1706.02275
- PettingZoo: https://arxiv.org/abs/2009.14471
- PettingZoo Parallel API: https://pettingzoo.farama.org/api/parallel/
- Gymnasium 自定义环境: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
- Stable-Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- MAPPO 官方参考实现: https://github.com/marlbenchmark/on-policy
