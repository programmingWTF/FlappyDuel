> [English](./README.md) | **简体中文**

# FlappyDuel

**Flappy Bird，不过这一次你是和 AI 在「同一个世界」里面对面较量。**

FlappyDuel 用深度强化学习训练一个能玩 Flappy Bird 的智能体（Double/Dueling
DQN，兼容 Rainbow），然后让你和它**对战**。加载训练好的模型后，左右两个同步面板会
把你（红色）和 AI（金色）放进**同一片管道场地**。一方撞管出局后，另一方继续飞，
直到双方都出局，再比分数。

项目从一份单文件原型重构为可维护的 `flappyrl/` 包。对战模式默认跑在 **CPU** 上，
没有显卡也能玩。

---

## 特性

- **向量化无渲染模拟器**（`flappyrl/sim.py`）——纯 numpy，训练时不依赖 pygame，
  支持 N 个并行环境快速采集数据。状态为 18 维手工特征。
- **兼容 Rainbow 的智能体**（`flappyrl/agent.py`）——每个组件都是开关：Double DQN、
  Dueling 网络、**Noisy 网络**、**优先经验回放（PER）**、**N 步回报**、以及
  **分布式（C51）** 学习。
- **人机对战模式**（`versus.py`）——共享世界、左右双同步面板，一方失败另一方继续
  （正是「直到双方都失败」的行为）。
- **对 CPU 友好**——对战只需 `pygame` + `torch`，无需显卡（`--device cuda` 才用 GPU）。
- 在 **RTX 5060 Ti（Blackwell，sm_120）** 上可用，依赖 CUDA 12.8 的 PyTorch。

---

## 快速开始 —— 直接和 AI 对战

```bash
# 1. 安装两个运行依赖
pip install pygame
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 开玩（最佳模型已随仓库提供）
python versus.py --model-path checkpoints/best.pt
```

对局**默认暂停**——按一次 **UP（上箭头）** 开始。之后：

| 操作 | 作用 |
|---|---|
| **UP / 空格 / 点击** | 拍翅 |
| **R** / 蓝色 **RESTART（重开）** 按钮 | 重开（重建管道场景，回到暂停态） |
| **UP / 空格**（双方都出局后） | 开始新一局 |
| **ESC** / 关闭窗口 | 退出 |

```
+----------------+   +----------------+
|  你（红）       |   |  AI（金）      |
|  分数 12        |   |  分数 48        |
+----------------+   +----------------+
   PAUSED - press UP to start
```

每个面板**只显示自己的鸟**；撞管出局的鸟会变深灰并标注 `OUT`，但尸体会一直停在
屏幕上，直到双方都出局。AI 和人类用的是**完全相同**的碰撞判定代码——没有任何特殊
待遇（由 `test_collision_symmetry.py` 验证）。

> pygame 只支持一个操作系统窗口，所以「两个窗口」其实是一个窗口拆成两个对齐面板——
> 它们天然完全同步。

---

## 开发环境

本仓库用一个位于 `./env` 的 **conda prefix** 环境开发（不往 base 装任何东西）。
如果你想继续在这里开发，重建它：

```bash
# Windows
setup_env.bat
# Linux / macOS
bash setup_env.sh
# 这个 prefix 没有 activate 脚本 —— 直接调用解释器：
./env/python.exe train.py --algo baseline     # Windows
./env/python    train.py --algo baseline      # Linux/mac
```

普通克隆本仓库不需要这个 prefix——只要 `pip install pygame torch` 用自己的 python 即可。

---

## 训练

```bash
# 实测下来，朴素 Double+Dueling 比完整 Rainbow 效果更好
python train.py --algo baseline           # 推荐（效果最好）
python train.py --algo rainbow            # 完整 Rainbow（所有技巧全开）
python train.py --algo baseline --total-timesteps 1000000 --n-envs 16

# 从检查点继续训练（--resume）；--device cuda 用 GPU 训练
python train.py --algo baseline --resume checkpoints/best.pt --total-timesteps 500000
```

检查点落在 `checkpoints/<tag>/`，日志（CSV + TensorBoard）落在 `logs/<tag>/`。

## 评测

```bash
python evaluate.py --model-path checkpoints/best.pt --episodes 30
python evaluate.py --model-path checkpoints/best.pt --render
```

---

## 结果（贪心策略，30 局）

随仓库提供的 **`checkpoints/best.pt`** 就是 `baseline_long` 在 97.5 万步的检查点：

| 指标 | 数值 |
|---|---|
| 平均分 | **303.67** |
| 最高分 | **991** |

本任务上，**朴素 Double + Dueling 明显胜过完整 Rainbow**：每个 Rainbow 变体最高只
到 5–18 根管道，而 Double+Dueling 达到约 300 均分 / 991 最高分。完整对比与原因见
`AGENT.md`。

> **评测坑。** `evaluate()` 曾经有个全局 `max_steps` 上限，导致任何像样的模型几局
> 就被掐断，所有报出来的分数都严重偏低。现在改成按单局封顶 + 很大的全局上限。
> **比较模型一定要用同样、不被截断的评测。**

---

## 项目结构

```
flappyrl/
  config.py      环境/智能体/训练的配置（dataclass）+ 预设
  sim.py         向量化 Flappy Bird 模拟器（numpy，无渲染）
  render.py      可选的 pygame 渲染器（评测 + 人机对战）
  networks.py    Dueling / Noisy / 分布式（C51）网络
  buffer.py      均匀 + 优先（SumTree）经验回放缓冲
  agent.py       RainbowDQN 智能体（所有组件接线完毕）
  logger.py      CSV +（可选）TensorBoard 日志
train.py         训练入口
evaluate.py      对检查点做贪心评测（+可选渲染）
versus.py        人机对战（共享场景）
versus_smoke.py  无头验证共享世界流程
test_collision_symmetry.py  证明 AI 和人类判定一致
archive/         原始单文件原型（仅作参考）
AGENT.md         给未来 agent 的「活文档」：进度 + 路线图
```

---

## 实现要点 / 值得留意的坑

- **PER 的 SumTree 容量必须是 2 的幂**——堆式寻址（`2*idx+1`）对非幂次容量是错的；
  缓冲会把容量向上取到最近的 2 的幂。
- **C51 的取值区间**——`v_min/v_max` 必须匹配真实的 Q 值范围（这里是 `[-2, 8]`）；
  太宽智能体就学不会。
- **共享世界的管道**——必须用「每只鸟各自的 `passed` 标记」，否则只有 0 号鸟能计分；
  而且共享管道必须每一步都前进，不能绑在某只鸟身上，否则人类先死时世界会冻结。

---

## 路线图

1. **训练稳定性**——实测从好检查点续训反而把策略训崩了，评测极抖（相邻两次在 0↔125
   之间跳）。先加学习率调度 / epsilon 下限 / 按评测挑检查点，再考虑加步数。
2. **用修正后的（不被截断的）评测重新测 Rainbow。**
3. **自对弈 / 课程学习**，让 AI 适应人类水平。
4. 可选升级：PPO/SAC、像素输入、更大的网络。

---

## 许可证

[MIT](LICENSE)
