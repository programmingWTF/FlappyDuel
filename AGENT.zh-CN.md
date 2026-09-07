> [English](./AGENT.md) | **简体中文**

# AGENT.md — FlappyDuel（Flappy Bird 深度强化学习）—— 给后续 agent 的「活文档」

> **GitHub：** https://github.com/programmingWTF/FlappyDuel（公开仓库，默认分支
> `master`）。最佳模型随仓库提供：`checkpoints/best.pt`。

> **这份文件是写给以后接手本项目的任何 AI agent（或人类）看的。** 它记录*当前状态*、
> *什么能用*、*什么坏了*，以及*带优先级的路线图*。**随着进展请持续更新它**——
> 用户明确要求留一份文档，让未来的 agent 读完后能无缝接着干。

---

## 0. 给下一位 agent 的速览

- 这是一个 **Flappy Bird 强化学习**项目。目标：训练一个玩得好的智能体，并持续优化。
  同时需要一个**人机对战**模式。
- 代码已从单文件的 Dueling-Double-DQN 原型重构为干净的 `flappyrl/` 包（向量化模拟器
  + 兼容 Rainbow 的智能体）。
- **硬件：** RTX 5060 Ti 16GB = **Blackwell，计算能力 sm_120**。
  ⚠️ 标准 PyTorch（cu121/cu124）**跑不了**它——必须用 **CUDA 12.8 的 torch**
  （`--index-url https://download.pytorch.org/whl/cu128`）。如果看到
  `sm_120 is not compatible`，说明装错 torch 了。
- **环境在 `./env`**（conda prefix，Python 3.11）。不要往 base 装任何东西。
  用 `setup_env.bat`（Win）/ `setup_env.sh`（Linux/mac）重建。
- 这个 prefix **没有 `activate` 脚本**——直接调用解释器：
  `./env/python.exe <脚本>`（Win）或 `./env/python <脚本>`（Linux/mac）。
  （早期文档写过 `env/Scripts/activate` —— 那个路径在这里**不存在**。）

---

## 1. 项目结构

```
D:\Code\DQN\
├── AGENT.md            # 本文件
├── README.md
├── requirements.txt
├── setup_env.bat / .sh # 重建 ./env conda 环境
├── flappyrl/           # 包
│   ├── config.py       # EnvConfig / AgentConfig / TrainConfig + 预设
│   ├── sim.py          # 向量化无渲染 Flappy 模拟器（numpy）
│   ├── render.py       # pygame 渲染器：单面板（评测）+ 左右并排（对战）
│   ├── networks.py     # Dueling / Noisy / 分布式（C51）网络
│   ├── buffer.py       # 均匀 + 优先（SumTree）经验回放缓冲
│   ├── agent.py        # RainbowDQN 智能体（所有组件接线完毕）
│   └── logger.py       # CSV（+可选 TensorBoard）日志
├── train.py            # 训练入口
├── evaluate.py         # 对检查点做贪心评测（+可选渲染）
├── versus.py           # 人机对战 —— 左右两个同步面板（CPU，默认暂停）
├── versus_smoke.py     # 无头验证对战流程是否跑通
├── test_collision_symmetry.py  # 证明人类和 AI 判定完全一致
├── eval_all.py         # 把各个最终检查点并排对比
├── archive/            # 原始单文件原型（仅作参考）
├── checkpoints/        # <tag>/ckpt_step_*.pt  （已被 gitignore）
└── logs/               # <tag>/*.csv（+ tb/）   （已被 gitignore）
```

## 2. 如何运行

```bat
:: Windows —— 环境是 conda prefix；直接调 python（没有 activate 脚本）
.\env\python.exe train.py --algo rainbow
:: 训练朴素 Double+Dueling baseline（验证更快）
.\env\python.exe train.py --algo baseline
:: 覆盖参数
.\env\python.exe train.py --algo rainbow --total-timesteps 600000 --n-envs 16 --tag rainbow_big

:: 评测检查点（默认指向最佳模型：checkpoints/best.pt）
.\env\python.exe evaluate.py --model-path checkpoints/best.pt --episodes 30
.\env\python.exe evaluate.py --model-path checkpoints/best.pt --render

:: 无头验证共享世界对战流程是否正常
.\env\python.exe versus_smoke.py

:: 人机对战 —— 左右两个同步面板（左=你，右=AI）；默认暂停
.\env\python.exe versus.py --model-path checkpoints/best.pt
```

### 对战模式（人机）工作原理

- **一个窗口里的两个同步面板。** pygame 只允许*一个*操作系统窗口，所以「两个窗口」是
  用一个窗口拆成两个对齐面板实现的（`flappyrl/render.py` 里的 `SideBySideRenderer`；
  600×564 = 两个 288 宽的面板）。两个面板渲染的是**同一个**共享世界，所以左右永远
  同步、可直接对比。
- **默认暂停。** 第一次按 UP 开始一局；之后 UP / 空格 / 点击 = 拍翅。`ESC` 退出。
- **重开：** `R` 键或屏幕右下角的蓝色 **RESTART** 按钮（HUD 栏内）随时可用；一局结束后
  UP/空格也能开新局。重开会重建共享管道场景并回到暂停态。（按钮是
  `SideBySideRenderer.restart_rect`；versus.py 对鼠标点击做命中检测。）
- **默认 CPU**（`--device cpu`），无显卡也能跑；只需 `pygame` + `torch`。
- 每个面板**只显示自己的鸟**——应要求去掉了对面的灰色幽灵鸟，画面更干净。撞管出局的鸟
  变深灰、标注 `OUT`，尸体会冻在原地一直显示到**双方**都出局——这是故意的，也是「AI
  明明蹭到管子却没死！」这种误解的常见来源。
- `R` 会重建共享管道场景（`FlappySim.reset_all`），所以重开绝不会把鸟丢进上一局的残留
  管道里。

## 3. 算法设计（当前）

智能体是**一个 `RainbowDQN` 类**，每个组件都由 `AgentConfig` 开关控制：

| 组件 | 配置开关 | 状态 |
|------|---------|------|
| Double DQN | `double` | ✅ 开 |
| Dueling 网络 | `dueling` | ✅ 开 |
| Noisy 网络 | `noisy` | ✅（Rainbow） |
| PER（SumTree） | `per` | ✅（Rainbow） |
| N 步回报 | `n_step` | ✅（Rainbow=3） |
| 分布式 | `distributional`（C51，51 个原子） | ✅（Rainbow） |
| Baseline（D+D） | 仅 double+dueling 开 | ✅ 预设 |

- **状态：** 18 维手工特征（鸟 y、速度、离地/离顶距离，以及接下来 2 根管道的相对 x、
  缝隙中心、缝隙上下沿、到达时间）。与原型一致 → 旧检查点结构兼容。
- **动作：** 0 = 不拍翅，1 = 拍翅。离散，2 个动作。
- **奖励：** 活着 +0.01，死亡 −1.0，外加一个鼓励贴近下一根管道缝隙中心的小塑形项。
- **向量化模拟：** `n_envs` 个独立回合并行采集；智能体把 N 步聚合后的转移存入回放缓冲。
- **优化器：** AdamW，CUDA 上用 GradScaler（AMP），梯度裁剪到 10。

## 4. 当前进度  *（请定期更新本段）*

> 最后更新：**2026-09-07** —— 找到并晋升最佳模型；修复了对战 bug。

- [x] 环境重构为向量化无渲染模拟器。
- [x] Rainbow 智能体实现（全部 6 个组件）。
- [x] `versus.py` 人机对战模式实现**并修 bug**（见 §5）。
- [x] 冒烟测试通过（baseline + rainbow 前向/反向、存/读）。
- [x] Baseline（Double+Dueling）训练到 **100 万步**（`baseline_long`）。
- [x] **最佳模型晋升 → `checkpoints/best.pt`**
      （= `baseline_long/ckpt_step_975000.pt`）：**贪心平均 303.67，最高 991**，30 局。
- [x] `evaluate()` 截断 bug 修复 —— 旧分数严重偏低。
- [x] 共享世界（对战）bug 修复 + 无头验证（`versus_smoke.py`）。
- [x] 人机碰撞公平性**验证**（`test_collision_symmetry.py`）。
- [x] 出局鸟在 `versus.py` 里变灰 + 标 `OUT`（冻住的尸体原本会叠在管道上，看着像
      「还没死」）。
- [ ] **续训失败 —— 需要换思路。** 从 `best.pt` 续训 50 万步反而把策略**训崩**了
      （见下方 §4 备注）。
- [ ] 用（现已正确的）不被截断的评测重新测 Rainbow。

### 最近的贪心评测（30 局，已修正封顶 —— 可信）
| 检查点 | 平均 | 最高 | 备注 |
|---|---|---|---|
| **baseline_long @975k → `best.pt`** | **303.67** | **991** | **当前最佳**；稳定（30 局全 ≥ 24） |
| baseline_long @725k | 180.23 | 1106 | 上限高但不稳 |
| baseline_long @225k | 76.17 | 430 | |
| baseline_long @675k | 65.67 | 204 | |
| baseline_long @400k | 46.70 | 255 | |
| baseline @200k（`baseline/ckpt_final`） | 41.23 | 133 | 旧「25.73 / 最高 72」是截断假象 |

**各 Rainbow 变体在本任务上都明显弱于朴素 Double+Dueling。** 训练中实测：
`rainbow` ≈ 0–1（修复前是坏的）、`rainbow_fix` 最高 15–18、`rainbow_nonoise` 最高
~12（到 30 万步塌到 0）、`rainbow_pn` 最高 ~5。目前结论：**本任务上 Double+Dueling
胜过完整 Rainbow**——noisy 网络 / C51 这些附加项在这种小体量、稠密奖励的问题上拖慢了
学习。

> **⚠️ 评测坑（浪费了真实诊断时间 —— 别再踩）。** `evaluate()` 曾经有个全局
> `max_steps=30000` 上限，导致任何像样的模型约 3 局就被掐断，所有报分都偏低（旧
> 「baseline 最高 72」其实约 133；更早「最高约 340」是训练期指标，**是错的**）。现在
> 改成 `max_steps=5_000_000` + 单局安全上限 `50_000`。**比较模型务必用同样、不被截断
> 的评测。**

### ⚠️ 续训反而更差（别盲目照做）
从 `best.pt` 继续（`--resume`，epsilon 从 0.1 重启）再跑 50 万步，**毁掉**了策略：

| 节点 | 评测平均 | 评测最高 |
|---|---|---|
| 续训起点（`best.pt`） | 303.67 | 991 |
| +10 万 | 125.60 | 434 |
| +20 万 | 4.90 | 9 |
| +30 万 | 31.50 | 87 |
| +40 万 | 2.10 | 7 |
| +50 万（终点） | 3.60 | 25 |

所以 **`checkpoints/best.pt` 依然是 `baseline_long@975k`**。本任务训练*非常*不稳定
（相邻评测在平均 0 和 125 之间跳）。继续加步数前，请**先解决稳定性**——训练后期降 LR、
保持更高的 epsilon 下限，或按评测挑检查点——而不是一味多跑。

### 碰撞公平性已验证（人机）
`test_collision_symmetry.py` 证明两只鸟判定完全一致：

| 场景 | 谁死 |
|---|---|
| 两只鸟都在管身里 | 都死 |
| 只有 AI 在管身里 | 只有 AI |
| 只有人类在管身里 | 只有人类 |
| 两只鸟都在缝隙中央 | 都不死 |
| 同一策略、共享世界 | **同一步一起死** |

画出来的圆**就是**判定圆：`render.py` 画的 `bird_radius` 圆，圆心/半径与 `_collides()`
判定用的完全一致；画的管道矩形也与碰撞矩形一致。所以不存在「看着撞了却没判」的偏差——
AI 只是接近最优，常常只差一两像素穿过缝隙。出局鸟现在在 `versus.py` 里变灰 + 标 `OUT`，
因为冻住的尸体会一直显示到双方都出局（这是「AI 蹭到管子却没死！」误解的常见来源）。

## 5. 已知问题 / 注意事项

- **⚠️ 续训让模型变差（已实测）。** 从 `best.pt`（平均 303.67）用 `--resume` + eps 0.1
  续跑 50 万步把它训崩了：评测平均 `125.60 → 4.90 → 31.50 → 2.10 → 3.60`。
  **别默认「步数越多越好」——本任务训练不稳定。** 再训练前先让它稳定（后期降 LR、保持
  更高的 epsilon 下限、或保留按评测最佳的那个检查点）；否则一次「续训」会毁掉好策略、白
  费一小时。

- **Blackwell 的 torch：** 如果内核报错（`sm_120 is not compatible`），说明装错 torch。
  用 `--index-url https://download.pytorch.org/whl/cu128` 重装。
- **N 步 + 自动重置：** 智能体在终止步会清空 N 步窗口（丢掉每回合最后 <n 个转移）——可接受。
- 模拟器 `FlappySim` 是*无渲染*的；渲染只发生在 `render.py`。训练从不 import pygame。

### 首轮（2026-09-07）发现并修复的 bug
这些花了很多诊断时间——别再引入：

1. **PER 的 SumTree 容量必须是 2 的幂。** `SumTree` 的 `2*idx+1` 堆式寻址只对 2 的幂
   片数有效。100k/200k 这种容量会让采样出错（索引损坏 → 智能体学不会）。修复：在
   `PrioritizedReplayBuffer` 里把容量向上取到最近的 2 的幂。
2. **C51 取值区间太宽。** 原来 `v_min=-10, v_max=10`，而 Q 值实际在 `[-1, 6]`
   （活着 +0.01、塑形 ≤+0.05、死亡 −1）。原子无法解析有意义的区间 → 学不会。改成
   `v_min=-2, v_max=8`。
3. **共享世界（对战）的管道只有一个全局 `passed` 标记。** `shared_world=True` 时所有鸟
   穿过*同一份*管道列表，鸟 0 过管时设了 `passed=True`，鸟 1 就永远无法得分——AI（鸟 1）
   实际得了 **0 分**，而人类（鸟 0）得了 8 分。修复：`passed` 现在是**每只鸟一个列表**
   （`[False] * n`），按鸟索引，于是共享场景里每只鸟独立计分。
4. **共享管道只在处理鸟 0 时移动。** 移动被 `if i == 0` 限制，但死掉的鸟在这行之前就被
   `continue` 跳过了——所以人类先死时管道**永远冻结**，存活的 AI 面对静止场景永不死
   （对战永不结束）。修复：共享管道在每步**进入按鸟循环之前统一前进一次**，只要还有至少
   一只鸟活着。

这两个对战 bug 都是被 `versus_smoke.py` 抓到的——它无头驱动模拟（没人、不显示）。改完
`sim.py` 后重跑它：

```bat
.\env\python.exe versus_smoke.py
```

预期输出：「AI vs AI」两只鸟同一步一起死、分数**相等**；「AI vs 摆烂人类」人类早早死
（0 分）、AI 继续飞到自己死 → `PASS`。

### 组件消融（首轮，7 万步短跑）
朴素 Double+Dueling 能用（评测↑）。每个附加项最初都*失败*，直到上面两个 bug 修好。修好后
用 baseline 的学习率（1e-4）重跑完整 Rainbow 确认每个组件有帮助。**Noisy 网络最娇气**——
如果完整 Rainbow 还是不如 baseline，就去掉 noisy（`--no-noisy`）靠 epsilon-greedy；那个
变体稳健。

## 6. 路线图（带优先级 —— 从顶部接着做）

1. ~~冒烟测试~~ ✅ 完成。
2. ~~训练 baseline（Double+Dueling）~~ ✅ 完成 —— 100 万步；最佳是
   `baseline_long@975k` → **平均 303.67 / 最高 991**，已晋升 `best.pt`。
3. ~~训练 Rainbow + 消融~~ ✅ 完成 —— 本任务上每个 Rainbow 变体都明显弱于朴素 Double+Dueling。
4. ~~修复 `evaluate()` 截断~~ ✅ 完成 —— 报分现在可信。
5. ~~修复 + 验证共享世界对战模式~~ ✅ 完成（修复 2 个真 bug）。
6. **先解决训练不稳定，再继续训练。** `--resume` 已实现且能用，但从 `best.pt` 续训*训崩*
   了策略（见 §4）。先稳定（LR 调度 / epsilon 下限 / 按评测挑检查点），再把平均推过 ~300、
   最高推过 ~1000。
7. **用修正后的评测重测 Rainbow** —— 它的数字来自被截断的评测，定论「baseline 胜」前先复查。
8. **用最佳模型演示人机对战**（`versus.py`）并记录定性行为。
9. **可选升级**（只有碰到天花板才做）：
   - 在同一向量化模拟器上跑 PPO / SAC 做对比。
   - 自对弈 / 基于群体的训练。
   - 像素-CNN 变体（原始画面），如果基于特征的到达瓶颈。
   - 用小脚本做超参扫描。
10. **每次有意义的改动后，保持本文件和 `README.md` 最新。** 到里程碑时把代码 + 更新后的
    文档提交到 git（`git add -A && git commit`）。

## 7. 现在就能接着跑的确切命令

```bat
:: 1) 确认最佳模型仍然表现好（预期平均 ~300，最高 ~1000）
.\env\python.exe evaluate.py --model-path checkpoints/best.pt --episodes 30

:: 2) 改过 sim.py 后，重新验证对战流程
.\env\python.exe versus_smoke.py

:: 3) 和模型对战：左右两个同步面板（左=你，右=AI）。
::    默认暂停 —— 按 UP 开始，之后 UP/空格/点击拍翅。
::    默认 CPU（--device cuda 用 GPU）。
.\env\python.exe versus.py --model-path checkpoints/best.pt

:: 4) 继续优化 —— 从最佳检查点续训
.\env\python.exe train.py --algo baseline --resume checkpoints/best.pt ^
    --total-timesteps 1000000 --tag baseline_cont

:: 5) 或从头跑新的一轮
.\env\python.exe train.py --algo baseline --total-timesteps 1000000 --tag baseline_x
```

## 8. 最佳模型从哪来

`checkpoints/best.pt` 是 `checkpoints/baseline_long/ckpt_step_975000.pt` 的逐字节拷贝 ——
由下面命令产出：

```bat
.\env\python.exe train.py --algo baseline --lr 1e-4 --total-timesteps 1000000 ^
    --tag baseline_long --save-every 100000
```

注意评测曲线**非常抖**（相邻两次评测可能从平均 0 跳到平均 125），所以永远别凭单次评测
判断一轮训练——一定要用完整 30 局贪心评测复查最靠前的几个检查点。
