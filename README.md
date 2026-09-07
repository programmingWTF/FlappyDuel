> **English** | [简体中文](./README.zh-CN.md)

# FlappyDuel

**Flappy Bird, but you play against the AI — side by side, in the same world.**

FlappyDuel trains a deep-RL agent to play Flappy Bird (Double/Dueling DQN,
Rainbow-ready), and then lets you **duel** it. Load a trained model and a
synchronized two-panel view drops you (red) and the AI (gold) into the *same*
pipe field. When one of you crashes, the other keeps flying until both are
out — then the scores are compared.

Rebuilt from a single-file prototype into a maintainable `flappyrl/` package.
The game runs on **CPU by default**, so anyone can play without a GPU.

---

## Features

- **Vectorized, headless simulator** (`flappyrl/sim.py`) — pure numpy, no
  pygame during training, supports N parallel environments for fast data
  collection. 18-dim engineered feature state.
- **Rainbow-ready agent** (`flappyrl/agent.py`) — every component is a toggle:
  Double DQN, Dueling nets, **Noisy nets**, **Prioritized Experience Replay**,
  **N-step returns**, and **Distributional (C51)** learning.
- **Human-vs-AI duel mode** (`versus.py`) — shared world, two synced panels,
  one side keeps going after the other crashes (exactly the "until both fail"
  behaviour).
- **CPU-friendly** — the play mode needs only `pygame` + `torch`; no GPU
  required (`--device cuda` to use one).
- Works on **RTX 5060 Ti (Blackwell, sm_120)** via CUDA 12.8 PyTorch.

---

## Quick start — play against the AI

```bash
# 1. install the two runtime deps
pip install pygame
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. play (the best model is included in the repo)
python versus.py --model-path checkpoints/best.pt
```

The round **starts paused** — press **UP** once to begin. Then:

| input | action |
|---|---|
| **UP** / **SPACE** / **click** | flap |
| **R** / blue **RESTART** button | restart (rebuilds the field, back to paused) |
| **UP** / **SPACE** (after game over) | start a new round |
| **ESC** / close window | quit |

```
+----------------+   +----------------+
|  YOU (red)     |   |  AI (gold)     |
|  score 12      |   |  score 48      |
+----------------+   +----------------+
   PAUSED - press UP to start
```

Each panel shows **only its own bird**; a crashed bird turns dark grey and is
labelled `OUT`, but its corpse stays on screen until both sides are out. The
AI is judged by the **exact same** collision code as you — there is no special
treatment (proven by `test_collision_symmetry.py`).

> pygame supports only one OS window, so the "two windows" are one window split
> into two clipped panels — they are always perfectly in lock-step.

---

## Development setup

This repo was developed with a **conda prefix** environment living at `./env`
(nothing installed into base). If you continue development here, recreate it:

```bash
# Windows
setup_env.bat
# Linux / macOS
bash setup_env.sh
# there is NO activate script — call the interpreter directly:
./env/python.exe train.py --algo baseline     # Windows
./env/python    train.py --algo baseline      # Linux/mac
```

For a normal clone you don't need the prefix — just `pip install pygame torch`
and use your own `python`.

---

## Train

```bash
# Empirically, plain Double+Dueling beats full Rainbow on this task.
python train.py --algo baseline           # recommended (best results)
python train.py --algo rainbow            # full Rainbow (all tricks on)
python train.py --algo baseline --total-timesteps 1000000 --n-envs 16

# continue from a checkpoint (--resume); --device cuda to train on GPU
python train.py --algo baseline --resume checkpoints/best.pt --total-timesteps 500000
```

Checkpoints land in `checkpoints/<tag>/`, logs (CSV + TensorBoard) in
`logs/<tag>/`.

## Evaluate

```bash
python evaluate.py --model-path checkpoints/best.pt --episodes 30
python evaluate.py --model-path checkpoints/best.pt --render
```

---

## Results (greedy, 30 episodes)

The shipped **`checkpoints/best.pt`** is the `baseline_long` checkpoint @ 975k
steps:

| metric | value |
|---|---|
| mean score | **303.67** |
| max score | **991** |

Plain **Double + Dueling DQN clearly beats full Rainbow** on this task: every
Rainbow variant topped out around 5–18 pipes, while Double+Dueling reached
~300 mean / ~991 max. See `AGENT.md` for the full comparison and reasoning.

> **Evaluation gotcha.** `evaluate()` used to cap at a global `max_steps`, so
> any decent model was cut off after a few episodes and every reported score
> was an understatement. It is now capped per-episode with a generous global
> ceiling. **Always compare models with the same, non-truncated eval.**

---

## Project structure

```
flappyrl/
  config.py      dataclass configs (env / agent / training) + presets
  sim.py         vectorized Flappy Bird simulator (numpy, headless)
  render.py      optional pygame renderer (eval + human-vs-AI)
  networks.py    Dueling / Noisy / Distributional (C51) networks
  buffer.py      uniform + prioritized (SumTree) replay buffers
  agent.py       RainbowDQN agent (all components wired together)
  logger.py      CSV + (optional) TensorBoard logger
train.py         training entry point
evaluate.py      greedy evaluation of a checkpoint (+ optional render)
versus.py        human vs AI in a shared scene
versus_smoke.py  headless verification of the shared-world pipeline
test_collision_symmetry.py  proves the AI is judged like the human
archive/         the original single-file prototype (kept for reference)
AGENT.md         living handoff doc: progress + roadmap for future agents
```

---

## Implementation notes / gotchas worth knowing

- **PER SumTree must be a power of 2** — heap navigation (`2*idx+1`) is wrong
  for other capacities; the buffer rounds capacity up to the next pow2.
- **C51 support range** — `v_min/v_max` must match the real Q range
  (`[-2, 8]` here); too wide and the agent never learns.
- **Shared-world pipes** — a per-bird `passed` flag is required or only bird 0
  can score; and shared pipes must advance every step regardless of which bird
  is alive, or the world freezes when the human dies first.

---

## Roadmap

1. **Training stability** — resuming from a good checkpoint collapsed the
   policy in testing; the eval is very noisy (swings 0 ↔ 125). Add LR
   schedule / epsilon floor / checkpoint-by-eval-selection before training
   longer.
2. **Re-benchmark Rainbow** with the corrected (non-truncated) eval.
3. **Self-play / curriculum** so the AI adapts to the human's skill.
4. Optional upgrades: PPO/SAC, frame-stack pixels, larger nets.

---

## License

[MIT](LICENSE)
