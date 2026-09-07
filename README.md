# Flappy Bird — Reinforcement Learning

A clean, modular, extensible toolkit for teaching an agent to play **Flappy
Bird** with modern deep-RL algorithms. Started from a single-file Dueling
Double-DQN prototype and rebuilt from scratch as a maintainable package, with
room to grow toward full Rainbow and beyond.

## Highlights

- **Vectorized, headless simulator** (`flappyrl/sim.py`) — pure numpy, no
  pygame during training, supports N parallel environments for fast data
  collection. Same 18-dim engineered feature state as the original.
- **Rainbow-ready agent** (`flappyrl/agent.py`) — every component is a toggle:
  Double DQN, Dueling nets, **Noisy nets**, **Prioritized Experience Replay**,
  **N-step returns**, and **Distributional (C51)** learning.
- **Human-vs-AI mode** (`versus.py`) — load a trained model and play against
  the AI in the *same* shared scene; when one side dies the other keeps going
  until both are out. Great for qualitatively judging the agent.
- Runs on the RTX 5060 Ti (Blackwell, sm_120) via CUDA 12.8 PyTorch.

## Layout

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
archive/         the original single-file prototype (kept for reference)
AGENT.md         living document for future agents (progress + roadmap)
```

## Setup

```bash
# Windows
setup_env.bat
# or Linux/mac
bash setup_env.sh
```

The environment is a **conda prefix** installed at `./env`. There is no
`activate` script — call the interpreter directly:

```bash
./env/python.exe train.py --algo rainbow     # Windows
./env/python    train.py --algo rainbow      # Linux/mac
```

## Train

```bash
# NOTE: empirically, plain Double+Dueling beats full Rainbow on this task.
./env/python.exe train.py --algo baseline           # recommended (best results)
./env/python.exe train.py --algo rainbow            # full Rainbow (all tricks on)
./env/python.exe train.py --algo baseline --total-timesteps 1000000 --n-envs 16
```

Checkpoints land in `checkpoints/<tag>/`, logs in `logs/<tag>/` (CSV +
TensorBoard).

## Results (greedy, 30 episodes)

The current best model is **`checkpoints/best.pt`** (= `baseline_long`
checkpoint @975k steps):

| metric | value |
|---|---|
| mean score | **303.67** |
| max score | **991** |

Plain **Double + Dueling DQN clearly beat full Rainbow** on this task:
every Rainbow variant topped out around 5–18 pipes, while Double+Dueling
reached ~300 mean / ~991 max. See `AGENT.md` for the full comparison and the
reasoning.

## Evaluate

```bash
./env/python.exe evaluate.py --model-path checkpoints/best.pt --episodes 30
./env/python.exe evaluate.py --model-path checkpoints/best.pt --render
```

## Human vs AI

```bash
./env/python.exe versus.py --model-path checkpoints/best.pt
```

Two **synchronized** views sit side by side, driven by the same shared world:

```
+----------------+   +----------------+
|  YOU (red)     |   |  AI (gold)     |
|  score 12      |   |  score 48      |
+----------------+   +----------------+
   PAUSED - press UP to start
```

- The round **starts paused** — press **UP** once to begin.
- Once running: **UP / SPACE / click** to flap, **ESC** to quit, **R** to
  restart after both sides are out.
- Each panel shows **only its own bird** (no ghost of the other side), so the
  view stays clean. A bird that has crashed turns dark grey and is labelled
  `OUT`, but its corpse stays on screen until both are out.
- When one side crashes the other keeps flying until both have failed, then the
  scores are compared.

It runs on the **CPU by default**, so anyone can play it without a GPU
(`--device cuda` to force the GPU). Only `pygame` and `torch` are needed:

```bash
pip install pygame
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> pygame only supports a single OS window, so the "two windows" are one window
> split into two aligned panels — they are always perfectly in lock-step.

See **AGENT.md** for the current status, results, and the optimization
roadmap that any future agent should continue.
