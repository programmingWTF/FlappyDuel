# AGENT.md — Flappy Bird Deep-RL Project (living handoff document)

> **This file is for any AI agent (or human) that picks up this project later.**
> It records the *current state*, *what works*, *what is broken*, and the
> *prioritized roadmap*. **Keep it updated as you make progress** — the user
> explicitly asked for a document that any future agent can read to continue
> seamlessly.

---

## 0. TL;DR for the next agent

- This is a **Flappy Bird RL** project. Goal: train an agent that plays Flappy
  Bird well, then keep optimizing. A **human-vs-AI** mode is also required.
- Code was rebuilt from a single-file Dueling-Double-DQN prototype into a
  clean package `flappyrl/` (vectorized sim + Rainbow-capable agent).
- **Hardware:** RTX 5060 Ti 16GB = **Blackwell, compute capability sm_120**.
  ⚠️ Standard PyTorch (cu121/cu124) does **NOT** run on it — you must use
  **torch with CUDA 12.8** (`--index-url https://download.pytorch.org/whl/cu128`).
  If you see `sm_120 is not compatible`, the wrong torch is installed.
- **Env lives at `./env`** (conda prefix, Python 3.11). Nothing goes in base.
  Recreate with `setup_env.bat` (Win) / `setup_env.sh` (Linux/mac).
- There is **no `activate` script** in this prefix — call the interpreter
  directly: `./env/python.exe <script>` (Win) or `./env/python <script>`
  (Linux/mac). (Earlier docs said `env/Scripts/activate` — that path does
  NOT exist here.)

---

## 1. Project layout

```
D:\Code\DQN\
├── AGENT.md            # this file
├── README.md
├── requirements.txt
├── setup_env.bat / .sh # recreate the ./env conda environment
├── flappyrl/           # the package
│   ├── config.py       # EnvConfig / AgentConfig / TrainConfig + presets
│   ├── sim.py          # vectorized HEADLESS Flappy sim (numpy)
│   ├── render.py       # pygame renderers: single-panel (eval) + side-by-side (versus)
│   ├── networks.py     # Dueling / Noisy / Distributional (C51) nets
│   ├── buffer.py       # uniform + prioritized (SumTree) replay buffers
│   ├── agent.py        # RainbowDQN agent (all components wired)
│   └── logger.py       # CSV (+ optional TensorBoard) logger
├── train.py            # training entry point
├── evaluate.py         # greedy eval of a checkpoint (+ optional render)
├── versus.py           # HUMAN vs AI — two synchronized panels (CPU, paused start)
├── versus_smoke.py     # headless check that the versus pipeline works
├── test_collision_symmetry.py  # proves human and AI are judged identically
├── eval_all.py         # compare every final checkpoint side by side
├── archive/            # ORIGINAL single-file prototype (reference only)
├── checkpoints/        # <tag>/ckpt_step_*.pt  (gitignored)
└── logs/               # <tag>/*.csv (+ tb/)   (gitignored)
```

## 2. How to run

```bat
:: Windows — the env is a conda prefix; call python directly (no activate script)
.\env\python.exe train.py --algo rainbow
:: Train plain Double+Dueling baseline (faster to validate)
.\env\python.exe train.py --algo baseline
:: Overrides
.\env\python.exe train.py --algo rainbow --total-timesteps 600000 --n-envs 16 --tag rainbow_big

:: Evaluate a checkpoint (defaults to the BEST model: checkpoints/best.pt)
.\env\python.exe evaluate.py --model-path checkpoints/best.pt --episodes 30
.\env\python.exe evaluate.py --model-path checkpoints/best.pt --render

:: Headless verification that the shared-world versus pipeline works
.\env\python.exe versus_smoke.py

:: Human vs AI — two synchronized panels (LEFT = you, RIGHT = AI); starts PAUSED
.\env\python.exe versus.py --model-path checkpoints/best.pt
```

### Versus mode (human vs AI) — how it works

- **Two synchronized panels in one window.** pygame allows only a *single* OS
  window, so "two windows" is implemented as one window split into two aligned
  panels (`SideBySideRenderer` in `flappyrl/render.py`; 600×564 = two 288-wide
  panels). Both panels render the **same** shared world, so the sides are
  always in lock-step and directly comparable.
- **Starts PAUSED.** The first UP press starts the round; later UP / SPACE /
  click presses flap. `ESC` quits.
- **Restart:** `R` or the on-screen blue RESTART button (bottom-right of the
  HUD bar) works at any time; once a round is over, UP/SPACE also start a new
  one. A restart rebuilds the shared pipe field and returns to the paused
  state. (The button is `SideBySideRenderer.restart_rect`; versus.py hit-tests
  mouse clicks against it.)
- **CPU by default** (`--device cpu`) so the game runs on any machine without a
  GPU; only `pygame` + `torch` are needed.
- Each panel shows **only its own bird** — the opponent's grey ghost was
  removed at the user's request, so the view stays clean. A crashed bird turns
  dark grey, is labelled `OUT`, and its corpse stays frozen on screen until
  **both** sides are out — intentional, and the usual source of "the AI
  clipped a pipe and survived!" confusion.
- `R` rebuilds the shared pipe field (`FlappySim.reset_all`), so a restart
  never drops the bird into the previous round's pipes.

## 3. Algorithm design (current)

The agent is **one `RainbowDQN` class** with every component toggled from
`AgentConfig`:

| Component        | Config flag | Status |
|------------------|-------------|--------|
| Double DQN       | `double`    | ✅ on  |
| Dueling net      | `dueling`   | ✅ on  |
| Noisy nets       | `noisy`     | ✅ (Rainbow) |
| PER (SumTree)    | `per`       | ✅ (Rainbow) |
| N-step returns   | `n_step`    | ✅ (Rainbow=3) |
| Distributional   | `distributional` (C51, 51 atoms) | ✅ (Rainbow) |
| Baseline (D+D)   | all off except double+dueling | ✅ preset |

- **State:** 18-dim engineered features (bird y, velocity, distances to ground/
  sky, and 2 upcoming pipes' relative x, gap center, gap edges, time-to-reach).
  Same as the original prototype → old checkpoints structurally compatible.
- **Action:** 0 = noop, 1 = flap. Discrete, 2 actions.
- **Reward:** +0.01 alive, −1.0 on death, plus a small shaping term rewarding
  staying near the next pipe's gap center.
- **Vectorized sim:** `n_envs` independent episodes collected in parallel; the
  agent stores n-step aggregated transitions into the replay buffer.
- **Optimizer:** AdamW, GradScaler (AMP) on CUDA, gradient clipping at 10.

## 4. Current progress  *(UPDATE THIS SECTION REGULARLY)*

> Last updated: **2026-09-07** — best model found & promoted; versus bugs fixed.

- [x] Environment rebuilt as vectorized headless sim.
- [x] Rainbow agent implemented (all 6 components).
- [x] `versus.py` human-vs-AI mode implemented **and bug-fixed** (see §5).
- [x] Smoke test passed (baseline + rainbow forward/backward, save/load).
- [x] Baseline (Double+Dueling) trained to **1M steps** (`baseline_long`).
- [x] **Best model promoted → `checkpoints/best.pt`**
      (= `baseline_long/ckpt_step_975000.pt`): **greedy mean 303.67, max 991**
      over 30 episodes.
- [x] `evaluate()` truncation bug fixed — old scores were badly understated.
- [x] Shared-world (versus) bugs fixed + verified headlessly (`versus_smoke.py`).
- [x] Collision fairness human-vs-AI **verified** (`test_collision_symmetry.py`).
- [x] Dead birds greyed out + labelled `OUT` in `versus.py` (a frozen corpse
      used to stay drawn on top of a pipe and looked like "not dead yet").
- [ ] **Continue training failed — needs a different approach.** Resuming from
      `best.pt` for 500k steps *collapsed* the policy (see §4 note below).
- [ ] Re-check Rainbow with the (now correct) non-truncated eval.

### Latest greedy evaluations (30 episodes, FIXED cap — trustworthy)
| checkpoint | mean | max | notes |
|---|---|---|---|
| **baseline_long @975k → `best.pt`** | **303.67** | **991** | **current best**; consistent (all 30 eps ≥ 24) |
| baseline_long @725k | 180.23 | 1106 | high ceiling but spiky |
| baseline_long @225k | 76.17 | 430 | |
| baseline_long @675k | 65.67 | 204 | |
| baseline_long @400k | 46.70 | 255 | |
| baseline @200k (`baseline/ckpt_final`) | 41.23 | 133 | old "25.73 / max 72" was a truncation artifact |

**Rainbow variants are all clearly weaker than plain Double+Dueling here.**
Measured during training: `rainbow` ≈ 0–1 (broken, pre-fix run),
`rainbow_fix` max 15–18, `rainbow_nonoise` max ~12 (collapsed to 0 by 300k),
`rainbow_pn` max ~5. Conclusion so far: **Double+Dueling beats full Rainbow on
this task** — the noisy-nets / C51 additions slow learning on this small,
dense-reward problem.

> **⚠️ Evaluation gotcha (cost real diagnostic time — don't repeat it).**
> `evaluate()` used to cap at `max_steps=30000` *globally*, so any decent model
> was cut off after ~3 episodes and every reported score was an understatement
> (the old "baseline max 72" is really ~133; the earlier "max ~340" claim was a
> training-time artifact and is **false**). It is now `max_steps=5_000_000`
> with a per-episode safety cap of `50_000`. **Always compare models using the
> same, non-truncated eval.**

### ⚠️ Resuming training made it WORSE (don't repeat blindly)
Continuing from `best.pt` (`--resume`, epsilon restarted at 0.1) for another
500k steps **destroyed** the policy:

| point | eval mean | eval max |
|---|---|---|
| resume start (`best.pt`) | 303.67 | 991 |
| +100k | 125.60 | 434 |
| +200k | 4.90 | 9 |
| +300k | 31.50 | 87 |
| +400k | 2.10 | 7 |
| +500k (final) | 3.60 | 25 |

So **`checkpoints/best.pt` is still `baseline_long@975k`**. Training on this
task is *very* unstable (adjacent evals swing between mean 0 and 125). Before
training longer, **fix the instability** — lower the LR late in training, keep
a higher epsilon floor, or select checkpoints by eval — rather than just
running more steps.

### ⚠️ Continuation training (`baseline_cont`) FAILED — `best.pt` unchanged
Resuming from `best.pt` (`--resume`, epsilon restarted at 0.1) for another
500k steps **destroyed** the policy:

| eval point | mean | max |
|---|---|---|
| start (`best.pt`) | 303.67 | 991 |
| +100k | 125.60 | 434 |
| +200k | 4.90 | 9 |
| +300k | 31.50 | 87 |
| +400k | 2.10 | 7 |
| +500k | 3.60 | 25 |

So **`checkpoints/best.pt` is still `baseline_long@975k`**. Training on this
task is *very* unstable (adjacent evals swing between mean 0 and 125). The fix
is to **stabilise** (LR schedule, higher epsilon floor, pick checkpoints by
eval) — not to simply train longer.

### Collision fairness is verified (human vs AI)
`test_collision_symmetry.py` proves both birds are judged identically:

| scenario | who dies |
|---|---|
| both birds inside pipe body | both |
| only AI inside pipe body | only AI |
| only human inside pipe body | only human |
| both in the middle of the gap | neither |
| identical policy, shared world | **both on the same step** |

The drawn circle **is** the hitbox: `render.py` draws a circle of
`bird_radius` at exactly the centre/radius that `_collides()` tests, and the
drawn pipe rects match the collision rects. So there is no "looks like a hit
but wasn't" discrepancy — the AI is simply near-optimal and threads gaps with
a pixel or two of clearance. Dead birds are now greyed out and labelled `OUT`
in `versus.py`, because a frozen corpse stays drawn on screen until both sides
are out (the usual source of "the AI clipped a pipe and survived!" confusion).


## 5. Known issues / watch-outs

- **⚠️ Resuming training made the model WORSE (measured).** Continuing from
  `best.pt` (mean 303.67) with `--resume` + eps 0.1 for 500k more steps
  collapsed it: eval means went `125.60 → 4.90 → 31.50 → 2.10 → 3.60`.
  **Do not assume "more steps = better" here — training is unstable.**
  Before attempting further training, stabilise it (lower LR late, keep a
  higher epsilon floor, or keep the best-by-eval checkpoint); otherwise a
  "continuation" run will destroy a good policy and waste an hour.

- **Blackwell torch**: if kernels fail (`sm_120 is not compatible`), the wrong
  torch is installed. Reinstall with `--index-url https://download.pytorch.org/whl/cu128`.
- **N-step + auto-reset**: the agent flushes the n-step window on terminal
  steps (losing the last <n transitions of each episode) — acceptable.
- The sim's `FlappySim` is *headless*; rendering only happens in `render.py`.
  Training never imports pygame.

### Bugs found & fixed during the first session (2026-09-07)
These cost a lot of diagnostic time — don't reintroduce them:

1. **PER SumTree needed power-of-2 capacity.** `SumTree`'s `2*idx+1` heap
   navigation is only valid for a power-of-2 leaf count. Buffer sizes like
   100k/200k broke sampling (corrupted indices → agent never learned). Fixed
   by rounding capacity up to the next pow2 in `PrioritizedReplayBuffer`.
2. **C51 support far too wide.** Original `v_min=-10, v_max=10` for Q-values
   that actually live in `[-1, 6]` (alive +0.01, shaping ≤+0.05, death −1).
   Atoms couldn't resolve the meaningful range → no learning. Fixed to
   `v_min=-2, v_max=8`.
3. **Shared-world (versus) pipes had ONE global `passed` flag.** In
   `shared_world=True` all birds fly through the *same* pipe list, so when
   bird 0 passed a pipe it set `passed=True` and bird 1 could never score it —
   the AI (bird 1) literally scored **0** while the human (bird 0) scored 8.
   Fixed: `passed` is now a **per-bird list** (`[False] * n`), indexed by bird
   index, so every bird scores independently in a shared scene.
4. **Shared pipes only moved when bird 0 was processed.** The move was gated
   on `if i == 0`, but a dead bird is `continue`d before that line — so when
   the human died first the pipes **froze forever** and the surviving AI faced
   a static scene and never died (the versus round never ended). Fixed: shared
   pipes are advanced **once per step before the per-bird loop**, as long as at
   least one bird is alive.

Both versus bugs were caught by `versus_smoke.py`, which drives the sim
headlessly (no human, no display). Re-run it after touching `sim.py`:

```bat
.\env\python.exe versus_smoke.py
```

Expected output: in "AI vs AI" both birds die at the same step with **equal**
scores; in "AI vs idle human" the human dies early (score 0) and the AI keeps
playing until it too dies → `PASS`.

### Component ablation (first session, 70k-step short runs)
Plain Double+Dueling works (eval ↑). Every single add-on initially *failed*
until the two bugs above were fixed. After the fixes, re-run full Rainbow with
the baseline's LR (1e-4) to confirm each component helps. **Noisy nets are the
most finicky** — if full Rainbow still underperforms, drop noisy
(`--no-noisy`) and rely on epsilon-greedy; that variant is robust.

## 6. Roadmap (prioritized — continue from the top)

1. ~~Pass a smoke test~~ ✅ done.
2. ~~Train baseline (Double+Dueling)~~ ✅ done — 1M steps; best is
   `baseline_long@975k` → **mean 303.67 / max 991**, promoted to `best.pt`.
3. ~~Train Rainbow + ablations~~ ✅ done — every Rainbow variant is clearly
   weaker than plain Double+Dueling on this task.
4. ~~Fix `evaluate()` truncation~~ ✅ done — reported scores are now trustworthy.
5. ~~Fix + verify shared-world versus mode~~ ✅ done (2 real bugs fixed).
6. **Fix training instability, then continue training.** `--resume` is
   implemented and works, but resuming from `best.pt` *collapsed* the policy
   (see §4). Stabilise first (LR schedule / epsilon floor / checkpoint
   selection by eval), then push mean past ~300 and max past ~1000.
7. **Re-measure Rainbow with the corrected eval** — its numbers came from the
   truncated eval, so re-check before finalizing "baseline wins".
8. **Demo human-vs-AI** (`versus.py`) with the best model and record
   qualitative behavior.
9. **Optional next-level upgrades** (only if the ceiling is hit):
   - PPO / SAC on the same vectorized sim for comparison.
   - Self-play / population-based training.
   - Pixel-CNN variant (raw frames) if feature-based plateaus.
   - Hyperparameter sweep via a small script.
7. **Keep this file and `README.md` current** after every meaningful change.
   Commit code + updated docs to git at milestones (`git add -A && git commit`).

## 7. Exact commands to continue right now

```bat
:: 1) confirm the best model still plays well (expect mean ~300, max ~1000)
.\env\python.exe evaluate.py --model-path checkpoints/best.pt --episodes 30

:: 2) re-verify the versus pipeline after ANY change to sim.py
.\env\python.exe versus_smoke.py

:: 3) play against the model: two synchronized panels (LEFT = you, RIGHT = AI).
::    Starts PAUSED — press UP to begin, then UP/SPACE/click to flap.
::    Runs on CPU by default (--device cuda to use the GPU).
.\env\python.exe versus.py --model-path checkpoints/best.pt

:: 4) keep optimizing — resume from the best checkpoint
.\env\python.exe train.py --algo baseline --resume checkpoints/best.pt ^
    --total-timesteps 1000000 --tag baseline_cont

:: 5) or start a fresh run
.\env\python.exe train.py --algo baseline --total-timesteps 1000000 --tag baseline_x
```

## 8. Where the best model came from

`checkpoints/best.pt` is a byte-copy of
`checkpoints/baseline_long/ckpt_step_975000.pt` — produced by:

```bat
.\env\python.exe train.py --algo baseline --lr 1e-4 --total-timesteps 1000000 ^
    --tag baseline_long --save-every 100000
```

Note the eval curve is **very noisy** (a checkpoint can swing from mean 0 to
mean 125 between adjacent evals), so never judge a run by a single eval —
always re-check the top few checkpoints with a full 30-episode greedy eval.
