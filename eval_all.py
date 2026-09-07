#!/usr/bin/env python3
"""Evaluate every final checkpoint and report greedy scores + config flags.

Used to pick the genuinely best reproducible model. Run:
    env\python.exe eval_all.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flappyrl.config import EnvConfig, AgentConfig
from flappyrl.agent import RainbowDQN

MODELS = [
    ("baseline",            "checkpoints/baseline/ckpt_final.pt"),
    ("baseline_long@500k",  "checkpoints/baseline_long/ckpt_step_500000.pt"),
    ("rainbow",             "checkpoints/rainbow/ckpt_final.pt"),
    ("rainbow_fix",         "checkpoints/rainbow_fix/ckpt_final.pt"),
    ("rainbow_nonoise",     "checkpoints/rainbow_nonoise/ckpt_final.pt"),
    ("rainbow_pn",          "checkpoints/rainbow_pn/ckpt_final.pt"),
]

EPISODES = 30


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  episodes={EPISODES}\n")
    print(f"{'model':<22} {'D':>1}{'Du':>2}{'N':>1}{'P':>1}{'n':>1}{'C':>1} "
          f"{'lr':>9} {'mean':>7} {'max':>5}  scores")
    print("-" * 120)
    for name, path in MODELS:
        if not os.path.exists(path):
            print(f"{name:<22} MISSING {path}")
            continue
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt.get("meta", {})
        cfg = AgentConfig(**meta.get("agent", {}))
        agent = RainbowDQN(cfg, EnvConfig(n_envs=1), device)
        agent.load(path, device)
        agent.online_net.eval()
        scores = agent.evaluate(EnvConfig(n_envs=1), episodes=EPISODES)
        mean = float(np.mean(scores)) if scores else 0.0
        mx = int(np.max(scores)) if scores else 0
        flags = (f"{int(cfg.double)}{int(cfg.dueling)}{int(cfg.noisy)}"
                 f"{int(cfg.per)}{cfg.n_step}{int(cfg.distributional)}")
        print(f"{name:<22} {flags:<8} {cfg.lr:>9.2e} {mean:>7.2f} {mx:>5}  "
              f"{scores}")


if __name__ == "__main__":
    main()
