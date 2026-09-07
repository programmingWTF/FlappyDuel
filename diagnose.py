#!/usr/bin/env python3
"""Diagnostic: train a custom agent for a short time and report eval.

Used to bisect which Rainbow component breaks learning. Not part of the
production pipeline.

    python diagnose.py --noisy --per --n-step 3 --dist --steps 70000 --tag diagX
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from flappyrl.config import EnvConfig, AgentConfig
from flappyrl.sim import FlappySim
from flappyrl.agent import RainbowDQN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy", action="store_true")
    p.add_argument("--per", action="store_true")
    p.add_argument("--dist", action="store_true")
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--double", action="store_true", default=True)
    p.add_argument("--dueling", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=6.25e-5)
    p.add_argument("--steps", type=int, default=70000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--tag", type=str, default="diag")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = EnvConfig(n_envs=a.n_envs, seed=0)
    cfg = AgentConfig(
        double=True, dueling=True,
        noisy=a.noisy, per=a.per, n_step=a.n_step, distributional=a.dist,
        lr=a.lr, batch_size=128, buffer_size=100_000, learning_starts=2000,
        target_update_freq=1000, eps_start=1.0, eps_end=0.05, eps_decay_steps=50000,
        per_alpha=0.6, per_beta_start=0.4, per_beta_end=1.0,
    )
    agent = RainbowDQN(cfg, env_cfg, device)
    sim = FlappySim(env_cfg)

    print(f"[diag {a.tag}] noisy={a.noisy} per={a.per} n_step={a.n_step} "
          f"dist={a.dist} lr={a.lr} device={device}")
    s = sim.states()
    global_step = 0
    scores = []
    last_eval = -1
    while global_step < a.steps:
        act = agent.act(s, training=True)
        s2, r, dones, info = sim.step(act)
        agent.store_transition(s, act, r, s2, dones)
        ts = info.get("terminal_scores"); dd = info.get("done")
        if ts is not None and dd is not None:
            for i in range(a.n_envs):
                if dd[i]:
                    scores.append(int(ts[i]))
        global_step += a.n_envs
        agent.anneal(global_step, a.steps)
        for _ in range(cfg.grad_steps_per_env_step):
            agent.learn()
        if global_step % cfg.target_update_freq == 0:
            agent.sync_target()
        if global_step % 25000 == 0 and global_step > 0:
            ev = agent.evaluate(env_cfg, episodes=15)
            mean_s = np.mean(ev) if ev else 0
            last_eval = mean_s
            print(f"  [{a.tag}] step {global_step} eval_mean={mean_s:.2f} "
                  f"max={max(ev) if ev else 0} buf={len(agent.buffer)}")
        s = s2

    ev = agent.evaluate(env_cfg, episodes=15)
    mean_s = float(np.mean(ev)) if ev else 0.0
    print(f"[diag {a.tag}] FINAL eval_mean={mean_s:.2f} max={max(ev) if ev else 0} "
          f"(mid-run best seen={last_eval:.2f})")


if __name__ == "__main__":
    main()
