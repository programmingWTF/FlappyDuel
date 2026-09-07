#!/usr/bin/env python3
"""Smoke test: validates the whole pipeline without a long training run.

Runs a few environment steps, performs gradient updates, and checks the
loss is finite and that save/load round-trips. Quick to run after (re)building
the environment.

    python smoke_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from flappyrl.config import EnvConfig, rainbow_config, baseline_config
from flappyrl.sim import FlappySim
from flappyrl.agent import RainbowDQN


def run_one(name, agent_cfg):
    print(f"\n=== smoke: {name} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_cfg = EnvConfig(n_envs=4, seed=0)
    # Small thresholds so the test actually learns within a few steps.
    agent_cfg.learning_starts = 16
    agent_cfg.target_update_freq = 20
    agent_cfg.batch_size = 16
    agent = RainbowDQN(agent_cfg, env_cfg, device)
    sim = FlappySim(env_cfg)

    s = sim.states()
    losses = []
    for t in range(50):
        a = agent.act(s, training=True)
        s2, r, dones, info = sim.step(a)
        agent.store_transition(s, a, r, s2, dones)
        agent.anneal(t * env_cfg.n_envs, 1000)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        s = s2

    agent.sync_target()
    scores = agent.evaluate(env_cfg, episodes=3)
    print(f"  final loss (last): {losses[-1] if losses else 'n/a'}")
    print(f"  eval scores (3 eps): {scores}")
    assert losses and np.isfinite(losses[-1]), "loss is not finite!"

    # save/load round-trip
    path = os.path.join("checkpoints", "_smoke", f"{name}.pt")
    agent.save(path)
    agent2 = RainbowDQN(agent_cfg, env_cfg, device)
    agent2.load(path, device)
    s = sim.states()
    a1 = agent.act(s, training=False)
    a2 = agent2.act(s, training=False)
    assert np.array_equal(a1, a2), "save/load mismatch!"
    print(f"  save/load OK ({path})")

    # cleanup
    try:
        os.remove(path)
        os.rmdir(os.path.dirname(path))
    except OSError:
        pass
    print(f"  [{name}] PASS")


def main():
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    run_one("baseline", baseline_config())
    run_one("rainbow", rainbow_config())
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
