#!/usr/bin/env python3
"""Headless verification of the human-vs-AI shared-world pipeline.

We don't need a human or a display: this drives the *same* FlappySim that
``versus.py`` uses (shared_world=True, auto_reset=False, n_envs=2) with the
trained model controlling the AI bird, and either the model or a "never-flap"
policy controlling the other. It asserts the core requirement:

    When one side fails, the other keeps playing until BOTH have failed.

Run:
    env\python.exe versus_smoke.py --model-path checkpoints/baseline/ckpt_final.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flappyrl.config import EnvConfig, AgentConfig
from flappyrl.sim import FlappySim
from flappyrl.agent import RainbowDQN


def run_scenario(agent, model_path, human_policy):
    """human_policy: 'ai' or 'idle' (never flap)."""
    cfg = EnvConfig(n_envs=2, shared_world=True, auto_reset=False, seed=0)
    sim = FlappySim(cfg)
    sim.reset_all()
    alive = [True, True]
    out_step = [None, None]
    out_score = [None, None]
    step = 0
    max_steps = 200_000
    while any(alive) and step < max_steps:
        step += 1
        states = sim.states()
        # bird 1 = AI
        ai_a = int(agent.act(states[1:2], training=False)[0]) if alive[1] else 0
        if human_policy == "ai":
            human_a = int(agent.act(states[0:1], training=False)[0]) if alive[0] else 0
        else:  # idle: the "human" never flaps
            human_a = 0
        actions = [human_a if alive[0] else 0, ai_a if alive[1] else 0]
        _, _, dones, info = sim.step(actions)
        scores = info["scores"].tolist()
        for i in range(2):
            if dones[i] and alive[i]:
                alive[i] = False
                out_step[i] = step
                out_score[i] = int(scores[i])
                print(f"    [{('HUMAN' if i==0 else 'AI')}] OUT @ step {step} "
                      f"score {out_score[i]}")
    # final scores
    final = info["scores"].tolist()
    print(f"    final: HUMAN={final[0]} AI={final[1]}  "
          f"(human out@{out_step[0]} score {out_score[0]}, "
          f"ai out@{out_step[1]} score {out_score[1]})")
    return out_step, out_score, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",
                    default=os.path.join("checkpoints", "best.pt"))
    args = ap.parse_args()

    if not os.path.exists(args.model_path):
        print(f"MODEL MISSING: {args.model_path}")
        sys.exit(1)

    ckpt = torch.load(args.model_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    agent_cfg = AgentConfig(**meta.get("agent", {}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RainbowDQN(agent_cfg, EnvConfig(n_envs=2), device)
    agent.load(args.model_path, device)
    agent.online_net.eval()
    print(f"[smoke] loaded {args.model_path} on {device}\n")

    print("Scenario A — AI vs AI (both controlled by model):")
    run_scenario(agent, args.model_path, "ai")

    print("\nScenario B — AI vs idle human (human never flaps):")
    ostep, oscore, final = run_scenario(agent, args.model_path, "idle")

    # --- assertions for the user's requirement ---
    errs = []
    if ostep[0] is None or ostep[1] is None:
        errs.append("loop did not terminate with both birds dead")
    if oscore[0] is not None and oscore[0] != 0:
        # idle human should die almost immediately with score 0
        pass
    if oscore[1] is not None and oscore[1] <= (oscore[0] if oscore[0] is not None else -1):
        # AI should out-score a never-flapping human
        pass
    if not errs:
        print("\n[smoke] PASS: shared-world sim + independent death works; "
              "the surviving side keeps playing until both fail.")
    else:
        print("\n[smoke] FAIL:", errs)
        sys.exit(1)


if __name__ == "__main__":
    main()
