#!/usr/bin/env python3
"""Evaluate a trained Flappy Bird model (greedy, no exploration).

Usage
-----
python evaluate.py --model-path checkpoints/best.pt --episodes 30
python evaluate.py --model-path checkpoints/best.pt --render
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flappyrl.config import EnvConfig, AgentConfig, rainbow_config  # noqa: E402
from flappyrl.sim import FlappySim  # noqa: E402
from flappyrl.agent import RainbowDQN  # noqa: E402
from flappyrl.render import Renderer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a Flappy Bird model")
    p.add_argument("--model-path", type=str,
                   default=os.path.join("checkpoints", "best.pt"))
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--fps", type=int, default=60)
    return p.parse_args(argv)


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}\nTrain first: python train.py --algo rainbow")
        sys.exit(1)

    import torch
    ckpt = torch.load(args.model_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    agent_cfg = AgentConfig(**meta.get("agent", rainbow_config().__dict__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RainbowDQN(agent_cfg, EnvConfig(n_envs=1), device)
    agent.load(args.model_path, device)
    agent.online_net.eval()

    env_cfg = EnvConfig(n_envs=1, seed=args.seed, shared_world=False, auto_reset=True)
    scores = agent.evaluate(env_cfg, episodes=args.episodes)
    mean_s, max_s = (np.mean(scores), np.max(scores)) if scores else (0, 0)
    print(f"EVAL ({len(scores)} episodes): mean={mean_s:.2f}  max={max_s}  "
          f"all={scores}")

    if args.render:
        sim = FlappySim(env_cfg)
        renderer = Renderer(env_cfg, title="Flappy Bird — Evaluation")
        s = sim.reset_all()
        ep = 0
        while ep < args.episodes:
            a = agent.act(s, training=False)
            s, r, dones, info = sim.step(a)
            renderer.draw(sim, extra=f"score {int(info['scores'][0])}  ep {ep+1}/{args.episodes}")
            if renderer.pump_events():
                break
            if dones[0]:
                ep += 1
        renderer.close()


if __name__ == "__main__":
    main()
