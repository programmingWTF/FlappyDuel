#!/usr/bin/env python3
"""Human-vs-AI mode.

Loads a trained model and lets a human play *alongside* the AI in the SAME
shared Flappy Bird scene. When one side fails, the other keeps going until
both have failed. The two birds' scores (pipes passed) are compared.

Controls
--------
- SPACE / UP arrow / mouse click : flap the HUMAN bird
- ESC / close window             : quit

Usage
-----
python versus.py --model-path checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flappyrl.config import EnvConfig, AgentConfig, rainbow_config  # noqa: E402
from flappyrl.sim import FlappySim  # noqa: E402
from flappyrl.agent import RainbowDQN  # noqa: E402
from flappyrl.render import Renderer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Human vs AI Flappy Bird")
    p.add_argument("--model-path", type=str,
                   default=os.path.join("checkpoints", "best.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pipe-gap", type=int, default=120)
    p.add_argument("--pipe-speed", type=float, default=2.5)
    return p.parse_args(argv)


def main():
    args = parse_args()

    # Rebuild the agent config from the checkpoint metadata so the architecture
    # always matches the saved weights.
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        print("Train first:  python train.py --algo rainbow")
        sys.exit(1)

    import torch
    ckpt = torch.load(args.model_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    agent_cfg = AgentConfig(**meta.get("agent", rainbow_config().__dict__))
    agent_cfg.checkpoint_dir = "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = RainbowDQN(agent_cfg, EnvConfig(n_envs=2), device)
    agent.load(args.model_path, device)
    agent.online_net.eval()
    print(f"[versus] loaded model from {args.model_path}  ({device})")

    env_cfg = EnvConfig(
        seed=args.seed, pipe_gap=args.pipe_gap, pipe_speed=args.pipe_speed,
        n_envs=2, shared_world=True, auto_reset=False,
    )
    sim = FlappySim(env_cfg)
    renderer = Renderer(env_cfg, title="Flappy Bird — Human vs AI")

    bird_colors = [(220, 20, 60), (255, 215, 0)]        # human=red, AI=gold
    labels = ["HUMAN", "AI"]

    sim.reset_all()
    alive = [True, True]
    out_step = [None, None]
    flap_pending = False

    print("\n=== Human vs AI ===")
    print("You control the RED bird. Flap with SPACE / UP / click.")
    print("When one side dies, the other keeps flying until both are out.\n")

    step = 0
    while True:
        step += 1
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close(); return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE,):
                renderer.close(); return
            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                if event.key in (pygame.K_SPACE, pygame.K_UP) or event.type == pygame.MOUSEBUTTONDOWN:
                    flap_pending = True

        human_a = 1 if flap_pending else 0
        flap_pending = False

        states = sim.states()
        ai_a = int(agent.act(states[1:2], training=False)[0]) if alive[1] else 0
        actions = [human_a if alive[0] else 0, ai_a if alive[1] else 0]

        s2, r, dones, info = sim.step(actions)
        scores = info["scores"].tolist()

        for i in range(2):
            if dones[i] and alive[i]:
                alive[i] = False
                out_step[i] = step
                print(f"  [{ 'HUMAN' if i==0 else 'AI' }] OUT at step {step} "
                      f"with score {scores[i]}")

        hud = (f"HUMAN {scores[0]}{'  OUT' if not alive[0] else ''}   |   "
               f"AI {scores[1]}{'  OUT' if not alive[1] else ''}")
        renderer.draw(sim, bird_colors=bird_colors, labels=labels, extra=hud)

        if not alive[0] and not alive[1]:
            break

    final_h, final_a = scores[0], scores[1]
    print("\n=== Result ===")
    print(f"HUMAN passed {final_h} pipes | AI passed {final_a} pipes")
    if final_h > final_a:
        print("Human wins! 🏆" if False else "Human wins!")
    elif final_a > final_h:
        print("AI wins!")
    else:
        print("Draw!")
    renderer.close()


if __name__ == "__main__":
    main()
