#!/usr/bin/env python3
"""Human-vs-AI mode — two SYNCHRONIZED side-by-side panels.

LEFT = you (red), RIGHT = the AI (gold). Both panels render the *same* shared
world at the same instant, so the two sides are always in lock-step and their
scores are directly comparable. When one side crashes the other keeps flying
until both have failed.

Controls
--------
- UP arrow            : START the round (it begins PAUSED), then flap
- SPACE / UP / click  : flap
- R / RESTART button  : restart at any time (UP or SPACE also start a new
                        round once the current one is over)
- ESC / close window  : quit

Runs on the **CPU by default**, so anyone can play it without a GPU
(pass ``--device cuda`` to use one).

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
from flappyrl.render import SideBySideRenderer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Flappy Bird - Human vs AI (side by side)")
    p.add_argument("--model-path", type=str,
                   default=os.path.join("checkpoints", "best.pt"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pipe-gap", type=int, default=120)
    p.add_argument("--pipe-speed", type=float, default=2.5)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                   help="cpu by default so the game runs anywhere without a GPU")
    p.add_argument("--fps", type=int, default=60)
    return p.parse_args(argv)


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        print("Train first:  python train.py --algo baseline")
        sys.exit(1)

    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Rebuild the agent config from the checkpoint metadata so the architecture
    # always matches the saved weights.
    ckpt = torch.load(args.model_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    agent_cfg = AgentConfig(**meta.get("agent", rainbow_config().__dict__))
    agent_cfg.checkpoint_dir = "checkpoints"

    agent = RainbowDQN(agent_cfg, EnvConfig(n_envs=2), device)
    agent.load(args.model_path, device)
    agent.online_net.eval()
    print(f"[versus] loaded {args.model_path} on {device}")

    env_cfg = EnvConfig(
        seed=args.seed, pipe_gap=args.pipe_gap, pipe_speed=args.pipe_speed,
        n_envs=2, shared_world=True, auto_reset=False, fps=args.fps,
    )
    sim = FlappySim(env_cfg)
    renderer = SideBySideRenderer(
        env_cfg, title="Flappy Bird  -  LEFT: you (red)   |   RIGHT: AI (gold)"
    )

    LIVE = [(220, 20, 60), (255, 215, 0)]      # human = red, AI = gold
    DEAD = (110, 110, 110)
    NAMES = ["YOU", "AI"]

    scores = [0, 0]
    alive = [True, True]
    out_step = [None, None]
    out_score = [None, None]
    step = 0
    paused = True            # the round starts PAUSED; UP begins it
    finished = False
    result = None
    flap_pending = False

    def new_round():
        """Reset everything for a fresh round (also used by the R key)."""
        nonlocal scores, alive, out_step, out_score, step
        nonlocal paused, finished, result, flap_pending
        sim.reset_all()
        scores = [0, 0]
        alive = [True, True]
        out_step = [None, None]
        out_score = [None, None]
        step = 0
        paused = True
        finished = False
        result = None
        flap_pending = False

    new_round()

    print("\n=== Human vs AI (side by side) ===")
    print("LEFT = you (red), RIGHT = AI (gold).")
    print("The round starts PAUSED - press UP to begin.")
    print("Then UP / SPACE / click to flap; R (or the RESTART button) restarts.")
    print("Once both sides are out, R or UP starts a new round.  ESC = quit.\n")

    while True:
        # ------------------------------------------------------------ input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close(); return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    renderer.close(); return
                if event.key == pygame.K_r:
                    new_round()
                    continue
                if event.key in (pygame.K_UP, pygame.K_SPACE):
                    if finished:
                        new_round()          # after a round: play again
                    elif paused:
                        paused = False       # first press starts the round
                    else:
                        flap_pending = True  # later presses flap
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # NOTE: mouse events have no `.key` attribute - never read it here.
                if renderer.restart_rect and renderer.restart_rect.collidepoint(event.pos):
                    new_round()              # clicked the on-screen RESTART button
                elif paused:
                    paused = False
                else:
                    flap_pending = True

        # ------------------------------------------------------------ simulate
        if not paused and not finished:
            step += 1
            human_a = 1 if flap_pending else 0
            flap_pending = False

            states = sim.states()
            ai_a = int(agent.act(states[1:2], training=False)[0]) if alive[1] else 0
            actions = [human_a if alive[0] else 0, ai_a if alive[1] else 0]

            _, _, dones, info = sim.step(actions)
            scores = info["scores"].tolist()

            for i in range(2):
                if dones[i] and alive[i]:
                    alive[i] = False
                    out_step[i] = step
                    out_score[i] = int(scores[i])
                    print(f"  [{NAMES[i]}] OUT at step {step} with score {out_score[i]}")

            if not alive[0] and not alive[1]:
                finished = True
                h, a = out_score[0], out_score[1]
                if h > a:
                    result = f"YOU WIN   {h} - {a}"
                elif a > h:
                    result = f"AI WINS   {a} - {h}"
                else:
                    result = f"DRAW   {h} - {a}"
                print("\n=== Result ===")
                print(f"You passed {h} pipes | AI passed {a} pipes  ->  {result}")

        # ------------------------------------------------------------ draw
        colors = [LIVE[i] if alive[i] else DEAD for i in range(2)]
        if finished:
            banner = result
            status = f"{result}        R or UP = play again    ESC = quit"
        elif paused:
            banner = "PAUSED - press UP to start"
            status = "PAUSED - press UP to start        UP/SPACE/click = flap    R = restart    ESC = quit"
        else:
            banner = None
            status = (f"YOU {scores[0]}{'  OUT' if not alive[0] else ''}   |   "
                      f"AI {scores[1]}{'  OUT' if not alive[1] else ''}"
                      f"        UP/SPACE/click = flap    R = restart    ESC = quit")

        renderer.draw(sim, colors=colors, labels=NAMES, alive=alive,
                      scores=scores, status=status, banner=banner)


if __name__ == "__main__":
    main()
