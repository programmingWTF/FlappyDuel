#!/usr/bin/env python3
"""Training entry point for Flappy Bird RL.

Examples
--------
# Plain Double+Dueling baseline — empirically the strongest on this task
python train.py --algo baseline

# Full Rainbow (all tricks on)
python train.py --algo rainbow

# Override a few hyper-parameters
python train.py --algo baseline --total-timesteps 1000000 --n-envs 16 --tag baseline_big

# Continue training from an existing checkpoint
python train.py --algo baseline --resume checkpoints/best.pt --tag baseline_cont
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Make project root importable when run as `python train.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flappyrl.config import (  # noqa: E402
    EnvConfig, AgentConfig, TrainConfig,
    baseline_config, rainbow_config,
)
from flappyrl.sim import FlappySim  # noqa: E402
from flappyrl.agent import RainbowDQN  # noqa: E402
from flappyrl.logger import Logger  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Flappy Bird RL training")
    p.add_argument("--algo", choices=["baseline", "rainbow"], default="rainbow")
    # Env
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pipe-gap", type=int, default=120)
    p.add_argument("--pipe-speed", type=float, default=2.5)
    # Agent / training
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--n-step", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tag", type=str, default=None)
    # Component overrides (handy for ablations / debugging)
    p.add_argument("--no-noisy", dest="noisy", action="store_false", default=None)
    p.add_argument("--no-per", dest="per", action="store_false", default=None)
    p.add_argument("--no-dist", dest="dist", action="store_false", default=None)
    p.add_argument("--no-nstep", dest="nstep", action="store_false", default=None)
    p.add_argument("--save-every", type=int, default=None,
                   help="env-steps between checkpoints (overrides default)")
    p.add_argument("--eval-every", type=int, default=None,
                   help="env-steps between greedy evaluations")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="episodes per greedy evaluation")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint path to CONTINUE training from")
    p.add_argument("--resume-eps", type=float, default=0.1,
                   help="epsilon to restart at when resuming (default 0.1 — "
                        "starts mostly-greedy so the loaded policy isn't destroyed)")
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--render", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()

    env_cfg = EnvConfig(
        n_envs=args.n_envs,
        seed=args.seed,
        pipe_gap=args.pipe_gap,
        pipe_speed=args.pipe_speed,
    )
    agent_cfg: AgentConfig = rainbow_config() if args.algo == "rainbow" else baseline_config()
    agent_cfg.hidden = args.hidden
    if args.lr is not None:
        agent_cfg.lr = args.lr
    if args.n_step is not None:
        agent_cfg.n_step = args.n_step
    if args.nstep is False:
        agent_cfg.n_step = 1
    if args.batch_size is not None:
        agent_cfg.batch_size = args.batch_size
    if args.noisy is False:
        agent_cfg.noisy = False
    if args.per is False:
        agent_cfg.per = False
    if args.dist is False:
        agent_cfg.distributional = False

    train_cfg = TrainConfig()
    if args.total_timesteps is not None:
        train_cfg.total_timesteps = args.total_timesteps
    if args.save_every is not None:
        train_cfg.save_every = args.save_every
    # Eval cost grows with model strength; allow tuning it from the CLI.
    if args.eval_every is not None:
        train_cfg.eval_every = args.eval_every
    if args.eval_episodes is not None:
        train_cfg.eval_episodes = args.eval_episodes
    if args.tag is not None:
        train_cfg.tag = args.tag
    else:
        train_cfg.tag = args.algo
    train_cfg.render = args.render

    device = (
        torch_device(args.device)
        if args.device != "auto"
        else (torch_device("cuda") if has_cuda() else torch_device("cpu"))
    )
    print(f"[train] device={device}  algo={args.algo}  n_envs={env_cfg.n_envs}")

    sim = FlappySim(env_cfg)
    agent = RainbowDQN(agent_cfg, env_cfg, device)

    # Continue from an existing checkpoint (keeps the learned policy, restarts
    # exploration at a low epsilon so we refine rather than destroy it).
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[train] resume checkpoint not found: {args.resume}")
            sys.exit(1)
        agent.load(args.resume, device)
        agent_cfg.eps_start = args.resume_eps
        agent.eps = args.resume_eps
        print(f"[train] resumed weights from {args.resume} "
              f"(eps restart {args.resume_eps})")

    log = Logger(os.path.join("logs", train_cfg.tag), tag=train_cfg.tag,
                 use_tensorboard=True)

    total = train_cfg.total_timesteps
    global_step = 0
    ep_scores = []
    best_eval = -1.0
    last_log_step = -1e9
    t0 = time.time()

    print(f"[train] starting — target {total} env-steps")
    while global_step < total:
        s = sim.states()
        a = agent.act(s, training=True)
        s2, r, dones, info = sim.step(a)
        agent.store_transition(s, a, r, s2, dones)

        # bookkeeping: episode scores
        ts = info.get("terminal_scores")
        dd = info.get("done")
        if ts is not None and dd is not None:
            for i in range(env_cfg.n_envs):
                if dd[i]:
                    ep_scores.append(int(ts[i]))

        global_step += env_cfg.n_envs
        agent.anneal(global_step, total)

        # gradient updates
        for _ in range(agent_cfg.grad_steps_per_env_step):
            loss = agent.learn()
            if loss is not None and global_step % train_cfg.log_every == 0 and global_step != last_log_step:
                last_log_step = global_step

        # target sync
        if global_step % agent_cfg.target_update_freq == 0:
            agent.sync_target()

        # periodic logging
        if global_step % train_cfg.log_every == 0:
            mean_score = np.mean(ep_scores[-50:]) if ep_scores else 0.0
            max_buf = max(ep_scores[-50:]) if ep_scores else 0
            log.log(
                global_step,
                loss=(loss if loss is not None else float("nan")),
                eps=agent.eps,
                beta=agent.beta,
                buffer_size=len(agent.buffer),
                mean_ep_score_50=mean_score,
                max_ep_score_50=max_buf,
                steps_per_sec=global_step / max(1e-6, (time.time() - t0)),
            )
            print(
                f"step {global_step:>8d} | loss {loss if loss is None else f'{loss:.4f}'} "
                f"| eps {agent.eps:.3f} | buf {len(agent.buffer):>7d} | "
                f"score50 {mean_score:6.2f} (max {max_buf}) | "
                f"{global_step/max(1e-6,(time.time()-t0)):6.1f} st/s"
            )

        # periodic evaluation (greedy)
        if global_step % train_cfg.eval_every == 0 and global_step > 0:
            scores = agent.evaluate(env_cfg, episodes=train_cfg.eval_episodes)
            mean_s, max_s = (np.mean(scores), np.max(scores)) if scores else (0, 0)
            best_eval = max(best_eval, mean_s)
            print(f"  >> EVAL @ {global_step}: mean={mean_s:.2f} max={max_s} "
                  f"(best_mean={best_eval:.2f})")
            log.log(global_step, eval_mean=mean_s, eval_max=max_s, eval_best=best_eval)
            # save eval checkpoint
            agent.save(os.path.join(agent_cfg.checkpoint_dir, train_cfg.tag,
                                    f"ckpt_step_{global_step}.pt"))

        # periodic checkpoint
        if global_step % train_cfg.save_every == 0 and global_step > 0:
            agent.save(os.path.join(agent_cfg.checkpoint_dir, train_cfg.tag,
                                    f"ckpt_step_{global_step}.pt"))

    # final
    agent.save(os.path.join(agent_cfg.checkpoint_dir, train_cfg.tag, "ckpt_final.pt"))
    scores = agent.evaluate(env_cfg, episodes=train_cfg.eval_episodes)
    mean_s, max_s = (np.mean(scores), np.max(scores)) if scores else (0, 0)
    best_eval = max(best_eval, mean_s)
    log.log(global_step, eval_mean=mean_s, eval_max=max_s, eval_best=best_eval)
    print(f"[train] DONE. final eval mean={mean_s:.2f} max={max_s} best_mean={best_eval:.2f}")
    print(f"[train] final checkpoint -> {os.path.join(agent_cfg.checkpoint_dir, train_cfg.tag, 'ckpt_final.pt')}")
    log.close()


def has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def torch_device(name: str):
    import torch
    return torch.device(name)


if __name__ == "__main__":
    main()
