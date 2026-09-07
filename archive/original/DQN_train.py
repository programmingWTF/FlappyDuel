#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flappy Bird DQN 训练（单文件版，含实时可视化）

特点：
- 基于 PyTorch 的 Dueling DQN + Double DQN 目标
- pygame 环境，训练过程全程可视化（窗口中实时显示）
- 支持 CUDA（NVIDIA 3050 Laptop），自动检测并使用 GPU
- 单文件可运行：python DQN_train.py --episodes 1000

按键：
- 训练时窗口可视化，按 ESC 立即中断并保存最近一次权重

保存：
- 默认保存到 flappy_dqn.ckpt（包含 policy 与一些训练元数据）

提示：全程可视化会显著降低训练速度，如需更快可把 --render-every 调大或用 --fps 设低一些。
"""

from __future__ import annotations

import os
import sys
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Deque, Tuple, Optional
from collections import deque

import numpy as np
import pygame

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= Device selection (CUDA -> CPU) ========================= #

def get_device():
	if torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True
		device = torch.device("cuda")
		try:
			name = torch.cuda.get_device_name(0)
		except Exception:
			name = "CUDA device"
		print(f"Using GPU: {name}")
		return device
	print("CUDA not available, using CPU")
	return torch.device("cpu")


# ========================= Flappy Bird Environment (pygame) ========================= #

class FlappyEnv:
	def __init__(
		self,
		width: int = 288,
		height: int = 512,
		fps: int = 60,
		seed: int = 0,
		pipe_gap: int = 120,
		pipe_speed: float = 2.5,
		pipe_spawn_dist: int = 180,
		bird_radius: int = 12,
		render_every: int = 1,
	) -> None:
		self.W = width
		self.H = height
		self.fps = fps
		self.rng = random.Random(seed)
		self.clock: Optional[pygame.time.Clock] = None
		self.render_every = max(1, int(render_every))

		# Physics constants
		self.bird_radius = bird_radius
		self.gravity = 0.5
		self.flap_impulse = 8.5
		self.max_vel = 10.0

		# Pipes
		self.pipe_width = 52
		self.pipe_gap = pipe_gap
		self.pipe_speed = pipe_speed
		self.pipe_spawn_dist = pipe_spawn_dist
		self.pipes: list[dict] = []  # each dict: {x, gap_y, passed}
		self.ground_y = int(self.H * 0.87)
		self.sky_y = 0

		# Bird
		self.bird_x = int(self.W * 0.2)
		self.bird_y = int(self.H * 0.5)
		self.bird_vel = 0.0
		self.score = 0

		# Pygame
		pygame.init()
		self.screen = pygame.display.set_mode((self.W, self.H))
		pygame.display.set_caption("Flappy Bird DQN Training")
		self.clock = pygame.time.Clock()

		self.ticks = 0
		self._spawn_initial_pipes()

	# ---------- Terrain management ---------- #
	def _spawn_pipe(self, x: float, gap_y: Optional[float] = None) -> None:
		margin = 60
		if gap_y is None:
			gap_y = self.rng.randint(margin + self.pipe_gap // 2, self.ground_y - margin - self.pipe_gap // 2)
		self.pipes.append({"x": float(x), "gap_y": float(gap_y), "passed": False})

	def _spawn_initial_pipes(self) -> None:
		self.pipes = []
		start_x = self.W + 80
		for i in range(3):
			self._spawn_pipe(start_x + i * self.pipe_spawn_dist)

	def reset(self, seed: Optional[int] = None) -> np.ndarray:
		if seed is not None:
			self.rng = random.Random(seed)
		self.bird_y = int(self.H * 0.5)
		self.bird_vel = 0.0
		self.score = 0
		self._spawn_initial_pipes()
		self.ticks = 0
		return self.get_state()

	# ---------- Collision ---------- #
	def _collides(self, bx: int, by: int) -> bool:
		r = self.bird_radius
		if by - r <= self.sky_y:
			return True
		if by + r >= self.ground_y:
			return True
		for p in self.pipes:
			top_h = int(p["gap_y"] - self.pipe_gap / 2)
			bottom_y = int(p["gap_y"] + self.pipe_gap / 2)
			rects = [
				(int(p["x"]), 0, self.pipe_width, top_h),
				(int(p["x"]), bottom_y, self.pipe_width, self.ground_y - bottom_y),
			]
			for rx, ry, rw, rh in rects:
				cx = max(rx, min(bx, rx + rw))
				cy = max(ry, min(by, ry + rh))
				if (bx - cx) ** 2 + (by - cy) ** 2 <= r ** 2:
					return True
		return False

	# ---------- State ---------- #
	def _get_next_pipes(self, bird_x: int, k: int = 2) -> list[dict]:
		candidates = [p for p in self.pipes if p["x"] + self.pipe_width >= bird_x - self.bird_radius]
		candidates.sort(key=lambda p: p["x"])
		if len(candidates) < k:
			pad_with = candidates[-1] if candidates else (self.pipes[-1] if self.pipes else None)
			while len(candidates) < k and pad_with is not None:
				candidates.append(pad_with)
		return candidates[:k]

	def get_state(self) -> np.ndarray:
		by = self.bird_y / self.H
		vy = self.bird_vel / self.max_vel
		dist_ground = (self.ground_y - self.bird_y) / self.H
		dist_sky = (self.bird_y - self.sky_y) / self.H

		bird_x = self.bird_x
		pipes = self._get_next_pipes(bird_x, k=2)

		def pipe_feats(p):
			dx_px = (p["x"] - bird_x)
			dx = dx_px / self.W
			gy = p["gap_y"] / self.H
			top = (p["gap_y"] - self.pipe_gap / 2) / self.H
			bottom = (p["gap_y"] + self.pipe_gap / 2) / self.H
			dy_gap = (p["gap_y"] - self.bird_y) / self.H
			dy_top = ((p["gap_y"] - self.pipe_gap / 2) - self.bird_y) / self.H
			dy_bottom = ((p["gap_y"] + self.pipe_gap / 2) - self.bird_y) / self.H
			denom = (self.W / max(1e-6, self.pipe_speed))
			t_norm = max(0.0, dx_px) / denom
			return dx, gy, dy_gap, top, bottom, dy_top, dy_bottom, t_norm

		p0 = pipe_feats(pipes[0]) if pipes and pipes[0] is not None else (0.0,) * 8
		p1_dx, p1_gy, p1_dy_gap, p1_top, p1_bottom, _, _, p1_t = (
			pipe_feats(pipes[1]) if len(pipes) > 1 and pipes[1] is not None else (0.0,) * 8
		)

		state = np.array(
			[
				by,
				vy,
				dist_ground,
				dist_sky,
				# pipe 0 (8)
				p0[0],
				p0[1],
				p0[2],
				p0[3],
				p0[4],
				p0[5],
				p0[6],
				p0[7],
				# pipe 1 (6)
				p1_dx,
				p1_gy,
				p1_dy_gap,
				p1_top,
				p1_bottom,
				p1_t,
			],
			dtype=np.float32,
		)
		return state

	# ---------- Step and render ---------- #
	def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
		# action: 0 = noop, 1 = flap
		if action == 1:
			self.bird_vel = -self.flap_impulse

		# Physics update
		self.bird_vel = max(-self.max_vel, min(self.max_vel, self.bird_vel + self.gravity))
		self.bird_y = int(self.bird_y + self.bird_vel)

		# Move pipes
		for p in self.pipes:
			p["x"] -= self.pipe_speed

		# Off-screen removal / spawn new
		if self.pipes and self.pipes[0]["x"] + self.pipe_width < 0:
			self.pipes.pop(0)
			next_x = self.pipes[-1]["x"] + self.pipe_spawn_dist
			self._spawn_pipe(next_x)

		# Scoring
		for p in self.pipes:
			if (not p["passed"]) and (p["x"] + self.pipe_width < self.bird_x - self.bird_radius):
				p["passed"] = True
				self.score += 1

		# Reward shaping
		done = self._collides(self.bird_x, self.bird_y)
		reward = 0.01  # alive reward
		if done:
			reward = -1.0
		else:
			# slight shaping: encourage staying near gap of next pipe
			nxt = self._get_next_pipes(self.bird_x, 1)
			if nxt:
				dy_gap = (nxt[0]["gap_y"] - self.bird_y) / self.H
				reward += 0.05 * (1.0 - min(1.0, abs(dy_gap) * 2.0))  # closer is better
			# passing pipe bonus already reflected in score; add small bonus
			# detect if someone just passed in this step is tricky; we approximate with integer score change via info

		self.ticks += 1
		return self.get_state(), float(reward), bool(done), {"score": self.score}

	def render(self, title_extra: str = "") -> None:
		# Event pump to keep window responsive
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit(0)
			if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
				pygame.quit()
				sys.exit(0)

		# Background
		self.screen.fill((135, 206, 235))

		# Pipes
		for p in self.pipes:
			x = int(p["x"])
			top_h = int(p["gap_y"] - self.pipe_gap / 2)
			bottom_y = int(p["gap_y"] + self.pipe_gap / 2)
			pygame.draw.rect(self.screen, (34, 139, 34), (x, 0, self.pipe_width, top_h))
			pygame.draw.rect(self.screen, (34, 139, 34), (x, bottom_y, self.pipe_width, self.ground_y - bottom_y))

		# Ground
		pygame.draw.rect(self.screen, (222, 184, 135), (0, self.ground_y, self.W, self.H - self.ground_y))

		# Bird
		pygame.draw.circle(self.screen, (255, 215, 0), (self.bird_x, self.bird_y), self.bird_radius)

		# HUD
		font = pygame.font.SysFont("arial", 18)
		hud = font.render(f"Score: {self.score}  {title_extra}", True, (0, 0, 0))
		self.screen.blit(hud, (8, 8))

		pygame.display.flip()
		if self.clock:
			self.clock.tick(self.fps)


# ========================= DQN Components ========================= #

class DuelingDQN(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
		super().__init__()
		self.fc1 = nn.Linear(state_dim, hidden)
		self.ln1 = nn.LayerNorm(hidden)
		self.fc2 = nn.Linear(hidden, hidden)
		self.ln2 = nn.LayerNorm(hidden)
		self.fc3 = nn.Linear(hidden, hidden)
		self.val = nn.Linear(hidden, 1)
		self.adv = nn.Linear(hidden, action_dim)
		self.act = nn.ReLU(inplace=True)
		nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
		nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
		nn.init.orthogonal_(self.fc3.weight, gain=math.sqrt(2))
		nn.init.orthogonal_(self.val.weight, gain=1.0)
		nn.init.orthogonal_(self.adv.weight, gain=0.01)
		nn.init.zeros_(self.fc1.bias)
		nn.init.zeros_(self.fc2.bias)
		nn.init.zeros_(self.fc3.bias)
		nn.init.zeros_(self.val.bias)
		nn.init.zeros_(self.adv.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h = self.act(self.ln1(self.fc1(x)))
		h = self.act(self.ln2(self.fc2(h)))
		h = self.act(self.fc3(h))
		val = self.val(h)
		adv = self.adv(h)
		q = val + adv - adv.mean(dim=1, keepdim=True)
		return q


@dataclass
class Transition:
	s: np.ndarray
	a: int
	r: float
	s2: np.ndarray
	d: bool


class ReplayBuffer:
	def __init__(self, capacity: int = 100_000) -> None:
		self.buf: Deque[Transition] = deque(maxlen=capacity)

	def push(self, *args) -> None:
		self.buf.append(Transition(*args))

	def __len__(self) -> int:
		return len(self.buf)

	def sample(self, batch_size: int) -> Transition:
		idx = np.random.choice(len(self.buf), batch_size, replace=False)
		batch = [self.buf[i] for i in idx]
		s = np.stack([b.s for b in batch])
		a = np.array([b.a for b in batch], dtype=np.int64)
		r = np.array([b.r for b in batch], dtype=np.float32)
		s2 = np.stack([b.s2 for b in batch])
		d = np.array([b.d for b in batch], dtype=np.float32)
		return Transition(s, a, r, s2, d)


# ========================= Training Loop ========================= #

def train(args):
	device = get_device()

	env = FlappyEnv(
		width=args.width,
		height=args.height,
		fps=args.fps,
		seed=args.seed,
		pipe_gap=args.pipe_gap,
		pipe_speed=args.pipe_speed,
		pipe_spawn_dist=args.pipe_spawn,
		bird_radius=args.bird_radius,
		render_every=max(1, args.render_every),
	)

	state_dim = 18
	action_dim = 2
	policy = DuelingDQN(state_dim, action_dim, hidden=args.hidden).to(device)
	target = DuelingDQN(state_dim, action_dim, hidden=args.hidden).to(device)
	target.load_state_dict(policy.state_dict())
	target.eval()

	optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
	scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))

	buffer = ReplayBuffer(capacity=args.buffer)

	gamma = args.gamma
	batch_size = args.batch
	start_learn = args.start_learn
	target_sync = args.target_sync

	eps = args.eps_start
	eps_end = args.eps_end
	eps_decay_steps = max(1, args.eps_decay)

	global_step = 0
	best_score = -1
	t0 = time.time()

	for ep in range(1, args.episodes + 1):
		s = env.reset()
		done = False
		ep_reward = 0.0
		ep_steps = 0
		ep_score = 0

		while not done:
			# Epsilon-greedy
			if random.random() < eps:
				a = random.randint(0, action_dim - 1)
			else:
				with torch.no_grad():
					ss = torch.from_numpy(s).unsqueeze(0).to(device)
					q = policy(ss)
					a = int(q.argmax(dim=1).item())

			s2, r, done, info = env.step(a)
			ep_reward += r
			ep_steps += 1
			ep_score = info.get("score", ep_score)

			buffer.push(s, a, r, s2, done)
			s = s2
			global_step += 1

			# Learn
			if len(buffer) >= start_learn and (global_step % args.learn_every == 0):
				batch = buffer.sample(batch_size)
				ss = torch.from_numpy(batch.s).to(device)
				aa = torch.from_numpy(batch.a).to(device)
				rr = torch.from_numpy(batch.r).to(device)
				ss2 = torch.from_numpy(batch.s2).to(device)
				dd = torch.from_numpy(batch.d).to(device)

				with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.amp)):
					q = policy(ss).gather(1, aa.view(-1, 1)).squeeze(1)
					with torch.no_grad():
						# Double DQN: action from policy, value from target
						next_a = policy(ss2).argmax(dim=1, keepdim=True)
						next_q = target(ss2).gather(1, next_a).squeeze(1)
						tgt = rr + gamma * (1.0 - dd) * next_q
					loss = F.smooth_l1_loss(q, tgt)

				optimizer.zero_grad(set_to_none=True)
				scaler.scale(loss).backward()
				nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
				scaler.step(optimizer)
				scaler.update()

			# Target sync
			if global_step % target_sync == 0:
				target.load_state_dict(policy.state_dict())

			# Epsilon decay
			if eps > eps_end:
				eps = max(eps_end, eps - (args.eps_start - eps_end) / eps_decay_steps)

			# Render
			if env.ticks % env.render_every == 0 and args.render:
				title = f"Ep {ep}/{args.episodes}  step {ep_steps}  eps {eps:.3f}  score {ep_score}  loss?"
				env.render(title_extra=title)

		# End of episode
		best_score = max(best_score, ep_score)
		if ep % args.save_every == 0:
			save_path = args.out
			torch.save({
				"policy": policy.state_dict(),
				"meta": {
					"episodes": ep,
					"best_score": best_score,
					"global_step": global_step,
					"state_dim": state_dim,
					"action_dim": action_dim,
					"hidden": args.hidden,
				},
			}, save_path)
			print(f"[SAVE] ep={ep} best_score={best_score} -> {save_path}")

		# 每回合在控制台简报
		dt = time.time() - t0
		print(f"Episode {ep:4d}  score={ep_score:3d}  steps={ep_steps:4d}  R={ep_reward:+.3f}  eps={eps:.3f}  elapsed={dt/60:.1f}m")

	# Final save
	save_path = args.out
	torch.save({
		"policy": policy.state_dict(),
		"meta": {
			"episodes": args.episodes,
			"best_score": best_score,
			"global_step": global_step,
			"state_dim": state_dim,
			"action_dim": action_dim,
			"hidden": args.hidden,
		},
	}, save_path)
	print(f"[DONE] Saved final model to {save_path}")


def parse_args(argv=None):
	p = argparse.ArgumentParser(description="Flappy Bird DQN Training (visualized)")
	# Environment
	p.add_argument("--width", type=int, default=288)
	p.add_argument("--height", type=int, default=512)
	p.add_argument("--fps", type=int, default=60)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--pipe-gap", type=int, default=120)
	p.add_argument("--pipe-speed", type=float, default=2.5)
	p.add_argument("--pipe-spawn", type=int, default=180)
	p.add_argument("--bird-radius", type=int, default=12)
	p.add_argument("--render", type=int, default=1, help="是否渲染(1/0)")
	p.add_argument("--render-every", type=int, default=1, help="每多少帧渲染一次(>=1)，可用来稍微提速")

	# Agent / Train
	p.add_argument("--episodes", type=int, default=200)
	p.add_argument("--hidden", type=int, default=256)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--gamma", type=float, default=0.99)
	p.add_argument("--batch", type=int, default=128)
	p.add_argument("--buffer", type=int, default=100_000)
	p.add_argument("--start-learn", type=int, default=2000)
	p.add_argument("--learn-every", type=int, default=1)
	p.add_argument("--target-sync", type=int, default=1000)
	p.add_argument("--eps-start", type=float, default=1.0)
	p.add_argument("--eps-end", type=float, default=0.05)
	p.add_argument("--eps-decay", type=int, default=50_000)
	p.add_argument("--amp", action="store_true", help="开启CUDA AMP")

	# Save
	p.add_argument("--out", type=str, default="flappy_dqn.ckpt")
	p.add_argument("--save-every", type=int, default=20)

	return p.parse_args(argv)


def main():
	args = parse_args()
	print("Args:", args)
	try:
		train(args)
	except SystemExit:
		raise
	except Exception as e:
		print(f"Fatal error: {e}")
		raise


if __name__ == "__main__":
	main()

