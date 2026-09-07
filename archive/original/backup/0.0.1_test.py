import os
import sys
import math
import argparse
import random

import numpy as np
import pygame

import torch
import torch.nn as nn


# ========================= Device selection (CUDA -> DirectML -> CPU) ========================= #

def get_device():
	if torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True
		return torch.device("cuda"), "cuda"
	try:
		import torch_directml

		dml_device = torch_directml.device()
		return dml_device, "dml"
	except Exception:
		pass
	return torch.device("cpu"), "cpu"


# ========================= Flappy Bird Environment (pygame) ========================= #

class FlappyBirdEnv:
	"""
	Same environment as in 0.0.1_train.py, with richer state (~18 dims) and optional reward shaping.
	For testing, we only use its rendering and state stepping.
	"""

	def __init__(
		self,
		width=288,
		height=512,
		render=True,
		fps=60,
		seed=0,
		bird_sprite_path: str | None = None,
		reward_gap_inside: float = 3.0,
		reward_gap_near: float = 1.0,
		near_window: int = 120,
		render_every: int = 1,
	):
		self.W = width
		self.H = height
		self.render_enabled = render
		self.fps = fps
		self.render_every = max(1, int(render_every))
		self.rng = random.Random(seed)

		# Physics
		self.bird_x = int(self.W * 0.2)
		self.bird_y = int(self.H * 0.5)
		self.bird_radius = 12
		self.vel_y = 0.0
		self.gravity = 0.5
		self.flap_impulse = 8.5
		self.max_vel = 10.0

		# Pipes
		self.pipe_width = 52
		self.pipe_gap = 120
		self.pipe_speed = 2.5
		self.pipe_spawn_dist = 180
		self.pipes = []
		self.ground_y = int(self.H * 0.87)
		self.sky_y = 0

		# Episode / score
		self.score = 0
		self.ticks = 0
		self.done = False

		# Reward shaping params (not critical for test)
		self.reward_gap_inside = float(reward_gap_inside)
		self.reward_gap_near = float(reward_gap_near)
		self.near_window = int(near_window)

		# Pygame
		if self.render_enabled:
			pygame.init()
			self.screen = pygame.display.set_mode((self.W, self.H))
			pygame.display.set_caption("Flappy DQN Inference")
			self.clock = pygame.time.Clock()
		else:
			self.screen = None
			self.clock = None

		# Load bird sprite (optional)
		self.bird_img = None
		if bird_sprite_path is not None and isinstance(bird_sprite_path, str):
			try:
				if os.path.exists(bird_sprite_path):
					img = pygame.image.load(bird_sprite_path).convert_alpha()
					target_h = self.bird_radius * 2
					w, h = img.get_size()
					if h != 0:
						scale = target_h / h
						new_size = (max(1, int(w * scale)), target_h)
						img = pygame.transform.smoothscale(img, new_size)
					self.bird_img = img
					print(f"Loaded bird sprite: {bird_sprite_path} -> size {self.bird_img.get_size()}")
				else:
					print(f"Bird sprite not found at: {bird_sprite_path}, fallback to circle drawing")
			except Exception as e:
				print(f"Failed to load bird sprite: {e}. Fallback to circle drawing.")

	def reset(self):
		self.bird_y = int(self.H * 0.5)
		self.vel_y = 0.0
		self.pipes = []
		self.score = 0
		self.ticks = 0
		self.done = False
		start_x = self.W + 80
		for i in range(3):
			self._spawn_pipe(start_x + i * self.pipe_spawn_dist)
		return self._get_state()

	def _spawn_pipe(self, x):
		margin = 60
		gap_y = self.rng.randint(margin + self.pipe_gap // 2, self.ground_y - margin - self.pipe_gap // 2)
		self.pipes.append({"x": float(x), "gap_y": float(gap_y), "passed": False})

	def _get_next_pipes(self, k: int = 2):
		candidates = [p for p in self.pipes if p["x"] + self.pipe_width >= self.bird_x - self.bird_radius]
		candidates.sort(key=lambda p: p["x"])
		if len(candidates) < k:
			pad_with = candidates[-1] if candidates else (self.pipes[-1] if self.pipes else None)
			while len(candidates) < k and pad_with is not None:
				candidates.append(pad_with)
		return candidates[:k]

	def _get_state(self):
		by = self.bird_y / self.H
		vy = self.vel_y / self.max_vel
		dist_ground = (self.ground_y - self.bird_y) / self.H
		dist_sky = (self.bird_y - self.sky_y) / self.H
		pipes = self._get_next_pipes(k=2)

		def pipe_feats(p):
			dx_px = (p["x"] - self.bird_x)
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

		p0_feats = pipe_feats(pipes[0]) if pipes and pipes[0] is not None else (0.0,) * 8
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
				p0_feats[0],
				p0_feats[1],
				p0_feats[2],
				p0_feats[3],
				p0_feats[4],
				p0_feats[5],
				p0_feats[6],
				p0_feats[7],
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

	def _collides(self):
		if self.bird_y - self.bird_radius <= self.sky_y:
			return True
		if self.bird_y + self.bird_radius >= self.ground_y:
			return True
		bx, by, r = self.bird_x, self.bird_y, self.bird_radius
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

	def step(self, action):
		if action == 1:
			self.vel_y = -self.flap_impulse
		self.vel_y = max(-self.max_vel, min(self.max_vel, self.vel_y + self.gravity))
		self.bird_y = int(self.bird_y + self.vel_y)
		for p in self.pipes:
			p["x"] -= self.pipe_speed
		if self.pipes and self.pipes[0]["x"] + self.pipe_width < 0:
			self.pipes.pop(0)
			self._spawn_pipe(self.pipes[-1]["x"] + self.pipe_spawn_dist)
		reward = 0.0  # inference: reward not important
		for p in self.pipes:
			if (not p["passed"]) and (p["x"] + self.pipe_width < self.bird_x - self.bird_radius):
				p["passed"] = True
				self.score += 1
		self.done = self._collides()
		if self.render_enabled:
			# Only render every N env ticks; otherwise just pump events to keep window responsive
			if (self.ticks % self.render_every) == 0:
				self._render()
			else:
				try:
					pygame.event.pump()
				except Exception:
					pass
		return self._get_state(), reward, self.done, {"score": self.score}

	def _render(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit(0)
		self.screen.fill((135, 206, 235))
		for p in self.pipes:
			x = int(p["x"])
			top_h = int(p["gap_y"] - self.pipe_gap / 2)
			bottom_y = int(p["gap_y"] + self.pipe_gap / 2)
			pygame.draw.rect(self.screen, (34, 139, 34), (x, 0, self.pipe_width, top_h))
			pygame.draw.rect(
				self.screen, (34, 139, 34), (x, bottom_y, self.pipe_width, self.ground_y - bottom_y)
			)
		pygame.draw.rect(self.screen, (222, 184, 135), (0, self.ground_y, self.W, self.H - self.ground_y))
		if self.bird_img is not None:
			angle = max(-25, min(25, -self.vel_y * 3.0))
			rotated = pygame.transform.rotate(self.bird_img, angle)
			rect = rotated.get_rect(center=(self.bird_x, self.bird_y))
			self.screen.blit(rotated, rect)
		else:
			pygame.draw.circle(self.screen, (255, 215, 0), (self.bird_x, self.bird_y), self.bird_radius)
		self._draw_text(f"Score: {self.score}", 18, 10, 10)
		pygame.display.flip()
		self.clock.tick(self.fps)

	def _draw_text(self, text, size, x, y):
		font = pygame.font.SysFont("arial", size)
		surf = font.render(text, True, (0, 0, 0))
		self.screen.blit(surf, (x, y))


# ========================= Dueling DQN (same as train) ========================= #

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

	def forward(self, x):
		h = self.act(self.ln1(self.fc1(x)))
		h = self.act(self.ln2(self.fc2(h)))
		h = self.act(self.fc3(h))
		val = self.val(h)
		adv = self.adv(h)
		q = val + adv - adv.mean(dim=1, keepdim=True)
		return q


# ========================= Inference Loop ========================= #

def load_policy(policy: DuelingDQN, model_path: str) -> bool:
	"""Load weights from .ckpt (with 'policy') or .pth (with 'policy' or raw state_dict)."""
	try:
		obj = torch.load(model_path, map_location="cpu")
		if isinstance(obj, dict) and "policy" in obj:
			policy.load_state_dict(obj["policy"], strict=False)
			print(f"Loaded policy from {model_path} (policy key)")
			return True
		elif isinstance(obj, dict):
			policy.load_state_dict(obj, strict=False)
			print(f"Loaded policy from {model_path} (raw state_dict)")
			return True
		else:
			print(f"Unrecognized model format in {model_path}")
			return False
	except Exception as e:
		print(f"Failed to load model from {model_path}: {e}")
		return False


def parse_args():
	p = argparse.ArgumentParser(description="Flappy Bird DQN Inference with live rendering")
	p.add_argument("--model", type=str, default=None, help="Path to model file (.ckpt or .pth)")
	p.add_argument("--episodes", type=int, default=1000000, help="Number of test episodes")
	p.add_argument("--fps", type=int, default=60)
	p.add_argument("--render", type=int, default=1, help="1=render (default), 0=headless")
	p.add_argument("--seed", type=int, default=42)
	p.add_argument(
		"--bird-sprite",
		type=str,
		default=os.path.join("assets", "bird.png"),
		help="Path to bird sprite image",
	)
	p.add_argument("--hidden", type=int, default=256)
	p.add_argument("--reward-gap-inside", type=float, default=3.0)
	p.add_argument("--reward-gap-near", type=float, default=1.0)
	p.add_argument("--near-window", type=int, default=120)
	# Match training option for fair comparison
	p.add_argument("--frame-skip", type=int, default=4, help="Repeat same action for K frames during inference")
	p.add_argument("--render-every", type=int, default=1, help="Only render every N env steps (test)")
	return p.parse_args()


def main():
	args = parse_args()
	device, accel = get_device()
	print(f"Device: {device} Accel: {accel}")

	# Default model path resolution
	model_path = args.model
	if model_path is None:
		# Prefer ckpt if exists, else pth
		ckpt = os.path.join(os.getcwd(), "flappy_dqn.ckpt")
		pth = os.path.join(os.getcwd(), "flappy_dqn.pth")
		if os.path.exists(ckpt):
			model_path = ckpt
		elif os.path.exists(pth):
			model_path = pth
		else:
			print("No model path provided and no flappy_dqn.ckpt/.pth found in current directory.")
			sys.exit(1)

	env = FlappyBirdEnv(
		render=bool(args.render),
		fps=args.fps,
		seed=args.seed,
		bird_sprite_path=args.bird_sprite,
		reward_gap_inside=args.reward_gap_inside,
		reward_gap_near=args.reward_gap_near,
		near_window=args.near_window,
		render_every=args.render_every,
	)

	# Build policy using env state dim
	state0 = env.reset()
	state_dim = int(len(state0))
	action_dim = 2
	policy = DuelingDQN(state_dim, action_dim, hidden=args.hidden).to(device)
	ok = load_policy(policy, model_path)
	if not ok:
		print("Warning: model not loaded, running with randomly initialized policy (for debugging).")
	policy.eval()

	episodes = args.episodes
	for ep in range(1, episodes + 1):
		state = state0 if ep == 1 else env.reset()
		done = False
		ep_steps = 0
		ep_score = 0
		while not done:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit(0)
			with torch.no_grad():
				s = torch.from_numpy(state).unsqueeze(0).to(device)
				q = policy(s)
				action = int(q.argmax(dim=1).item())
			# Repeat action for smoother play and to match training frame-skip
			for _ in range(max(1, args.frame_skip)):
				next_state, _, done, info = env.step(action)
				state = next_state
				ep_steps += 1
				ep_score = info.get("score", ep_score)
				if done:
					break
		print(f"[Episode {ep}/{episodes}] steps={ep_steps} score={ep_score}")

	if pygame.get_init():
		pygame.quit()


if __name__ == "__main__":
	main()

