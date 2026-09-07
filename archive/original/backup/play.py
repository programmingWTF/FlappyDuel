import os
import sys
import math
import random
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pygame

import torch
import torch.nn as nn
from copy import deepcopy


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


# ========================= Dueling DQN (same as trainer/test) ========================= #

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


def load_policy(policy: DuelingDQN, model_path: str) -> bool:
	"""Load weights from .ckpt (with 'policy') or .pth (with 'policy' or raw state_dict).
	Prefer safe loading with weights_only=True when available, fallback otherwise.
	"""
	try:
		try:
			# PyTorch 2.4+: safe tensors-only load
			obj = torch.load(model_path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
		except TypeError:
			# Older PyTorch versions don't support weights_only
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


# ========================= Two-player shared-terrain environment ========================= #

class VersusFlappy:
	"""
	两个并排的 Flappy 面板，但各自有独立且完全相同的一组地形（管道）并保持锁步同步。
	左侧=模型；右侧=玩家。每一轮通过相同的随机种子生成，确保公平。
	"""

	def __init__(
		self,
		width=288,
		height=512,
		fps=60,
		seed=0,
		pipe_gap=120,
		pipe_speed=2.5,
		pipe_spawn_dist=180,
		render_every=1,
		bird_radius=12,
		bird_sprite_path: str | None = None,
	):
		self.W = width
		self.H = height
		self.fps = fps
		self.rng = random.Random(seed)
		self.clock = None
		self.render_every = max(1, int(render_every))

		# Physics constants (same as trainer)
		self.bird_radius = bird_radius
		self.gravity = 0.5
		self.flap_impulse = 8.5
		self.max_vel = 10.0

		# Pipes (two identical terrains, one per side)
		self.pipe_width = 52
		self.pipe_gap = pipe_gap
		self.pipe_speed = pipe_speed
		self.pipe_spawn_dist = pipe_spawn_dist
		# self.pipes[0] -> model's terrain; self.pipes[1] -> human's terrain
		self.pipes = [[], []]  # each item: dict {x, gap_y, passed}
		self.ground_y = int(self.H * 0.87)
		self.sky_y = 0

		# Players (0=model on left, 1=human on right)
		self.bird_x_panel = int(self.W * 0.2)
		self.birds = [
			{"y": int(self.H * 0.5), "vel": 0.0, "score": 0, "done": False},
			{"y": int(self.H * 0.5), "vel": 0.0, "score": 0, "done": False},
		]

		# Pygame setup
		pygame.init()
		self.screen = pygame.display.set_mode((self.W * 2, self.H))
		pygame.display.set_caption("Flappy Bird: Model vs You")
		self.clock = pygame.time.Clock()

		# Two independent panel surfaces (two separate game screens)
		self.panels = [
			pygame.Surface((self.W, self.H)).convert_alpha(),
			pygame.Surface((self.W, self.H)).convert_alpha(),
		]

		# Bird sprite (optional, same on both sides)
		self.bird_img = None
		if bird_sprite_path is not None and isinstance(bird_sprite_path, str) and os.path.exists(bird_sprite_path):
			try:
				img = pygame.image.load(bird_sprite_path).convert_alpha()
				target_h = self.bird_radius * 2
				w, h = img.get_size()
				if h != 0:
					scale = target_h / h
					img = pygame.transform.smoothscale(img, (max(1, int(w * scale)), target_h))
				self.bird_img = img
			except Exception as e:
				print(f"Failed to load bird sprite: {e}")

		self.ticks = 0
		self._spawn_initial_pipes()

	# ---------- Terrain management ---------- #
	def _spawn_pipe_both(self, x, gap_y=None):
		margin = 60
		if gap_y is None:
			gap_y = self.rng.randint(margin + self.pipe_gap // 2, self.ground_y - margin - self.pipe_gap // 2)
		p = {"x": float(x), "gap_y": float(gap_y), "passed": False}
		self.pipes[0].append(deepcopy(p))
		self.pipes[1].append(deepcopy(p))

	def _spawn_initial_pipes(self):
		self.pipes = [[], []]
		start_x = self.W + 80
		for i in range(3):
			self._spawn_pipe_both(start_x + i * self.pipe_spawn_dist)

	def reset(self, seed=None):
		if seed is not None:
			self.rng = random.Random(seed)
		self.birds[0].update({"y": int(self.H * 0.5), "vel": 0.0, "score": 0, "done": False})
		self.birds[1].update({"y": int(self.H * 0.5), "vel": 0.0, "score": 0, "done": False})
		self._spawn_initial_pipes()
		self.ticks = 0

	# ---------- Physics and collision ---------- #
	def _collides(self, bx, by, side: int):
		r = self.bird_radius
		if by - r <= self.sky_y:
			return True
		if by + r >= self.ground_y:
			return True
		# check collision against the specified side's terrain
		for p in self.pipes[side]:
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

	# ---------- State (matches training/test features) ---------- #
	def _get_next_pipes(self, side, bird_x, k=2):
		terrain = self.pipes[side]
		candidates = [p for p in terrain if p["x"] + self.pipe_width >= bird_x - self.bird_radius]
		candidates.sort(key=lambda p: p["x"])
		if len(candidates) < k:
			pad_with = candidates[-1] if candidates else (terrain[-1] if terrain else None)
			while len(candidates) < k and pad_with is not None:
				candidates.append(pad_with)
		return candidates[:k]

	def get_state(self, player_idx: int) -> np.ndarray:
		by = self.birds[player_idx]["y"] / self.H
		vy = self.birds[player_idx]["vel"] / self.max_vel
		dist_ground = (self.ground_y - self.birds[player_idx]["y"]) / self.H
		dist_sky = (self.birds[player_idx]["y"] - self.sky_y) / self.H

		bird_x = self.bird_x_panel
		pipes = self._get_next_pipes(player_idx, bird_x, k=2)

		def pipe_feats(p):
			dx_px = (p["x"] - bird_x)
			dx = dx_px / self.W
			gy = p["gap_y"] / self.H
			top = (p["gap_y"] - self.pipe_gap / 2) / self.H
			bottom = (p["gap_y"] + self.pipe_gap / 2) / self.H
			dy_gap = (p["gap_y"] - self.birds[player_idx]["y"]) / self.H
			dy_top = ((p["gap_y"] - self.pipe_gap / 2) - self.birds[player_idx]["y"]) / self.H
			dy_bottom = ((p["gap_y"] + self.pipe_gap / 2) - self.birds[player_idx]["y"]) / self.H
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
	def step(self, action_model: int, action_human: int):
		# Actions
		if action_model == 1:
			self.birds[0]["vel"] = -self.flap_impulse
		if action_human == 1:
			self.birds[1]["vel"] = -self.flap_impulse

		# Physics update (both)
		for i in (0, 1):
			self.birds[i]["vel"] = max(-self.max_vel, min(self.max_vel, self.birds[i]["vel"] + self.gravity))
			self.birds[i]["y"] = int(self.birds[i]["y"] + self.birds[i]["vel"])

		# Move pipes for both terrains (kept identical by spawning in lockstep)
		for side in (0, 1):
			for p in self.pipes[side]:
				p["x"] -= self.pipe_speed
		# Off-screen removal/spawn (lockstep)
		if self.pipes[0] and self.pipes[0][0]["x"] + self.pipe_width < 0:
			# pop both
			self.pipes[0].pop(0)
			self.pipes[1].pop(0)
			# spawn both with same x and gap
			next_x = self.pipes[0][-1]["x"] + self.pipe_spawn_dist
			self._spawn_pipe_both(next_x)

		# Scoring (per player)
		for i in (0, 1):
			bx = self.bird_x_panel
			for p in self.pipes[i]:
				if (not p["passed"]) and (p["x"] + self.pipe_width < bx - self.bird_radius):
					p["passed"] = True
					self.birds[i]["score"] += 1

		# Collisions (per player)
		for i in (0, 1):
			bx = self.bird_x_panel
			by = self.birds[i]["y"]
			# collision against that side's terrain
			self.birds[i]["done"] = self._collides(bx, by, side=i)

		self.ticks += 1

	def render(self):
		# Keep window responsive but don't consume key events here
		pygame.event.pump()

		# Draw each panel independently, then blit onto the main window side-by-side
		bg_left = (135, 206, 235)
		bg_right = (132, 202, 232)  # slightly different to emphasize two independent screens
		panel_bgs = [bg_left, bg_right]
		pipe_color = (34, 139, 34)
		ground_color = (222, 184, 135)
		colors = [(255, 215, 0), (0, 200, 200)]  # model: gold, human: cyan

		for side in (0, 1):
			panel = self.panels[side]
			panel.fill(panel_bgs[side])

			# Pipes for this side only
			for p in self.pipes[side]:
				x = int(p["x"])  # local to panel
				top_h = int(p["gap_y"] - self.pipe_gap / 2)
				bottom_y = int(p["gap_y"] + self.pipe_gap / 2)
				pygame.draw.rect(panel, pipe_color, (x, 0, self.pipe_width, top_h))
				pygame.draw.rect(panel, pipe_color, (x, bottom_y, self.pipe_width, self.ground_y - bottom_y))

			# Ground for this side
			pygame.draw.rect(panel, ground_color, (0, self.ground_y, self.W, self.H - self.ground_y))

			# Bird for this side
			xb = self.bird_x_panel
			yb = self.birds[side]["y"]
			if self.bird_img is not None:
				angle = max(-25, min(25, -self.birds[side]["vel"] * 3.0))
				rotated = pygame.transform.rotate(self.bird_img, angle)
				rect = rotated.get_rect(center=(xb, yb))
				panel.blit(rotated, rect)
			else:
				pygame.draw.circle(panel, colors[side], (xb, yb), self.bird_radius)

			# Side score
			font = pygame.font.SysFont("arial", 20)
			label = "Model" if side == 0 else "You"
			text = font.render(f"{label}: {self.birds[side]['score']}", True, (0, 0, 0))
			panel.blit(text, (10, 10))

		# Compose to screen
		self.screen.blit(self.panels[0], (0, 0))
		self.screen.blit(self.panels[1], (self.W, 0))

		# Divider line and thin borders to accentuate two screens
		pygame.draw.line(self.screen, (0, 0, 0), (self.W, 0), (self.W, self.H), 2)
		pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.W, self.H), 1)
		pygame.draw.rect(self.screen, (0, 0, 0), (self.W, 0, self.W, self.H), 1)

		pygame.display.flip()
		self.clock.tick(self.fps)

	def _draw_text(self, text, size, x, y):
		font = pygame.font.SysFont("arial", size)
		surf = font.render(text, True, (0, 0, 0))
		self.screen.blit(surf, (x, y))

	def show_countdown(self, seconds: int, wins_model: int, wins_human: int):
		"""Show an inter-round countdown overlay; keep pumping events for responsiveness."""
		for sec in range(seconds, 0, -1):
			# dim background
			overlay = pygame.Surface((self.W * 2, self.H), pygame.SRCALPHA)
			overlay.fill((0, 0, 0, 120))
			self.screen.blit(overlay, (0, 0))
			# big number
			font_big = pygame.font.SysFont("arial", 64)
			text = font_big.render(str(sec), True, (255, 255, 255))
			rect = text.get_rect(center=(self.W, int(self.H * 0.45)))
			self.screen.blit(text, rect)
			# scoreboard
			font_mid = pygame.font.SysFont("arial", 28)
			sb = font_mid.render(f"Wins  Model {wins_model} : {wins_human} You", True, (255, 255, 255))
			sb_rect = sb.get_rect(center=(self.W, int(self.H * 0.62)))
			self.screen.blit(sb, sb_rect)
			pygame.display.flip()
			# pump and wait ~1s
			start = pygame.time.get_ticks()
			while pygame.time.get_ticks() - start < 1000:
				for event in pygame.event.get():
					if event.type == pygame.QUIT:
						pygame.quit()
						sys.exit(0)
				pygame.time.delay(10)


# ========================= Main: Model vs Human ========================= #

def main():
	# Choose model via dialog
	tk_root = tk.Tk()
	tk_root.withdraw()
	model_path = filedialog.askopenfilename(
		title="Select model file (.ckpt or .pth)",
		filetypes=[("Model files", "*.ckpt *.pth"), ("All files", "*.*")],
		initialdir=os.getcwd(),
	)
	tk_root.destroy()
	if not model_path:
		print("No model selected.")
		return

	device, accel = get_device()
	print(f"Device: {device} Accel: {accel}")

	# Build env and policy
	env = VersusFlappy(
		width=288,
		height=512,
		fps=60,
		seed=0,
		render_every=1,
		bird_sprite_path=os.path.join("assets", "bird.png"),
	)

	# State dim from training (~18)
	state_dim = 18
	action_dim = 2
	policy = DuelingDQN(state_dim, action_dim, hidden=256).to(device)
	ok = load_policy(policy, model_path)
	if not ok:
		print("Warning: model not loaded; playing with random policy.")
	policy.eval()

	round_seed = 0
	human_flap = 0
	wins_model = 0
	wins_human = 0

	# Render initial frame so user sees the game immediately
	env.render()

	while True:
		# Handle input
		human_flap = 0
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit(0)
			if event.type == pygame.KEYDOWN:
				if event.key in (pygame.K_ESCAPE,):
					pygame.quit()
					sys.exit(0)
				if event.key in (pygame.K_SPACE, pygame.K_UP):
					human_flap = 1
				if event.key in (pygame.K_r,):
					round_seed += 1
					env.reset(seed=round_seed)

		# Model action (greedy)
		with torch.no_grad():
			s = torch.from_numpy(env.get_state(0)).unsqueeze(0).to(device)
			q = policy(s)
			a_model = int(q.argmax(dim=1).item())

		# Step both players with shared terrain
		env.step(a_model, human_flap)

		# If both ended, auto-restart with new terrain
		if env.birds[0]["done"] and env.birds[1]["done"]:
			round_seed += 1
			env.reset(seed=round_seed)


		# Check end-of-round and handle scoreboard/countdown
		if env.birds[0]["done"] or env.birds[1]["done"]:
			# decide winner
			if env.birds[0]["done"] and not env.birds[1]["done"]:
				wins_human += 1
			elif env.birds[1]["done"] and not env.birds[0]["done"]:
				wins_model += 1
			# countdown
			# First, render once to show final state + overlay scoreboard
			env.render()
			env.show_countdown(3, wins_model=wins_model, wins_human=wins_human)
			# reset next round
			round_seed += 1
			env.reset(seed=round_seed)
			continue

		# Render (throttled inside)
		if env.ticks % env.render_every == 0:
			env.render()
			# draw inter-round scoreboard persistent overlay at top center
			font_mid = pygame.font.SysFont("arial", 20)
			sb = font_mid.render(f"Wins  Model {wins_model} : {wins_human} You", True, (0, 0, 0))
			env.screen.blit(sb, sb.get_rect(center=(env.W, 22)))
			pygame.display.flip()
		else:
			try:
				pygame.event.pump()
			except Exception:
				pass


if __name__ == "__main__":
	main()

