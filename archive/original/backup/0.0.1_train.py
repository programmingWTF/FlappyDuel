import os
import sys
import math
import time
import random
import argparse
from collections import deque, namedtuple

import numpy as np
import pygame

import torch
import torch.nn as nn
import torch.optim as optim

# ========================= Device selection (CUDA -> DirectML -> CPU) ========================= #

def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda"), "cuda"
    # Try DirectML (for Intel Arc iGPU)
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
    Minimal Flappy Bird-like environment for RL.
    - Discrete action space: 0 = do nothing, 1 = flap
    - State (~18 dims): richer geometric features for the next two pipes (see _get_state)
    - Reward: +1 per step alive, +10 per passed pipe, -100 on crash,
              plus extra shaping reward when approaching/inside the gap (configurable)
    - Rendering via pygame window (always available when render=True)
    """
    def __init__(self, width=288, height=512, render=True, fps=60, seed=0,
                 bird_sprite_path: str | None = None,
                 reward_gap_inside: float = 3.0,
                 reward_gap_near: float = 1.0,
                 near_window: int = 120,
                 render_every: int = 1):
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
        self.pipe_spawn_dist = 180  # distance between pipe groups
        self.pipes = []  # list of dicts: {x, gap_y}
        self.ground_y = int(self.H * 0.87)
        self.sky_y = 0

        # Episode / score
        self.score = 0
        self.ticks = 0
        self.done = False

        # Reward shaping params
        self.reward_gap_inside = float(reward_gap_inside)
        self.reward_gap_near = float(reward_gap_near)
        self.near_window = int(near_window)

        # Pygame
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("Flappy DQN Training")
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
                    # Scale height to 2*radius while keeping aspect ratio
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
        # Spawn initial pipes
        start_x = self.W + 80
        for i in range(3):
            self._spawn_pipe(start_x + i * self.pipe_spawn_dist)
        return self._get_state()

    def _spawn_pipe(self, x):
        margin = 60
        gap_y = self.rng.randint(margin + self.pipe_gap//2, self.ground_y - margin - self.pipe_gap//2)
        self.pipes.append({"x": float(x), "gap_y": float(gap_y), "passed": False})

    def _get_next_pipes(self, k: int = 2):
        """Return the next k upcoming pipes sorted by x (including the current closest)."""
        candidates = [p for p in self.pipes if p["x"] + self.pipe_width >= self.bird_x - self.bird_radius]
        candidates.sort(key=lambda p: p["x"])
        if len(candidates) < k:
            # pad with last known pipe to keep length stable
            pad_with = candidates[-1] if candidates else (self.pipes[-1] if self.pipes else None)
            while len(candidates) < k and pad_with is not None:
                candidates.append(pad_with)
        return candidates[:k]

    def _get_state(self):
        """
        Richer state vector to improve policy precision (slightly slower training):
        Base (4):
            - bird_y_norm, vel_norm, dist_to_ground_norm, dist_to_sky_norm
        Pipe0 (8):
            - dx0, gap_y0, dy_gap0, top0, bottom0, dy_top0, dy_bottom0, t0 (time-to-reach normalized)
        Pipe1 (6):
            - dx1, gap_y1, dy_gap1, top1, bottom1, t1
        Total: 18 dims
        """
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
            # normalize time-to-reach by W/pipe_speed so typical values in ~[0,1]
            denom = (self.W / max(1e-6, self.pipe_speed))
            t_norm = max(0.0, dx_px) / denom
            return dx, gy, dy_gap, top, bottom, dy_top, dy_bottom, t_norm

        p0_feats = pipe_feats(pipes[0]) if pipes and pipes[0] is not None else (0.0,) * 8
        p1_dx, p1_gy, p1_dy_gap, p1_top, p1_bottom, _, _, p1_t = pipe_feats(pipes[1]) if len(pipes) > 1 and pipes[1] is not None else (0.0,) * 8

        state = np.array([
            by, vy, dist_ground, dist_sky,
            # pipe 0 (8)
            p0_feats[0], p0_feats[1], p0_feats[2], p0_feats[3], p0_feats[4], p0_feats[5], p0_feats[6], p0_feats[7],
            # pipe 1 (6) - subset to keep vector compact
            p1_dx, p1_gy, p1_dy_gap, p1_top, p1_bottom, p1_t
        ], dtype=np.float32)
        return state

    def _collides(self):
        # Ground / sky
        if self.bird_y - self.bird_radius <= self.sky_y:
            return True
        if self.bird_y + self.bird_radius >= self.ground_y:
            return True
        # Pipe rectangle intersection
        bx, by, r = self.bird_x, self.bird_y, self.bird_radius
        for p in self.pipes:
            # top pipe rect: (p.x, 0) to (p.x+pipe_width, gap_y - gap/2)
            top_h = int(p["gap_y"] - self.pipe_gap/2)
            bottom_y = int(p["gap_y"] + self.pipe_gap/2)
            rects = [
                (int(p["x"]), 0, self.pipe_width, top_h),
                (int(p["x"]), bottom_y, self.pipe_width, self.ground_y - bottom_y)
            ]
            for rx, ry, rw, rh in rects:
                cx = max(rx, min(bx, rx + rw))
                cy = max(ry, min(by, ry + rh))
                if (bx - cx) ** 2 + (by - cy) ** 2 <= r ** 2:
                    return True
        return False

    def step(self, action):
        # Process action (1 = flap)
        if action == 1:
            self.vel_y = -self.flap_impulse
        # Physics update
        self.vel_y = max(-self.max_vel, min(self.max_vel, self.vel_y + self.gravity))
        self.bird_y = int(self.bird_y + self.vel_y)
        # Move pipes
        for p in self.pipes:
            p["x"] -= self.pipe_speed
        # Remove off-screen and spawn new
        if self.pipes and self.pipes[0]["x"] + self.pipe_width < 0:
            self.pipes.pop(0)
            self._spawn_pipe(self.pipes[-1]["x"] + self.pipe_spawn_dist)
        # Score update when passing a pipe
        reward = 1.0  # survive reward per step
        for p in self.pipes:
            if (not p["passed"]) and (p["x"] + self.pipe_width < self.bird_x - self.bird_radius):
                p["passed"] = True
                self.score += 1
                reward += 10.0
        # Reward shaping: encourage being near the gap center when approaching/inside the pipe
        if self.pipes:
            p0 = self._get_next_pipes(k=1)[0]
            if p0 is not None:
                gap_cy = p0["gap_y"]
                dy = abs(gap_cy - self.bird_y)
                # closeness in [0,1], 1 at center, 0 at gap edges
                closeness = max(0.0, 1.0 - dy / max(1e-6, (self.pipe_gap / 2)))
                # inside horizontal corridor
                inside_x = (self.bird_x + self.bird_radius) >= p0["x"] and (self.bird_x - self.bird_radius) <= (p0["x"] + self.pipe_width)
                if inside_x:
                    reward += self.reward_gap_inside * closeness
                else:
                    # approaching from left within a window
                    dx_ahead = p0["x"] - (self.bird_x - self.bird_radius)
                    if dx_ahead > 0 and dx_ahead < self.near_window:
                        weight = 1.0 - (dx_ahead / self.near_window)  # nearer -> larger
                        reward += self.reward_gap_near * weight * closeness
        # Collision
        self.done = self._collides()
        if self.done:
            reward = -100.0
        self.ticks += 1
        # Render
        if self.render_enabled:
            # Only render every N env ticks to reduce overhead; still pump events to keep window responsive
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
        self.screen.fill((135, 206, 235))  # sky blue
        # Draw pipes
        for p in self.pipes:
            x = int(p["x"])
            top_h = int(p["gap_y"] - self.pipe_gap/2)
            bottom_y = int(p["gap_y"] + self.pipe_gap/2)
            pygame.draw.rect(self.screen, (34, 139, 34), (x, 0, self.pipe_width, top_h))
            pygame.draw.rect(self.screen, (34, 139, 34), (x, bottom_y, self.pipe_width, self.ground_y - bottom_y))
        # Ground
        pygame.draw.rect(self.screen, (222, 184, 135), (0, self.ground_y, self.W, self.H - self.ground_y))
        # Bird (sprite or circle)
        if self.bird_img is not None:
            # Angle tilt based on vertical velocity
            angle = max(-25, min(25, -self.vel_y * 3.0))
            rotated = pygame.transform.rotate(self.bird_img, angle)
            rect = rotated.get_rect(center=(self.bird_x, self.bird_y))
            self.screen.blit(rotated, rect)
        else:
            pygame.draw.circle(self.screen, (255, 215, 0), (self.bird_x, self.bird_y), self.bird_radius)
        # Score
        self._draw_text(f"Score: {self.score}", 18, 10, 10)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_text(self, text, size, x, y):
        font = pygame.font.SysFont("arial", size)
        surf = font.render(text, True, (0, 0, 0))
        self.screen.blit(surf, (x, y))

# ========================= DQN (Dueling) ========================= #

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
        # init
        nn.init.orthogonal_(self.fc1.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.val.weight, gain=1.0)
        nn.init.orthogonal_(self.adv.weight, gain=0.01)
        nn.init.zeros_(self.fc1.bias); nn.init.zeros_(self.fc2.bias); nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.val.bias); nn.init.zeros_(self.adv.bias)

    def forward(self, x):
        h = self.act(self.ln1(self.fc1(x)))
        h = self.act(self.ln2(self.fc2(h)))
        h = self.act(self.fc3(h))
        val = self.val(h)
        adv = self.adv(h)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

# ========================= Replay Buffer ========================= #

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.mem = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.mem.append(Transition(s.astype(np.float32), a, np.float32(r), ns.astype(np.float32), np.float32(d)))

    def sample(self, batch_size: int):
        batch = random.sample(self.mem, batch_size)
        states = np.stack([b.state for b in batch])
        actions = np.array([b.action for b in batch], dtype=np.int64)
        rewards = np.array([b.reward for b in batch], dtype=np.float32)
        next_states = np.stack([b.next_state for b in batch])
        dones = np.array([b.done for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.mem)

# ========================= Training Loop ========================= #

def train(args):
    device, accel = get_device()
    print(f"Device: {device} Accel: {accel}")
    # Optional faster matmul on supported backends
    try:
        if accel == "cuda":
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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

    # Derive state_dim dynamically from a single reset; reuse for episode 1
    state = env.reset()
    state_dim = int(len(state))
    action_dim = 2

    policy = DuelingDQN(state_dim, action_dim, hidden=args.hidden)
    target = DuelingDQN(state_dim, action_dim, hidden=args.hidden)

    # Move to device (supports CUDA/DirectML/CPU)
    policy.to(device)
    target.to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()  # Huber loss

    # Optional AMP (CUDA only)
    use_amp = bool(getattr(args, 'amp', 0)) and accel == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Optional torch.compile (PyTorch 2.x+; guard for unsupported backends)
    if getattr(args, 'compile', 0):
        try:
            policy = torch.compile(policy, mode="reduce-overhead", fullgraph=False)
            print("Policy model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not available/failed: {e}")

    buffer = ReplayBuffer(args.buffer_size, state_dim)

    eps = args.eps_start
    eps_decay = (args.eps_start - args.eps_end) / max(1, args.eps_decay_steps)

    global_step = 0
    best_score = -1
    best_eval = -1.0

    # Auto-resume if checkpoint exists (partial load allowed when shapes changed)
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            load_msg_p = policy.load_state_dict(ckpt['policy'], strict=False)
            load_msg_t = target.load_state_dict(ckpt['target'], strict=False)
            try:
                optimizer.load_state_dict(ckpt['optim'])
            except Exception as e:
                print(f"Optimizer state not loaded: {e}")
            global_step = ckpt.get('step', 0)
            best_score = ckpt.get('best_score', -1)
            print(f"Resumed from {args.checkpoint} at step {global_step}, best_score={best_score}")
            # Report partial load info if any
            if hasattr(load_msg_p, 'missing_keys') and (load_msg_p.missing_keys or load_msg_p.unexpected_keys):
                print(f"Policy missing: {len(load_msg_p.missing_keys)}, unexpected: {len(load_msg_p.unexpected_keys)} (due to changed input size)")
            if hasattr(load_msg_t, 'missing_keys') and (load_msg_t.missing_keys or load_msg_t.unexpected_keys):
                print(f"Target missing: {len(load_msg_t.missing_keys)}, unexpected: {len(load_msg_t.unexpected_keys)}")
        except Exception as e:
            print(f"Found checkpoint but failed to resume: {e}. Starting fresh.")
    elif os.path.exists(args.output):
        # Fallback: try load weights from output model file
        try:
            ck = torch.load(args.output, map_location="cpu")
            if isinstance(ck, dict) and 'policy' in ck:
                policy.load_state_dict(ck['policy'], strict=False)
                print(f"Loaded policy weights from {args.output} (policy key). Starting optimizer fresh.")
            elif isinstance(ck, dict):
                policy.load_state_dict(ck, strict=False)
                print(f"Loaded policy weights from {args.output} (raw state_dict). Starting optimizer fresh.")
            else:
                print(f"Unrecognized format in {args.output}; skipping load.")
        except Exception as e:
            print(f"Failed to load from output model {args.output}: {e}")

    # Prepare a non-render eval env if needed
    eval_env = None
    if getattr(args, 'eval_every', 0) and args.eval_every > 0:
        eval_env = FlappyBirdEnv(
            render=False,
            fps=args.fps,
            seed=args.seed + 999,
            bird_sprite_path=args.bird_sprite,
            reward_gap_inside=args.reward_gap_inside,
            reward_gap_near=args.reward_gap_near,
            near_window=args.near_window,
        )

    for ep in range(1, args.episodes + 1):
        # reuse the already-reset state for episode 1
        if ep == 1 and state is not None:
            pass
        else:
            state = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        while True:
            # One agent step may repeat the same action for multiple env frames (frame-skip)
            ep_steps += 1
            # Epsilon-greedy action
            if random.random() < eps:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    s = torch.from_numpy(state).unsqueeze(0).to(device)
                    q = policy(s)
                    action = int(q.argmax(dim=1).item())

            total_reward = 0.0
            done = False
            next_state = None
            for _ in range(max(1, args.frame_skip)):
                ns, r, d, info = env.step(action)
                total_reward += r
                next_state = ns
                if d:
                    done = True
                    break

            ep_reward += total_reward
            buffer.push(state, action, total_reward, next_state, float(done))
            state = next_state
            global_step += 1

            # Learn
            if len(buffer) >= args.learn_start:
                for _ in range(args.gradient_steps):
                    batch = buffer.sample(args.batch_size)
                    states, actions, rewards, next_states, dones = batch
                    states_t = torch.from_numpy(states).to(device)
                    actions_t = torch.from_numpy(actions).to(device)
                    rewards_t = torch.from_numpy(rewards).to(device)
                    next_states_t = torch.from_numpy(next_states).to(device)
                    dones_t = torch.from_numpy(dones).to(device)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        # Q(s,a)
                        q = policy(states_t).gather(1, actions_t.view(-1, 1)).squeeze(1)
                        # Target Q using Double DQN trick
                        with torch.no_grad():
                            next_actions = policy(next_states_t).argmax(dim=1, keepdim=True)
                            next_q = target(next_states_t).gather(1, next_actions).squeeze(1)
                            target_q = rewards_t + args.gamma * (1.0 - dones_t) * next_q
                        loss = criterion(q, target_q)
                    optimizer.zero_grad(set_to_none=True)
                    if use_amp:
                        scaler.scale(loss).backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
                        optimizer.step()

            # Target network update
            if global_step % args.target_update == 0:
                target.load_state_dict(policy.state_dict())

            # Epsilon decay
            if eps > args.eps_end:
                eps = max(args.eps_end, eps - eps_decay)

            if done or ep_steps >= args.max_steps:
                best_score = max(best_score, info.get('score', 0))
                print(f"Episode {ep}/{args.episodes}  steps={ep_steps}  score={info.get('score', 0)}  reward={ep_reward:.1f}  eps={eps:.3f}")
                break

        # Save checkpoint periodically
        if ep % args.save_every == 0:
            torch.save({
                'policy': policy.state_dict(),
                'target': target.state_dict(),
                'optim': optimizer.state_dict(),
                'step': global_step,
                'best_score': best_score,
            }, args.checkpoint)
            print(f"Saved checkpoint to {args.checkpoint}")

        # Periodic evaluation (greedy, no exploration, no render)
        if eval_env is not None and args.eval_every > 0 and (ep % args.eval_every == 0):
            policy.eval()
            eval_scores = []
            for i in range(max(1, args.eval_episodes)):
                # re-seed for diversity across eval episodes
                try:
                    eval_env.rng.seed(args.seed + 10000 + ep * 7 + i)
                except Exception:
                    pass
                st = eval_env.reset()
                steps = 0
                done_eval = False
                score_eval = 0
                while not done_eval and steps < args.max_steps:
                    with torch.no_grad():
                        s_t = torch.from_numpy(st).unsqueeze(0).to(device)
                        q_t = policy(s_t)
                        a = int(q_t.argmax(dim=1).item())
                    # repeat action for eval frame-skip
                    for _ in range(max(1, args.frame_skip)):
                        st, _, done_eval, info_eval = eval_env.step(a)
                        steps += 1
                        score_eval = info_eval.get('score', score_eval)
                        if done_eval:
                            break
                eval_scores.append(score_eval)
            avg_eval = float(np.mean(eval_scores)) if eval_scores else 0.0
            print(f"[Eval @ ep {ep}] avg_score={avg_eval:.2f} over {len(eval_scores)} episodes")
            # Save best eval checkpoint
            if avg_eval > best_eval:
                best_eval = avg_eval
                best_eval_path = os.path.splitext(args.checkpoint)[0] + '.best_eval.ckpt'
                torch.save({
                    'policy': policy.state_dict(),
                    'target': target.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': global_step,
                    'best_score': best_score,
                    'best_eval': best_eval,
                }, best_eval_path)
                print(f"New best eval {best_eval:.2f}. Saved to {best_eval_path}")
            policy.train()

    # Final save
    torch.save({'policy': policy.state_dict()}, args.output)
    print(f"Training finished. Model saved to {args.output}")

# ========================= Main / CLI ========================= #

def parse_args():
    p = argparse.ArgumentParser(description="DQN training for Flappy Bird with live visualization and DirectML support")
    # Training
    p.add_argument('--episodes', type=int, default=1000000, help='Number of training episodes')
    p.add_argument('--max-steps', type=int, default=5000, help='Max steps per episode')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=100_000)
    p.add_argument('--gradient-steps', type=int, default=1, help='Optimization steps per agent step (>=1)')
    p.add_argument('--learn-start', type=int, default=5000, help='Replay size before learning starts')
    p.add_argument('--target-update', type=int, default=1000, help='Steps between target net sync')
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--frame-skip', type=int, default=4, help='Repeat same action for K frames to speed up training')
    # Epsilon schedule
    p.add_argument('--eps-start', type=float, default=1.0)
    p.add_argument('--eps-end', type=float, default=0.05)
    p.add_argument('--eps-decay-steps', type=int, default=20_000)
    # Env / render
    p.add_argument('--render', type=int, default=1, help='1=render (default), 0=headless')
    p.add_argument('--fps', type=int, default=60)
    p.add_argument('--render-every', type=int, default=4, help='Only render every N env steps to reduce overhead')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bird-sprite', type=str, default=os.path.join('assets', 'bird.png'), help='Path to bird sprite image (PNG with alpha). Default: assets/bird.png')
    p.add_argument('--reward-gap-inside', type=float, default=3.0, help='Extra reward coefficient when inside pipe corridor and close to gap center')
    p.add_argument('--reward-gap-near', type=float, default=1.0, help='Extra reward coefficient when approaching the next pipe within near window')
    p.add_argument('--near-window', type=int, default=120, help='Pixels ahead of the next pipe to start applying near-gap shaping reward')
    # IO
    p.add_argument('--checkpoint', type=str, default='flappy_dqn.ckpt')
    p.add_argument('--output', type=str, default='flappy_dqn.pth')
    p.add_argument('--resume', action='store_true')
    # Save frequency
    p.add_argument('--save-every', type=int, default=20, help='Save checkpoint every N episodes')
    # Evaluation
    p.add_argument('--eval-every', type=int, default=0, help='Run greedy evaluation every N episodes (0=disable)')
    p.add_argument('--eval-episodes', type=int, default=5, help='Number of episodes per evaluation run')
    # Performance toggles
    p.add_argument('--amp', type=int, default=0, help='Enable mixed precision on CUDA for faster training')
    p.add_argument('--compile', type=int, default=0, help='Compile the policy with torch.compile (PyTorch 2.x)')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        train(args)
    finally:
        # Ensure pygame quits cleanly
        if pygame.get_init():
            pygame.quit()
