"""Optional pygame renderer for the Flappy Bird simulator.

Rendering is fully decoupled from the simulation: training never imports this
module. It supports drawing a single bird (evaluation) or several birds in the
shared world (human-vs-AI).
"""

from __future__ import annotations

import pygame

from .sim import FlappySim
from .config import EnvConfig


class Renderer:
    def __init__(self, env_cfg: EnvConfig, title: str = "Flappy Bird"):
        pygame.init()
        self.W = env_cfg.width
        self.H = env_cfg.height
        self.pipe_gap = env_cfg.pipe_gap
        self.bird_radius = env_cfg.bird_radius
        self.fps = env_cfg.fps
        self.pipe_width = 52
        self.ground_y = int(self.H * 0.87)
        self.bird_x = int(self.W * 0.2)
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)

    def _pipes(self, sim: FlappySim):
        return sim.pipes if sim.shared else sim.pipes[0]

    def draw(self, sim: FlappySim, bird_colors=None, labels=None, extra: str = ""):
        self.screen.fill((135, 206, 235))
        pipes = self._pipes(sim)
        for p in pipes:
            x = int(p["x"])
            top_h = int(p["gap_y"] - self.pipe_gap / 2)
            bottom_y = int(p["gap_y"] + self.pipe_gap / 2)
            pygame.draw.rect(self.screen, (34, 139, 34), (x, 0, self.pipe_width, top_h))
            pygame.draw.rect(self.screen, (34, 139, 34),
                             (x, bottom_y, self.pipe_width, self.ground_y - bottom_y))
        pygame.draw.rect(self.screen, (222, 184, 135),
                         (0, self.ground_y, self.W, self.H - self.ground_y))

        colors = bird_colors or [(255, 215, 0)]
        labels = labels or ["" for _ in range(sim.n)]
        for i in range(sim.n):
            by = int(sim.bird_y[i])
            color = colors[i % len(colors)]
            pygame.draw.circle(self.screen, color, (self.bird_x, by), self.bird_radius)
            label = labels[i]
            if label:
                txt = self.font.render(label, True, (0, 0, 0))
                self.screen.blit(txt, (self.bird_x + self.bird_radius + 4, by - 8))

        if extra:
            txt = self.font.render(extra, True, (0, 0, 0))
            self.screen.blit(txt, (8, 8))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def pump_events(self) -> bool:
        """Return True if the user requested to quit (ESC / close window)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def close(self):
        pygame.quit()


class SideBySideRenderer:
    """Two synchronized panels in ONE window — LEFT = human, RIGHT = AI.

    pygame only supports a single OS window, so "two windows" is implemented
    as one window split into two aligned panels. Both panels render the *same*
    shared world, so the two sides are always in lock-step and directly
    comparable.

    Each panel shows only its OWN bird (no ghost of the opponent), so the view
    stays clean. A bird that has already failed is drawn dark grey (its corpse
    stays on screen, frozen, until both sides are out).
    """

    DEAD = (110, 110, 110)

    def __init__(self, env_cfg: EnvConfig, gap: int = 24, hud_h: int = 52,
                 title: str = "Flappy Bird - Human vs AI"):
        pygame.init()
        self.W = env_cfg.width
        self.H = env_cfg.height
        self.pipe_gap = env_cfg.pipe_gap
        self.bird_radius = env_cfg.bird_radius
        self.fps = env_cfg.fps
        self.pipe_width = 52
        self.ground_y = int(self.H * 0.87)
        self.bird_x = int(self.W * 0.2)
        self.gap = gap
        self.hud_h = hud_h
        self.win_w = 2 * self.W + gap
        self.win_h = self.H + hud_h
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.big = pygame.font.SysFont("arial", 26, bold=True)
        # Clickable "RESTART" button, drawn in the bottom HUD bar. Kept as an
        # attribute so versus.py can hit-test mouse clicks against it.
        self.restart_rect = None

    def _origin(self, idx: int) -> int:
        return idx * (self.W + self.gap)

    def _draw_panel(self, sim: FlappySim, idx: int, colors, alive):
        ox = self._origin(idx)
        clip = pygame.Rect(ox, 0, self.W, self.H)
        self.screen.set_clip(clip)          # stop pipes bleeding into the next panel
        self.screen.fill((135, 206, 235), clip)

        pipes = sim.pipes if sim.shared else sim.pipes[0]
        for p in pipes:
            x = int(p["x"]) + ox
            top_h = int(p["gap_y"] - self.pipe_gap / 2)
            bottom_y = int(p["gap_y"] + self.pipe_gap / 2)
            pygame.draw.rect(self.screen, (34, 139, 34), (x, 0, self.pipe_width, top_h))
            pygame.draw.rect(self.screen, (34, 139, 34),
                             (x, bottom_y, self.pipe_width, self.ground_y - bottom_y))
        pygame.draw.rect(self.screen, (222, 184, 135),
                         (ox, self.ground_y, self.W, self.H - self.ground_y))

        # Only this panel's own bird is drawn — no ghost of the opponent.
        by = int(sim.bird_y[idx])
        col = colors[idx] if alive[idx] else self.DEAD
        pygame.draw.circle(self.screen, col, (self.bird_x + ox, by), self.bird_radius)
        self.screen.set_clip(None)

    def draw(self, sim: FlappySim, colors, labels, alive, scores, status,
             banner=None):
        for i in range(2):
            self._draw_panel(sim, i, colors, alive)
            ox = self._origin(i)
            name = labels[i] + ("" if alive[i] else "  OUT")
            col = colors[i] if alive[i] else self.DEAD
            txt = self.font.render(f"{name}   {scores[i]}", True, col)
            self.screen.blit(txt, (ox + 8, 8))

        pygame.draw.rect(self.screen, (30, 30, 30),
                         (0, self.H, self.win_w, self.hud_h))

        # Clickable restart button (right side of the HUD bar).
        bw, bh = 132, 30
        self.restart_rect = pygame.Rect(
            self.win_w - bw - 10, self.H + (self.hud_h - bh) // 2, bw, bh)

        # Clip the status text so it can never run under the button.
        self.screen.set_clip(pygame.Rect(0, self.H, self.restart_rect.left - 8,
                                         self.hud_h))
        t = self.font.render(status, True, (255, 255, 255))
        self.screen.blit(t, (10, self.H + 16))
        self.screen.set_clip(None)

        pygame.draw.rect(self.screen, (70, 130, 200), self.restart_rect,
                         border_radius=6)
        bt = self.font.render("RESTART (R)", True, (255, 255, 255))
        self.screen.blit(bt, bt.get_rect(center=self.restart_rect.center))

        if banner:
            t = self.big.render(banner, True, (0, 0, 0))
            r = t.get_rect(center=(self.win_w // 2, self.H // 2))
            pygame.draw.rect(self.screen, (255, 255, 255), r.inflate(24, 14))
            pygame.draw.rect(self.screen, (0, 0, 0), r.inflate(24, 14), 2)
            self.screen.blit(t, r)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
