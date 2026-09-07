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
