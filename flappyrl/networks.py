"""Network architectures: Dueling / Noisy / Distributional (C51).

A single :class:`RainbowNet` implements every variant through flags so the same
class backs both the plain DQN baseline and the full Rainbow agent.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AgentConfig


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer (Fortunato et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


def _linear(cfg: AgentConfig, in_f: int, out_f: int) -> nn.Module:
    if cfg.noisy:
        return NoisyLinear(in_f, out_f, std_init=cfg.noisy_std)
    return nn.Linear(in_f, out_f)


class RainbowNet(nn.Module):
    """Dueling + (optional) Noisy + (optional) Distributional Q-network.

    Output: if ``distributional`` -> logits of shape ``(B, A, num_atoms)``
            else                  -> Q-values of shape ``(B, A)``
    """

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.num_atoms = cfg.num_atoms
        self.dist = cfg.distributional
        self.dueling = cfg.dueling

        # Atom support (C51)
        if self.dist:
            self.register_buffer(
                "atoms", torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms)
            )
            self.delta = (cfg.v_max - cfg.v_min) / (cfg.num_atoms - 1)

        # Feature extractor: 2 hidden layers
        self.fc1 = _linear(cfg, cfg.state_dim, cfg.hidden)
        self.ln1 = nn.LayerNorm(cfg.hidden)
        self.fc2 = _linear(cfg, cfg.hidden, cfg.hidden)
        self.ln2 = nn.LayerNorm(cfg.hidden)
        self.act = nn.ReLU(inplace=True)

        # Heads
        if self.dueling:
            if self.dist:
                self.val_fc = _linear(cfg, cfg.hidden, cfg.num_atoms)
                self.adv_fc = _linear(cfg, cfg.hidden, cfg.action_dim * cfg.num_atoms)
            else:
                self.val_fc = _linear(cfg, cfg.hidden, 1)
                self.adv_fc = _linear(cfg, cfg.hidden, cfg.action_dim)
        else:
            if self.dist:
                self.out_fc = _linear(cfg, cfg.hidden, cfg.action_dim * cfg.num_atoms)
            else:
                self.out_fc = _linear(cfg, cfg.hidden, cfg.action_dim)

        self._init_orthogonal()

    def _init_orthogonal(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                nn.init.orthogonal_(m.weight_mu, gain=math.sqrt(2))
                nn.init.zeros_(m.bias_mu)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ln1(self.fc1(x)))
        h = self.act(self.ln2(self.fc2(h)))
        if self.dueling:
            if self.dist:
                v = self.val_fc(h).view(-1, 1, self.num_atoms)
                a = self.adv_fc(h).view(-1, self.action_dim, self.num_atoms)
                q = v + a - a.mean(dim=1, keepdim=True)  # (B, A, atoms)
            else:
                v = self.val_fc(h)                        # (B, 1)
                a = self.adv_fc(h)                        # (B, A)
                q = v + a - a.mean(dim=1, keepdim=True)    # (B, A)
        else:
            if self.dist:
                q = self.out_fc(h).view(-1, self.action_dim, self.num_atoms)
            else:
                q = self.out_fc(h)
        return q

    # ---- helpers used by the agent ----
    def distribution(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax over atoms -> probability distributions ``(B, A, atoms)``."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def expected_q(self, x: torch.Tensor) -> torch.Tensor:
        """Expected Q-values ``(B, A)`` from the distribution (or plain Q)."""
        if self.dist:
            p = self.distribution(x)
            return (p * self.atoms.view(1, 1, -1)).sum(dim=-1)
        return self.forward(x)


def build_net(cfg: AgentConfig) -> RainbowNet:
    return RainbowNet(cfg)
