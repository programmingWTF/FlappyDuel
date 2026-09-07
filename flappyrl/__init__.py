"""Flappy Bird RL — modular implementation.

A clean, vectorized, extensible reinforcement-learning toolkit for learning to
play Flappy Bird. Supports a spectrum of algorithms from a plain Double+Dueling
DQN up to full Rainbow (Double + Dueling + Noisy nets + Prioritized Experience
Replay + N-step + Distributional/C51), all toggled from a single config.

Package layout
--------------
- config.py   : dataclass-based configuration (env / agent / training).
- sim.py      : fast vectorized, headless Flappy Bird simulator (numpy).
- render.py   : optional pygame renderer (single + human-vs-AI shared world).
- networks.py : Dueling / Noisy / Distributional networks.
- buffer.py   : uniform + prioritized (SumTree) replay buffers, N-step support.
- agent.py    : RainbowDQN agent tying everything together.
- logger.py   : lightweight CSV + console logger.
- train.py    : training entry point.
"""

from .config import EnvConfig, AgentConfig, TrainConfig
from .sim import FlappySim
from .agent import RainbowDQN
from .logger import Logger

__all__ = [
    "EnvConfig",
    "AgentConfig",
    "TrainConfig",
    "FlappySim",
    "RainbowDQN",
    "Logger",
]
