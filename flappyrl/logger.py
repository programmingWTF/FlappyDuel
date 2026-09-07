"""Lightweight logger: console + CSV (and optional TensorBoard)."""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, Optional


class Logger:
    def __init__(self, log_dir: str, tag: str = "run", use_tensorboard: bool = False):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{tag}.csv")
        self.f = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.f)
        self.header: Optional[list] = None
        self.use_tb = False
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(log_dir, "tb", tag))
                self.use_tb = True
            except Exception:
                self.use_tb = False
        self.buffer: Dict[str, float] = {}
        self.t0 = time.time()

    def log(self, step: int, **kwargs):
        row = {"step": step, "elapsed_min": (time.time() - self.t0) / 60.0}
        row.update(kwargs)
        if self.header is None:
            self.header = list(row.keys())
            self.writer.writerow(self.header)
        self.writer.writerow([row.get(k, "") for k in self.header])
        self.f.flush()
        if self.use_tb:
            for k, v in kwargs.items():
                try:
                    self.tb.add_scalar(k, float(v), step)
                except Exception:
                    pass

    def close(self):
        self.f.close()
        if self.use_tb:
            self.tb.close()
