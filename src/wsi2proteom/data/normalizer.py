from __future__ import annotations

import numpy as np
import torch


class TargetNormalizer:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, targets: np.ndarray):
        self.mean = np.mean(targets, axis=0, keepdims=True)
        self.std = np.std(targets, axis=0, keepdims=True) + self.epsilon
        return self

    def transform(self, targets: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(targets, np.ndarray):
            return torch.from_numpy((targets - self.mean) / self.std).float()
        m = torch.from_numpy(self.mean).to(targets)
        s = torch.from_numpy(self.std).to(targets)
        return (targets - m) / s

    def inverse_transform(self, predictions: torch.Tensor) -> torch.Tensor:
        m = torch.from_numpy(self.mean).to(predictions)
        s = torch.from_numpy(self.std).to(predictions)
        return predictions * s + m

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"]
        self.std = state["std"]
