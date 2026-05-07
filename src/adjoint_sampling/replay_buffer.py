"""Replay buffer  ℬ = {(X_1^(i), ∇g^(i))}  (Algorithm 1, arXiv:2504.11713).

Stores terminal samples and their energy gradients; sampled uniformly for the
inner-loop L_RAM update. Uses a circular batch list capped at max_size samples.
"""

import torch
from torch import Tensor


class ReplayBuffer:

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._x1: list[Tensor] = []
        self._grad_g: list[Tensor] = []
        self._n = 0

    def add(self, x1: Tensor, grad_g: Tensor) -> None:
        """Append a batch of (X_1, ∇g(X_1)) pairs (stored on CPU)."""
        self._x1.append(x1.detach().cpu())
        self._grad_g.append(grad_g.detach().cpu())
        self._n += x1.shape[0]
        # Drop oldest batches until within capacity.
        while len(self._x1) > 1 and self._n - self._x1[0].shape[0] >= self._max_size:
            self._n -= self._x1.pop(0).shape[0]
            self._grad_g.pop(0)

    def sample(self, batch_size: int, device=None) -> tuple[Tensor, Tensor]:
        """Uniformly sample a mini-batch from ℬ."""
        all_x1 = torch.cat(self._x1, dim=0)
        all_grad_g = torch.cat(self._grad_g, dim=0)
        idx = torch.randint(len(all_x1), (batch_size,))
        return all_x1[idx].to(device), all_grad_g[idx].to(device)

    def __len__(self) -> int:
        return self._n

    @property
    def ready(self) -> bool:
        return self._n > 0
