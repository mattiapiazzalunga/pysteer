"""Training-time COLD-Kernel gradient mediator."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from vector_update_strategy.BaseVectorUpdateStrategy import BaseVectorUpdateStrategy


class ColdKernelGradientMediator(BaseVectorUpdateStrategy):
    """
    pysteer adaptation of unit-kernel COLD-Steer for residual-stream outputs.

    The source method is https://arxiv.org/abs/2603.06495. This implementation
    specializes it to a signed response cross-entropy objective and produces a
    static direction; it does not implement COLD-FD or general kernels.

    It accumulates mean d(objective)/d(hidden) on in-context examples and stores
    the steering vector -mean_gradient. The objective is CE for reference>=thr
    and -CE for reference<thr, so subtracting its gradient increases desired
    responses and suppresses undesired responses.
    """

    def __init__(self, normalize_to_unit_norm: bool = True, eps: float = 1e-12) -> None:
        """Initialize the instance and validate its configuration."""
        self._layers: List[int] = []
        self._hidden: Optional[int] = None
        self._sum_grad: Dict[int, Tensor] = {}
        self._count: Dict[int, float] = {}
        self._correction_vectors: Dict[int, Tensor] = {}
        self._normalize = bool(normalize_to_unit_norm)
        self._eps = float(eps)

    def init_layers(self, layers: Iterable[int], hidden_size: int) -> None:
        """Initialize per-layer storage for a hidden size."""
        self._layers = sorted({int(l) for l in layers})
        self._hidden = int(hidden_size)
        self._sum_grad = {L: torch.zeros(int(hidden_size), dtype=torch.float32) for L in self._layers}
        self._count = {L: 0.0 for L in self._layers}
        self._correction_vectors = {}

    def update(self, layer_idx: int, acts: Tensor, correctness_labels: Tensor, starts: Tensor,
               tokens_window: int, attention_mask: Optional[torch.Tensor], group_ids: Optional[torch.Tensor]) -> None:
        """Consume one layer activation batch and update accumulated statistics."""
        raise RuntimeError("ColdKernelGradientMediator uses update_from_gradients(), not update().")

    @torch.no_grad()
    def update_from_gradients(self, grads: Dict[int, Tensor], response_mask: Tensor) -> None:
        """Consume hidden-state gradients and update accumulated gradient means."""
        if self._hidden is None:
            raise RuntimeError("ColdKernelGradientMediator not initialized.")
        mask = response_mask.to(dtype=torch.bool)
        for L, g in grads.items():
            if int(L) not in self._sum_grad or g is None:
                continue
            gg = torch.nan_to_num(g.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if gg.dim() != 3:
                continue
            m = mask.to(device=gg.device)
            if m.shape[:2] != gg.shape[:2]:
                continue
            valid = m.any(dim=1)
            if not valid.any():
                continue
            m3 = m[valid].unsqueeze(-1).to(dtype=gg.dtype)
            per_sample = (gg[valid] * m3).sum(dim=1) / m3.sum(dim=1).clamp_min(1.0)
            self._sum_grad[int(L)] += per_sample.detach().cpu().sum(dim=0)
            self._count[int(L)] += float(per_sample.size(0))

    def finalize(self) -> None:
        """Finalize accumulated state into reusable artifacts."""
        if self._correction_vectors:
            return
        out: Dict[int, Tensor] = {}
        for L in self._layers:
            c = float(self._count.get(int(L), 0.0))
            if c <= 0.0:
                out[int(L)] = torch.zeros(int(self._hidden), dtype=torch.float32)
                continue
            v = -self._sum_grad[int(L)] / c
            if self._normalize:
                v = F.normalize(v, dim=0, eps=self._eps)
            out[int(L)] = v.detach().cpu().to(torch.float32).contiguous()
        self._correction_vectors = out
        self._sum_grad.clear(); self._count.clear()

    @property
    def correction_vectors(self) -> Dict[int, Tensor]:
        """Return finalized correction vectors."""
        return dict(self._correction_vectors)

    def reset(self) -> None:
        """Reset accumulated state while keeping configuration."""
        if self._hidden is None:
            raise RuntimeError("ColdKernelGradientMediator not initialized.")
        self.init_layers(self._layers, int(self._hidden))
