"""Training-time mediator for angular steering planes."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.ModelUtils import ModelUtils
from vector_update_strategy.BaseVectorUpdateStrategy import BaseVectorUpdateStrategy

logger = logging.getLogger(__name__)


class AngularVectorMediator(BaseVectorUpdateStrategy):
    """
    Learn pysteer's Angular Steering plane adaptation from contrastive data.

    The source method is https://arxiv.org/abs/2510.26243. This implementation
    uses the same prompt/response/reference table used by CMD/CPCA and derives
    one shared plane across the requested transformer-block residual outputs.

    Candidate direction per layer: mean(correct) - mean(incorrect).
    Feature direction: candidate with the highest mean cosine similarity to
    other candidates. Second axis: first PC of candidate directions, then
    orthogonalized against the feature direction.
    """

    def __init__(
        self,
        correctness_threshold: float = 0.5,
        incorrectness_threshold: float = 0.5,
        use_last_token_for_response: bool = True,
        *,
        layerwise_first_direction: bool = False,
        eps: float = 1e-12,
    ) -> None:
        """Initialize the instance and validate its configuration."""
        self._layers: List[int] = []
        self._hidden: Optional[int] = None
        self._thr_g = float(correctness_threshold)
        self._thr_b = float(incorrectness_threshold)
        self._use_last = bool(use_last_token_for_response)
        self._layerwise_first_direction = bool(layerwise_first_direction)
        self._eps = float(eps)
        self._good: Dict[int, List[Tensor]] = {}
        self._bad: Dict[int, List[Tensor]] = {}
        self._planes: Dict[int, Tuple[Tensor, Tensor]] = {}

    def init_layers(self, layers: Iterable[int], hidden_size: int) -> None:
        """Initialize per-layer storage for a hidden size."""
        self._layers = sorted({int(l) for l in layers})
        self._hidden = int(hidden_size)
        self._good = {L: [] for L in self._layers}
        self._bad = {L: [] for L in self._layers}
        self._planes = {}

    @staticmethod
    def _masked_mean(hs: Tensor, mask: Tensor) -> Tensor:
        """Compute a mask-weighted mean over sequence positions."""
        m = mask.to(dtype=hs.dtype).unsqueeze(-1)
        return (hs * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _last_token_repr(hs: Tensor, mask: Tensor) -> Tensor:
        """Select the last masked token representation for each row."""
        bb, ss, hh = hs.shape
        pos = torch.arange(ss, device=hs.device).unsqueeze(0).expand(bb, -1)
        last_pos = pos.masked_fill(~mask.to(torch.bool), -1).max(dim=1).values
        out = torch.zeros((bb, hh), dtype=hs.dtype, device=hs.device)
        valid = last_pos >= 0
        if valid.any():
            b_idx = torch.arange(bb, device=hs.device)[valid]
            out[valid] = hs[b_idx, last_pos[valid], :]
        return out

    @torch.inference_mode()
    def update(
        self,
        layer_idx: int,
        acts: Tensor,
        correctness_labels: Tensor,
        starts: Tensor,
        tokens_window: int,
        attention_mask: Optional[torch.Tensor],
        group_ids: Optional[torch.Tensor],
    ) -> None:
        """Consume one layer activation batch and update accumulated statistics."""
        if self._hidden is None or int(layer_idx) not in self._good:
            return
        acts = ModelUtils.sanitize_fp16(acts).to(torch.float32)
        acts = ModelUtils.ensure_bsh(acts, int(self._hidden), from_layout="BSH")
        dev = acts.device
        starts = starts.to(dev, dtype=torch.int64)
        labs = correctness_labels.to(dev, dtype=torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        _, ss, _ = acts.shape
        pos = torch.arange(ss, device=dev).unsqueeze(0)
        mask = pos >= starts.unsqueeze(1)
        if tokens_window > 0:
            mask &= pos < (starts + int(tokens_window)).unsqueeze(1)
        if torch.is_tensor(attention_mask) and attention_mask.dim() == 2 and attention_mask.shape == mask.shape:
            mask &= attention_mask.to(device=dev, dtype=torch.bool)
        valid = mask.any(dim=1)
        if not valid.any():
            return

        x = acts[valid]
        m = mask[valid]
        labs = labs[valid]
        reps = self._last_token_repr(x, m) if self._use_last else self._masked_mean(x, m)
        g = reps[labs >= self._thr_g].detach().to("cpu", torch.float32)
        b = reps[labs < self._thr_b].detach().to("cpu", torch.float32)
        self._good[int(layer_idx)].extend([g[i].contiguous() for i in range(int(g.size(0)))])
        self._bad[int(layer_idx)].extend([b[i].contiguous() for i in range(int(b.size(0)))])

    def finalize(self) -> None:
        """Finalize accumulated state into reusable artifacts."""
        if self._planes:
            return
        if self._hidden is None:
            raise RuntimeError("AngularVectorMediator not initialized.")
        candidates: Dict[int, Tensor] = {}
        for L in self._layers:
            good = self._good.get(int(L), [])
            bad = self._bad.get(int(L), [])
            if not good or not bad:
                logger.warning("AngularVectorMediator: missing class at layer %d; using zero candidate.", int(L))
                candidates[int(L)] = torch.zeros(int(self._hidden), dtype=torch.float32)
                continue
            mu_g = torch.stack(good, 0).mean(0)
            mu_b = torch.stack(bad, 0).mean(0)
            candidates[int(L)] = (mu_g - mu_b).to(torch.float32)

        mat = torch.stack([candidates[L] for L in self._layers], 0)
        mat_n = F.normalize(mat, dim=-1, eps=self._eps)
        sims = mat_n @ mat_n.t()
        if len(self._layers) > 1:
            mean_sim = (sims.sum(dim=1) - 1.0) / float(len(self._layers) - 1)
            first_global = mat_n[int(torch.argmax(mean_sim).item())]
        else:
            first_global = mat_n[0]

        centered = mat_n - mat_n.mean(0, keepdim=True)
        if int(centered.size(0)) >= 2 and torch.linalg.vector_norm(centered) > self._eps:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            second = vh[0].to(torch.float32)
        else:
            second = mat_n[-1].to(torch.float32)
        second = second - torch.dot(second, first_global) * first_global
        if torch.linalg.vector_norm(second) <= self._eps:
            # deterministic fallback: use the least aligned coordinate axis
            j = int(torch.argmin(torch.abs(first_global)).item())
            second = torch.zeros_like(first_global)
            second[j] = 1.0
            second = second - torch.dot(second, first_global) * first_global
        second = F.normalize(second, dim=0, eps=self._eps)

        for L in self._layers:
            first = F.normalize(candidates[int(L)], dim=0, eps=self._eps) if self._layerwise_first_direction else first_global
            self._planes[int(L)] = (first.detach().cpu().contiguous(), second.detach().cpu().contiguous())

        self._good.clear()
        self._bad.clear()

    def get_angular_params(self) -> Dict[int, Tuple[Tensor, Tensor]]:
        """Return finalized angular steering planes."""
        if not self._planes:
            raise RuntimeError("AngularVectorMediator.finalize() not yet called.")
        return dict(self._planes)

    def reset(self) -> None:
        """Reset accumulated state while keeping configuration."""
        if self._hidden is None:
            raise RuntimeError("AngularVectorMediator not initialized.")
        self.init_layers(self._layers, int(self._hidden))
