"""Runtime angular steering strategy."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from steering_strategy.BaseSteeringStrategy import BaseSteeringStrategy, AlphaSpec
from utils.ModelUtils import ModelUtils


class AngularSteeringStrategy(BaseSteeringStrategy):
    """
    Apply pysteer's residual-stream adaptation of Adaptive Angular Steering.

    alpha is interpreted as a target angle in degrees. Example: alpha=90.0.
    If adaptive=True, only tokens whose projection on gate_direction is above
    gate_threshold are rotated.
    """

    def __init__(
        self,
        steering_planes: Dict[int, Tuple[Tensor, Tensor]],
        *,
        adaptive: bool = True,
        gate_direction: str = "first",  # "first" or "second"
        gate_threshold: float = 0.0,
        normalize_eps: float = 1e-12,
        cpu_store_dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize the instance and validate its configuration."""
        if not steering_planes:
            raise ValueError("AngularSteeringStrategy: empty steering_planes.")
        self._normalize_eps = float(normalize_eps)
        self._cpu_store_dtype = cpu_store_dtype
        self.adaptive = bool(adaptive)
        self.gate_direction = str(gate_direction).lower()
        if self.gate_direction not in ("first", "second"):
            raise ValueError("gate_direction must be 'first' or 'second'.")
        self.gate_threshold = float(gate_threshold)

        self._planes_cpu: Dict[int, Tuple[Tensor, Tensor]] = {}
        for k, pair in steering_planes.items():
            first, second = pair
            first = first.detach().to("cpu", torch.float32).flatten().contiguous()
            second = second.detach().to("cpu", torch.float32).flatten().contiguous()
            if first.numel() != second.numel() or first.numel() == 0:
                raise ValueError(f"Bad angular plane at layer {k}.")
            b1 = ModelUtils.safe_normalize(first, dim=0, eps=self._normalize_eps)
            b2 = second - torch.dot(second, b1) * b1
            b2 = ModelUtils.safe_normalize(b2, dim=0, eps=self._normalize_eps)
            self._planes_cpu[int(k)] = (
                b1.to(dtype=cpu_store_dtype).contiguous(),
                b2.to(dtype=cpu_store_dtype).contiguous(),
            )

        self.target_layers = sorted(self._planes_cpu.keys())
        self._cache: Dict[tuple[int, str, int], Tuple[Tensor, Tensor]] = {}

    @property
    def target_layers(self) -> List[int]:
        """Target layers."""
        return self._target_layers

    @target_layers.setter
    def target_layers(self, layers: List[int]) -> None:
        """Set the target_layers value after validation."""
        self._target_layers = [int(x) for x in layers]

    def _get_plane(self, layer_idx: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Return cached or derived plane data."""
        dev_idx = -1 if device.index is None else int(device.index)
        key = (int(layer_idx), device.type, dev_idx)
        out = self._cache.get(key)
        if out is None:
            b1, b2 = self._planes_cpu[int(layer_idx)]
            out = (b1.to(device=device, dtype=torch.float32), b2.to(device=device, dtype=torch.float32))
            self._cache[key] = out
        return out

    @torch.inference_mode()
    def steer(self, layer_idx: int, hidden_states: Tensor, alpha: AlphaSpec, mask_tok: torch.Tensor) -> Tensor:
        """Apply this strategy to selected hidden-state tokens."""
        if int(layer_idx) not in self._planes_cpu or not torch.is_tensor(mask_tok) or not mask_tok.any():
            return hidden_states
        if isinstance(alpha, dict):
            if int(layer_idx) not in alpha:
                return hidden_states
            degree = ModelUtils.finite_scalar_for_dtype(float(alpha[int(layer_idx)]), torch.float32)
        else:
            degree = ModelUtils.finite_scalar_for_dtype(float(alpha), torch.float32)

        orig_dtype = hidden_states.dtype
        dev = hidden_states.device
        hs = hidden_states.to(torch.float32).contiguous()
        b1, b2 = self._get_plane(int(layer_idx), dev)

        theta = math.radians(degree)
        rotated_component = math.cos(theta) * b1 + math.sin(theta) * b2

        mask = mask_tok.to(device=dev, dtype=torch.bool)
        if self.adaptive:
            gate = b1 if self.gate_direction == "first" else b2
            mask = mask & ((hs @ gate) > float(self.gate_threshold))
            if not mask.any():
                return hidden_states

        # Project each activation into the 2D plane, preserve the projection norm,
        # and replace only the plane component with the target-angle component.
        c1 = hs @ b1
        c2 = hs @ b2
        px = c1.unsqueeze(-1) * b1 + c2.unsqueeze(-1) * b2
        scale = torch.linalg.vector_norm(px, dim=-1, keepdim=True)
        replacement = scale * rotated_component.view(1, 1, -1)
        delta = replacement - px

        hs[mask] = hs[mask] + delta[mask]
        ModelUtils.clamp_floating_to_dtype_(hs, orig_dtype)
        return hs.to(orig_dtype)
