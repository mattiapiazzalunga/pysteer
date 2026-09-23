"""Runtime additive steering strategy for static correction vectors."""

from typing import Dict, List, Optional

import torch
from torch import Tensor

from steering_strategy.BaseSteeringStrategy import BaseSteeringStrategy, AlphaSpec
from utils.ModelUtils import ModelUtils


class GeneralSteeringStrategy(BaseSteeringStrategy):
    """Apply one additive correction vector per target layer."""
    def __init__(
            self,
            correction_vectors: Dict[int, Tensor],
            *,
            layer_target_norms: Optional[Dict[int, float]] = None,
            normalize_eps: float = 1e-12,
            normalize_to_unit_norm: bool = True,
    ) -> None:
        """Initialize the instance and validate its configuration."""
        self.correction_vectors = correction_vectors
        self.target_layers = sorted(correction_vectors.keys())
        if not self.target_layers:
            raise ValueError("GeneralSteeringStrategy: empty correction_vectors.")
        self._delta_cache: Dict[tuple, Tensor] = {}
        self._delta_cpu: Dict[int, Tensor] = {}
        self.layer_target_norms: Optional[Dict[int, float]] = (
            {int(k): float(v) for k, v in layer_target_norms.items()}
            if layer_target_norms is not None
            else None
        )
        self._normalize_eps: float = float(normalize_eps)
        self._normalize_to_unit_norm = bool(normalize_to_unit_norm)
        eps = self._normalize_eps
        for layer_idx, v in self.correction_vectors.items():
            base = v.detach().to("cpu", torch.float32).contiguous()
            if self._normalize_to_unit_norm:
                base = ModelUtils.safe_normalize(base, dim=0, eps=eps)
            elif self.layer_target_norms is not None:
                tgt = self.layer_target_norms.get(int(layer_idx), None)
                if tgt is not None and float(tgt) > 0.0:
                    base = ModelUtils.safe_normalize(base, dim=0, eps=eps) * float(tgt)
            self._delta_cpu[int(layer_idx)] = base.contiguous()

    @property
    def correction_vectors(self) -> Dict[int, Tensor]:
        """Return finalized correction vectors."""
        return self._correction_vectors

    @correction_vectors.setter
    def correction_vectors(self, value: Dict[int, Tensor]) -> None:
        """Return finalized correction vectors."""
        if not isinstance(value, dict):
            raise ValueError("correction_vectors must be a dict[int, Tensor].")
        for layer_idx, tensor in value.items():
            if not isinstance(layer_idx, int):
                raise ValueError("correction_vectors keys must be layer indexes as ints.")
            if not torch.is_tensor(tensor) or tensor.ndim != 1 or int(tensor.numel()) == 0:
                shape = None if not torch.is_tensor(tensor) else tuple(tensor.shape)
                raise ValueError(
                    f"correction_vectors[{layer_idx}] must be a non-empty 1D tensor, got {shape}."
                )
        self._correction_vectors = value

    @property
    def target_layers(self) -> List[int]:
        """Target layers."""
        return self._target_layers

    @target_layers.setter
    def target_layers(self, value: List[int]) -> None:
        """Set the target_layers value after validation."""
        if not all(isinstance(l, int) for l in value):
            raise ValueError("target_layers must be a list of ints.")
        self._target_layers = value

    def _get_delta_cached(
            self,
            layer_idx: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> Tensor:
        """Return cached or derived delta cached data."""
        dev_idx = -1 if device.index is None else int(device.index)
        key = (int(layer_idx), device.type, dev_idx, dtype)
        v = self._delta_cache.get(key)
        if v is None or (v.device != device) or (v.dtype != dtype):
            base = self._delta_cpu[int(layer_idx)]
            v = base.to(device=device, dtype=dtype)

            self._delta_cache[key] = v
        return v

    @torch.inference_mode()
    def steer(
            self,
            layer_idx: int,
            hidden_states: Tensor,
            alpha: AlphaSpec,
            mask_tok: torch.Tensor,
    ) -> Tensor:
        """Apply this strategy to selected hidden-state tokens."""
        if isinstance(alpha, dict):
            raise TypeError(
                "GeneralSteeringStrategy does not support alpha mappings."
            )

        if not (torch.is_tensor(hidden_states) and torch.is_tensor(mask_tok)):
            return hidden_states
        if mask_tok.numel() == 0 or not mask_tok.any():
            return hidden_states
        if int(layer_idx) not in self._delta_cpu:
            return hidden_states

        orig_dtype = hidden_states.dtype
        device = hidden_states.device
        working_dtype = ModelUtils.steering_work_dtype(orig_dtype)
        alpha_f = ModelUtils.finite_scalar_for_dtype(float(alpha), working_dtype)
        if alpha_f == 0.0:
            return hidden_states
        steered = hidden_states.to(dtype=working_dtype)
        delta = self._get_delta_cached(layer_idx, device=device, dtype=working_dtype)
        mask = mask_tok.to(device=device, dtype=torch.bool)

        if steered.dim() == 3 and mask.dim() == 2 and mask.shape[:2] == steered.shape[:2]:
            _, s, h = steered.shape

            if s == 1:
                m = mask[:, 0]
                if m.any():
                    steered[m, 0, :] = steered[m, 0, :] + alpha_f * delta
            else:
                flat_m = mask.reshape(-1)
                if flat_m.any():
                    steered = steered.contiguous()
                    steered2 = steered.view(-1, h)
                    steered2[flat_m] = steered2[flat_m] + alpha_f * delta
        else:
            expanded_mask = mask.unsqueeze(-1).expand_as(steered)
            expanded_delta = delta.view(1, 1, -1).expand_as(steered)
            steered = torch.where(expanded_mask, steered + alpha_f * expanded_delta, steered)

        ModelUtils.clamp_floating_to_dtype_(steered, orig_dtype)
        return steered.to(orig_dtype)
