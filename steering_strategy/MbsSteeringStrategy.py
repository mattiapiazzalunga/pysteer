"""Runtime steering strategy for layer-balanced MBS vectors."""

from typing import Dict

import torch
from torch import Tensor

from steering_strategy.BaseSteeringStrategy import AlphaSpec
from steering_strategy.GeneralSteeringStrategy import GeneralSteeringStrategy


class MbsSteeringStrategy(GeneralSteeringStrategy):

    """Apply MBS vectors with optional per-layer alpha values."""
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
            if int(layer_idx) not in alpha:
                return hidden_states
            alpha = float(alpha[int(layer_idx)])

        return super().steer(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            alpha=alpha,
            mask_tok=mask_tok,
        )