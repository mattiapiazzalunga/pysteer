"""Abstract runtime steering strategy contract."""

from abc import ABC, abstractmethod
from typing import List, Union, Dict

import torch
from torch import Tensor

AlphaSpec = Union[float, Dict[int, float]]


class BaseSteeringStrategy(ABC):
    """Abstract interface implemented by all runtime steering strategies."""
    _target_layers: List[int]

    @property
    @abstractmethod
    def target_layers(self) -> List[int]:
        """Target layers."""
        pass

    @target_layers.setter
    @abstractmethod
    def target_layers(self, layers: List[int]) -> None:
        """Set the target_layers value after validation."""
        pass

    @abstractmethod
    def steer(
            self,
            layer_idx: int,
            hidden_states: Tensor,
            alpha: AlphaSpec,
            mask_tok: torch.Tensor
    ) -> Tensor:
        """Apply this strategy to selected hidden-state tokens."""
        pass
