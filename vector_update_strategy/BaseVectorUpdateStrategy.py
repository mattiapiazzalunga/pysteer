"""Abstract contract for training-time vector update strategies."""

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch
from torch import Tensor


class BaseVectorUpdateStrategy(ABC):
    """Abstract interface for derivation strategies updated from activation batches."""
    @abstractmethod
    def init_layers(
            self,
            layers: Iterable[int],
            hidden_size: int,
    ) -> None:
        """Initialize per-layer storage for a hidden size."""
        pass

    @abstractmethod
    def update(
            self,
            layer_idx: int,
            acts: Tensor,
            correctness_labels: Tensor,
            starts: Tensor,
            tokens_window: int,
            attention_mask: Optional[torch.Tensor],
            group_ids: Optional[torch.Tensor]
    ) -> None:
        """Consume one layer activation batch and update accumulated statistics."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state while keeping configuration."""
        pass
