"""Training-time layer-balanced contrastive mean-difference mediator."""

import logging
from typing import Optional

import torch
from torch import Tensor

from vector_update_strategy.CmdVectorMediator import CmdVectorMediator

logger = logging.getLogger(__name__)


class MbsCmdVectorMediator(CmdVectorMediator):

    """Apply pysteer's layer-routed extension of CMD during derivation."""
    def __init__(
            self,
            correctness_threshold: float = 0.5,
            incorrectness_threshold: float = 0.5,
            use_last_token_for_response: bool = True,
            *,
            fail_on_missing_layer_samples: bool = True,
    ):
        """Initialize the instance and validate its configuration."""
        super().__init__(
            correctness_threshold=correctness_threshold,
            incorrectness_threshold=incorrectness_threshold,
            use_last_token_for_response=use_last_token_for_response,
        )
        self._fail_on_missing_layer_samples = bool(fail_on_missing_layer_samples)

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
        if group_ids is None:
            raise ValueError(
                "MbsCmdVectorMediator requires group_ids. "
                "Executor passes df['mbs_layer'] as group_ids."
            )

        if int(layer_idx) not in self.layers:
            return

        if not torch.is_tensor(acts) or acts.dim() != 3:
            raise RuntimeError(
                f"MbsCmdVectorMediator.update: expected acts [B,S,H], got "
                f"{None if not torch.is_tensor(acts) else tuple(acts.shape)}"
            )

        dev = acts.device
        gids = group_ids.to(device=dev, dtype=torch.long)

        if gids.dim() != 1 or int(gids.size(0)) != int(acts.size(0)):
            raise RuntimeError(
                f"MbsCmdVectorMediator.update: group_ids must be [B]. "
                f"got group_ids={tuple(gids.shape)}, acts={tuple(acts.shape)}"
            )

        sel = gids == int(layer_idx)
        if not bool(sel.any().item()):
            return

        attn_sel = None
        if torch.is_tensor(attention_mask):
            am = attention_mask.to(device=dev)
            if am.dim() >= 1 and int(am.size(0)) == int(gids.size(0)):
                attn_sel = am[sel]

        super().update(
            layer_idx=int(layer_idx),
            acts=acts[sel],
            correctness_labels=correctness_labels.to(device=dev)[sel],
            starts=starts.to(device=dev, dtype=torch.int64)[sel],
            tokens_window=tokens_window,
            attention_mask=attn_sel,
            group_ids=None,
        )

    def finalize(self) -> None:
        """Finalize accumulated state into reusable artifacts."""
        if getattr(self, "_correction_vectors", None):
            return

        missing = []
        for L in self.layers:
            cw = float(self.correct_weights.get(int(L), 0.0))
            iw = float(self.incorrect_weights.get(int(L), 0.0))
            if cw <= 0.0 or iw <= 0.0:
                missing.append((int(L), cw, iw))

        if missing and self._fail_on_missing_layer_samples:
            preview = missing[:20]
            more = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
            raise RuntimeError(
                "MBS_CMD finalize failed: every MBS layer needs both positive and negative samples.\n"
                f"Missing or degenerate layers: {preview}{more}\n"
                "Check df['mbs_layer'] and df['reference']. "
                "Each selected layer must receive rows with reference>=correctness_threshold "
                "and rows with reference<incorrectness_threshold."
            )

        if missing:
            logger.warning(
                "MBS_CMD finalize: some layers have missing positive/negative samples: %s",
                missing,
            )

        super().finalize()
