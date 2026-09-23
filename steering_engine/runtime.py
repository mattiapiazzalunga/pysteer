"""Runtime wrapper factories for executable steering strategies."""

from __future__ import annotations

import inspect
from typing import Any, Dict

from activation_manager.SteeredModelWrapper import SteeredModelWrapper
from steering_engine.components import RuntimeContext


def build_residual_stream_runtime_wrapper(ctx: RuntimeContext) -> SteeredModelWrapper:
    """Default runtime adapter for residual-stream strategies.

    Methods that steer heads, SAE latents, logits, KV cache, or custom controller
    state should provide `MethodDefinition.runtime_wrapper_factory` instead of
    forcing those semantics into `SteeredModelWrapper`.
    """

    kwargs: Dict[str, Any] = {
        "base_model": ctx.model,
        "tokenizer": ctx.tokenizer,
        "strategy": ctx.strategy,
        "alpha": ctx.alpha,
        "apply_from_token": int(ctx.apply_from_token),
        "apply_from_mode": ctx.apply_from_mode,
        "tokens_window": int(ctx.tokens_window),
        "padding_side": str(ctx.padding_side),
    }

    try:
        params = inspect.signature(SteeredModelWrapper).parameters
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )
    except (TypeError, ValueError):
        params = {}
        accepts_var_kwargs = False

    if accepts_var_kwargs or "generation_kwargs" in params:
        kwargs["generation_kwargs"] = dict(ctx.generation_kwargs)

    if accepts_var_kwargs or "dtype" in params:
        kwargs["dtype"] = ctx.dtype

    return SteeredModelWrapper(**kwargs)
