"""Model inspection and tensor-layout helpers."""

from __future__ import annotations

import math
import re
import weakref
from typing import Callable, List, Optional

import torch
import torch.nn as nn


class ModelUtils:

    """Namespace for model unwrapping, layer discovery, and tensor layout helpers."""
    SUPPORTED_STEERING_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    _TRANSFORMER_LAYER_CACHE = weakref.WeakKeyDictionary()

    @staticmethod
    def validate_steering_dtype(dtype: torch.dtype) -> torch.dtype:
        """Validate the public runtime dtype supported by steering wrappers."""
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be an instance of torch.dtype")
        if dtype not in ModelUtils.SUPPORTED_STEERING_DTYPES:
            names = ", ".join(str(d) for d in ModelUtils.SUPPORTED_STEERING_DTYPES)
            raise ValueError(f"dtype must be one of: {names}")
        return dtype

    @staticmethod
    def steering_work_dtype(dtype: torch.dtype) -> torch.dtype:
        """Return the dtype used for overflow-sensitive steering math."""
        return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

    @staticmethod
    def safe_vector_norm(
            tensor: torch.Tensor,
            *,
            dim: int = -1,
            keepdim: bool = False,
    ) -> torch.Tensor:
        """Compute a float32 norm, falling back to float64 only if it overflows."""
        x = tensor.to(torch.float32)
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)
        if torch.isfinite(norm).all():
            return norm

        norm64 = torch.linalg.vector_norm(x.to(torch.float64), dim=dim, keepdim=keepdim)
        return torch.nan_to_num(norm64, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).to(torch.float32)

    @staticmethod
    def safe_normalize(tensor: torch.Tensor, *, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
        """Normalize in float32, falling back to float64 only if the norm overflows."""
        x = tensor.to(torch.float32)
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
        if torch.isfinite(norm).all():
            return x / norm.clamp_min(float(eps))

        x64 = x.to(torch.float64)
        norm64 = torch.linalg.vector_norm(x64, dim=dim, keepdim=True).clamp_min(float(eps))
        out = (x64 / norm64).to(torch.float32)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def clamp_floating_to_dtype_(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Clamp a floating tensor in-place to the finite range representable by dtype."""
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            return tensor
        try:
            info = torch.finfo(dtype)
        except (TypeError, ValueError):
            return tensor
        return tensor.clamp_(min=-info.max, max=info.max)

    @staticmethod
    def finite_scalar_for_dtype(value: float, dtype: torch.dtype) -> float:
        """Clamp a scalar to a finite value representable by dtype."""
        out = float(value)
        if math.isnan(out):
            return 0.0
        try:
            info = torch.finfo(dtype)
        except (TypeError, ValueError):
            return out
        if out > info.max:
            return float(info.max)
        if out < -info.max:
            return float(-info.max)
        return out

    @staticmethod
    def sanitize_fp16(t: torch.Tensor) -> torch.Tensor:
        """Convert a tensor to finite float32 values."""
        return torch.nan_to_num(t.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _get_by_path(obj, path: str):
        """Return cached or derived by path data."""
        cur = obj
        for p in path.split("."):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    @staticmethod
    def ensure_bsh(
            t: torch.Tensor,
            hidden_size: int,
            *,
            strict: bool = True,
            from_layout: Optional[str] = None,
            batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return a tensor in batch-sequence-hidden layout."""
        if t.ndim != 3:
            if strict:
                raise ValueError(f"Expected 3D tensor (B,S,H); got shape={tuple(t.shape)}")
            return t

        if t.size(-1) != hidden_size:
            if strict:
                raise ValueError(
                    f"Expected hidden_size={hidden_size} on last dim; got shape={tuple(t.shape)}"
                )
            return t.contiguous()

        if from_layout is not None:
            fl = str(from_layout).upper()
            if fl == "BSH":
                return t.contiguous()
            if fl == "SBH":
                return t.permute(1, 0, 2).contiguous()
            raise ValueError(f"Unsupported from_layout={from_layout!r}")

        if batch_size is not None:
            b0 = (t.size(0) == batch_size)
            b1 = (t.size(1) == batch_size)

            if b0 and not b1:
                return t.contiguous()
            if b1 and not b0:
                return t.permute(1, 0, 2).contiguous()

            if b0 and b1:
                raise ValueError(
                    f"Ambiguous tensor layout for shape={tuple(t.shape)} and batch_size={batch_size}: "
                    "both dim0 and dim1 match batch_size. Pass from_layout explicitly."
                )

            raise ValueError(
                f"Cannot infer BSH/SBH for shape={tuple(t.shape)} and batch_size={batch_size}."
            )

        if strict:
            raise ValueError(
                f"Ambiguous 3D tensor layout for shape={tuple(t.shape)}. "
                "Pass from_layout='BSH'/'SBH' explicitly. "
                "Using only batch_size is insufficient when dim0 == dim1 == batch_size."
            )

        return t.contiguous()

    @staticmethod
    def unwrap_model(model: nn.Module) -> nn.Module:
        """Peel common wrapper modules to reach the underlying model."""
        m = model
        seen = set()
        while hasattr(m, "module") and getattr(m, "module") is not m and id(m) not in seen:
            seen.add(id(m))
            m = m.module
        return m

    @staticmethod
    def clear_transformer_layer_cache(model: Optional[nn.Module] = None) -> None:
        """Clear cached transformer-layer discovery results."""
        if model is None:
            ModelUtils._TRANSFORMER_LAYER_CACHE.clear()
            return

        try:
            base = ModelUtils.unwrap_model(model)
            ModelUtils._TRANSFORMER_LAYER_CACHE.pop(base, None)
        except TypeError:
            return

    @staticmethod
    def _get_cached_transformer_layers(base: nn.Module) -> Optional[List[nn.Module]]:
        """Return cached transformer layers for a model when available."""
        try:
            cached = ModelUtils._TRANSFORMER_LAYER_CACHE.get(base)
        except TypeError:
            return None
        if cached is None:
            return None
        return list(cached)

    @staticmethod
    def _store_cached_transformer_layers(base: nn.Module, layers: List[nn.Module]) -> None:
        """Store transformer layers in the weak model cache."""
        try:
            ModelUtils._TRANSFORMER_LAYER_CACHE[base] = list(layers)
        except TypeError:
            return

    @staticmethod
    def _residual_block_predicate(mod: nn.Module, path: str) -> bool:
        """Residual block predicate."""
        name = (path or "").lower()
        leaf = name.split(".")[-1]

        deny_suffixes = (
            "self_attn", "selfattention", "attention", "attn",
            "mlp", "ff", "ffn", "feed_forward",
            "layernorm", "rmsnorm", "norm",
            "proj", "o_proj", "k_proj", "q_proj", "v_proj",
            "moe", "experts", "expert", "router", "gate", "gating",
            "embed", "embedding", "dropout",
        )
        if any(leaf.endswith(suf) for suf in deny_suffixes):
            return False

        cname = mod.__class__.__name__.lower()

        allow_exact = {
            "llamadecoderlayer", "llama3decoderlayer",
            "mistraldecoderlayer",
            "openelmdecoderlayer",
            "qwen2decoderlayer", "qwen25decoderlayer", "qwen2_5decoderlayer",
            "qwen3decoderlayer",
            "gemma3decoderlayer", "gemmadecoderlayer", "gemmablock", "gemma3block",
            "mixtraldecoderlayer", "falcondecoderlayer", "optdecoderlayer",
            "bloomblock", "mptblock", "gptneoxlayer", "gptjblock",
            "gpt2block", "gptbigcodeblock", "deepseekdecoderlayer", "phi3decoderlayer",
        }
        if cname in allow_exact:
            return True

        has_attn = any(
            hasattr(mod, a) for a in (
                "self_attn", "self_attention", "attn", "attention"
            )
        )
        has_ff = any(
            hasattr(mod, a) for a in (
                "mlp", "feed_forward", "ff", "ffn", "dense_h_to_4h", "block_sparse_moe", "moe_layer"
            )
        )
        has_norm = any(
            hasattr(mod, a) for a in (
                "input_layernorm", "post_attention_layernorm",
                "norm", "rmsnorm", "layernorm", "ln1", "ln_1", "ln",
                "attn_norm", "ffn_norm"
            )
        )

        return has_attn and has_ff and has_norm

    @staticmethod
    def find_transformer_layers(
            model: nn.Module,
            *,
            predicate: Optional[Callable[[nn.Module, str], bool]] = None,
            allow_heuristics: bool = True,
    ) -> List[nn.Module]:
        """Find transformer block modules in a supported or heuristic way."""
        base = ModelUtils.unwrap_model(model)
        use_cache = predicate is None and bool(allow_heuristics)
        if use_cache:
            cached = ModelUtils._get_cached_transformer_layers(base)
            if cached is not None:
                return cached

        config = getattr(base, "config", None)
        expected = int(
            getattr(config, "num_hidden_layers", 0)
            or getattr(config, "num_transformer_layers", 0)
            or getattr(config, "n_layer", 0)
            or getattr(config, "num_layers", 0)
            or getattr(config, "n_layers", 0)
            or 0
        )

        model_type = getattr(config, "model_type", None) if config is not None else None

        if model_type in {"llama", "qwen2", "qwen3"}:
            known = [
                "model.layers",
                "model.model.layers",
            ]
        elif model_type == "gemma3":
            known = [
                "model.layers",
                "language_model.model.layers",
                "language_model.layers",
                "model.model.layers",
            ]
        elif model_type == "openelm":
            known = [
                "transformer.layers",
                "model.transformer.layers",
            ]
        else:
            known = [
                "model.layers",
                "model.model.layers",
                "transformer.layers",
                "model.transformer.layers",
                "transformer.h",
                "model.transformer.h",
                "transformer.blocks",
                "model.transformer.blocks",
                "gpt_neox.layers",
                "model.decoder.layers",
                "decoder.layers",
                "language_model.model.layers",
                "language_model.layers",
                "text_model.model.layers",
                "text_model.layers",
                "layers",
            ]

        for kp in known:
            layers = ModelUtils._get_by_path(base, kp)
            if isinstance(layers, (nn.ModuleList, list)) and layers and all(isinstance(x, nn.Module) for x in layers):
                if expected and len(layers) != expected:
                    continue
                out = list(layers)
                if use_cache:
                    ModelUtils._store_cached_transformer_layers(base, out)
                return out

        if not allow_heuristics:
            raise RuntimeError("Could not locate transformer layers via known paths.")

        pred = predicate or ModelUtils._residual_block_predicate
        found: List[tuple[str, nn.Module]] = []

        def dfs(mod: nn.Module, path: str) -> bool:
            """Depth-first search used by transformer layer discovery."""
            matched_child = False
            for child_name, child in mod.named_children():
                child_path = f"{path}.{child_name}" if path else child_name
                if dfs(child, child_path):
                    matched_child = True

            if pred(mod, path):
                if not matched_child:
                    found.append((path, mod))
                return True

            return matched_child

        dfs(base, "")

        def key(item):
            """Sort key for discovered transformer layer paths."""
            path, _ = item
            nums = [int(x) for x in re.findall(r"\b(\d+)\b", path)]
            return nums or [10 ** 9, hash(path) & 0xFFFF]

        found.sort(key=key)
        out = [m for _, m in found]
        if use_cache:
            ModelUtils._store_cached_transformer_layers(base, out)
        return out

    @staticmethod
    def get_hidden_size(config) -> int:
        """Read the hidden size from a model config."""
        for k in ("hidden_size", "n_embd", "d_model", "model_dim"):
            if hasattr(config, k):
                return int(getattr(config, k))
        raise AttributeError("Could not determine hidden size from config (no hidden_size/n_embd/d_model).")
