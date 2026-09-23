"""Focused services used internally by the Executor facade."""

from __future__ import annotations

import gc
import logging
import math
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        """Fallback progress wrapper when tqdm is unavailable."""
        return iterable if iterable is not None else []

from activation_manager.ActivationExtractor import ActivationExtractor
from activation_manager.SteeredModelWrapper import SteeredModelWrapper
from activation_manager.VectorMediator import VectorMediator
from enums.ApplyFromModeEnum import ApplyFromModeEnum
from enums.TaskTypeEnum import TaskTypeEnum
from steering_engine.components import CompileContext
from steering_engine.components import DerivationContext
from steering_engine.components import RuntimeContext
from steering_engine.runtime import build_residual_stream_runtime_wrapper
from utils.ModelUtils import ModelUtils
from vector_update_strategy.ColdKernelGradientMediator import ColdKernelGradientMediator

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None


class PromptDataset(Dataset):
    """Small in-memory dataset for already validated prompt rows."""
    def __init__(self, data_iter: Sequence[Dict[str, Any]]) -> None:
        """Initialize the instance and validate its configuration."""
        self._data = list(data_iter)

    def __len__(self) -> int:
        """Return the number of stored rows."""
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one stored row by integer index."""
        return self._data[idx]


def identity_collate(batch):
    """Return a batch unchanged for row-dictionary loaders."""
    return batch


def count_forward_hooks(model: nn.Module) -> int:
    """Count forward and pre-forward hooks on an unwrapped model."""
    base = ModelUtils.unwrap_model(model)
    tot = 0
    for m in base.modules():
        tot += len(getattr(m, "_forward_hooks", {}))
        tot += len(getattr(m, "_forward_pre_hooks", {}))
    return tot


def assert_no_forward_hooks(model: nn.Module, where: str = "") -> None:
    """Raise if an unwrapped model still has active forward hooks."""
    base = ModelUtils.unwrap_model(model)
    n = count_forward_hooks(base)
    if n != 0:
        raise RuntimeError(
            f"[HOOK LEAK] Found {n} forward/pre-forward hooks on base(unwrapped) model {where}. "
            "This can contaminate baseline generations."
        )


@contextmanager
def attached_activation_extractor(extractor: ActivationExtractor):
    """Attach an activation extractor for the duration of a context block."""
    extractor.attach()
    try:
        yield
    finally:
        extractor.remove()


class ExecutorMethodContext:
    """Method metadata and registry-facing options for an Executor facade."""

    def __init__(self, owner: Any) -> None:
        """Initialize the instance and validate its configuration."""
        self.owner = owner

    def definition(self):
        """Return the active method definition."""
        return self.owner.method_registry.get(self.owner.method)

    def options(self, *, n_layers: Optional[int] = None) -> Dict[str, Any]:
        """Return registry options derived from the owning executor."""
        options: Dict[str, Any] = {
            "use_last_token_for_response": bool(self.owner.use_last_token_for_response),
            "tokens_window": int(self.owner.tokens_window),
            "cast_tensor_map": self.owner._cast_tensor_map,
        }
        if n_layers is not None:
            options["n_layers"] = int(n_layers)
        return options

    def is_gradient_based(self) -> bool:
        """Return whether the active method needs gradient-based training."""
        return bool(self.definition().spec.training_requires_grad)

    def supports_routing(self) -> bool:
        """Return whether the active method uses prompt routing."""
        return bool(self.definition().spec.supports_routing)

    def training_layers(self) -> List[int]:
        """Return the concrete layer set needed for training."""
        layers = sorted({int(i) for i in self.owner.layers_to_extract})
        n_layers = len(ModelUtils.find_transformer_layers(self.owner.model))
        return list(
            self.owner.method_registry.training_layers_for(
                self.owner.method,
                layers,
                self.options(n_layers=n_layers),
            )
        )

    def require_activations(self, acts_full: Dict[int, torch.Tensor]) -> None:
        """Validate that required layer activations were captured."""
        owner = self.owner
        if not acts_full:
            raise RuntimeError(
                "Activation extraction produced an empty dict. "
                "This indicates hooks did not capture any activations for the requested layers."
            )
        required_layers = (
            owner.activation_extractor.layers_to_extract
            if owner.activation_extractor is not None
            else owner.layers_to_extract
        )
        missing = [i for i in required_layers if i not in acts_full]
        if missing:
            got = sorted(acts_full.keys())
            raise RuntimeError(
                f"Missing activations for requested layers {missing}. Got layers={got}. "
                "This is treated as fatal to avoid silent training failures."
            )

    def require_group_ids_if_needed(self, group_ids: Optional[torch.Tensor]) -> None:
        """Validate group identifiers required by routed methods."""
        owner = self.owner
        required_columns = set(self.definition().spec.required_columns)
        if "task_id" in required_columns and group_ids is None:
            raise ValueError("This steering method requires a non-null 'task_id' for every row.")
        if "mbs_layer" in required_columns and group_ids is None:
            raise ValueError("This steering method requires a non-null 'mbs_layer' for every row.")
        if owner.method == "mbs_cmd" and group_ids is None:
            raise ValueError("MBS_CMD requires mbs_layer for every row (group_ids=None).")


class ExecutorTokenizerIO:
    """Tokenizer state, batching, and device inference."""

    def __init__(self, owner: Any) -> None:
        """Initialize the instance and validate its configuration."""
        self.owner = owner

    def ensure_pad_token(self) -> None:
        """Ensure the tokenizer and model config expose a pad token."""
        owner = self.owner
        tok = owner.tokenizer
        if tok is None:
            raise ValueError("Executor requires a tokenizer.")

        if getattr(tok, "pad_token_id", None) is not None:
            return

        eos_token = getattr(tok, "eos_token", None)
        eos_token_id = getattr(tok, "eos_token_id", None)

        try:
            if eos_token is not None:
                tok.pad_token = eos_token
            elif eos_token_id is not None and hasattr(tok, "convert_ids_to_tokens"):
                eos_token = tok.convert_ids_to_tokens(int(eos_token_id))
                if eos_token is None:
                    raise ValueError("convert_ids_to_tokens returned None")
                tok.pad_token = eos_token
            else:
                raise ValueError("missing eos_token/eos_token_id")
        except Exception as e:
            raise ValueError(
                "Tokenizer has no pad_token. Set tokenizer.pad_token explicitly, "
                "or provide a tokenizer with eos_token so pad_token can be derived automatically."
            ) from e

        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            raise ValueError("Tokenizer pad_token could not be initialized. Set tokenizer.pad_token explicitly.")

        try:
            base = ModelUtils.unwrap_model(owner.model)
            cfg = getattr(base, "config", None)
            if cfg is not None and getattr(cfg, "pad_token_id", None) is None:
                cfg.pad_token_id = int(pad_id)

            gen_cfg = getattr(base, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
                gen_cfg.pad_token_id = int(pad_id)
        except Exception:
            pass

    @contextmanager
    def tokenizer_sides(self):
        """Temporarily apply executor tokenizer padding and truncation sides."""
        owner = self.owner
        tok = owner.tokenizer
        if tok is None:
            yield
            return

        self.ensure_pad_token()

        had_pad = hasattr(tok, "padding_side")
        had_trunc = hasattr(tok, "truncation_side")

        old_pad = tok.padding_side if had_pad else None
        old_trunc = tok.truncation_side if had_trunc else None

        try:
            if had_pad and old_pad != owner.padding_side:
                tok.padding_side = owner.padding_side
            if had_trunc and old_trunc != owner.truncation_side:
                tok.truncation_side = owner.truncation_side
            yield
        finally:
            if had_pad:
                tok.padding_side = old_pad
            if had_trunc:
                tok.truncation_side = old_trunc

    def encode_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize prompts with the executor tokenizer policy."""
        with self.tokenizer_sides():
            return self.owner.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.owner.max_prompt_tokens,
            )

    def pad_from_ids(self, id_lists: List[List[int]]) -> Dict[str, torch.Tensor]:
        """Pad already-tokenized sequences with the executor tokenizer."""
        items = [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in id_lists]
        with self.tokenizer_sides():
            return self.owner.tokenizer.pad(items, padding=True, return_tensors="pt")

    @staticmethod
    def is_missing_response_value(v: Any) -> bool:
        """Return whether a response cell should be treated as missing."""
        if v is None:
            return True
        if isinstance(v, (list, tuple)):
            return False

        if pd is None:
            try:
                return bool(math.isnan(v))
            except (TypeError, ValueError):
                return False

        try:
            missing = pd.isna(v)
        except Exception:
            return False

        if isinstance(missing, bool):
            return missing

        try:
            return bool(missing)
        except (TypeError, ValueError):
            return False

    def trim_generated_tokens(self, gen_ids: List[int]) -> List[int]:
        """Remove terminal EOS and pad tokens from generated token ids."""
        if not gen_ids:
            return gen_ids

        eos_id = getattr(self.owner.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.owner.tokenizer, "pad_token_id", None)

        if eos_id is not None:
            try:
                j = gen_ids.index(int(eos_id))
                gen_ids = gen_ids[:j]
            except ValueError:
                pass

        if pad_id is not None:
            while gen_ids and gen_ids[-1] == int(pad_id):
                gen_ids.pop()

        return gen_ids

    @staticmethod
    def _cuda_or_cpu(index: Optional[int] = None) -> torch.device:
        """Return a concrete CUDA device only when it is locally usable."""
        if not torch.cuda.is_available():
            return torch.device("cpu")

        idx = 0 if index is None else int(index)
        if idx < 0:
            return torch.device("cpu")

        try:
            count = int(torch.cuda.device_count())
        except Exception:
            count = 0
        if count > 0 and idx >= count:
            return torch.device("cpu")

        return torch.device(f"cuda:{idx}")

    @classmethod
    def as_torch_device(cls, v) -> torch.device:
        """Convert common device descriptors to a torch device."""
        if isinstance(v, torch.device):
            if v.type == "cuda":
                return cls._cuda_or_cpu(v.index)
            if v.type == "meta":
                return torch.device("cpu")
            return v
        if v is None:
            return torch.device("cpu")
        if isinstance(v, int):
            return cls._cuda_or_cpu(int(v))

        s = str(v).strip().lower()
        if s in {"disk", "meta", "offload", "nvme"}:
            return torch.device("cpu")
        if s.isdigit():
            return cls._cuda_or_cpu(int(s))
        if s == "cuda":
            return cls._cuda_or_cpu()
        if s.startswith("cuda:"):
            try:
                return cls._cuda_or_cpu(int(s.split(":", 1)[1]))
            except (TypeError, ValueError):
                return torch.device("cpu")

        try:
            return torch.device(s)
        except (TypeError, ValueError):
            return torch.device("cpu")

    @staticmethod
    def _first_real_tensor_device(module: Optional[nn.Module]) -> Optional[torch.device]:
        """Return the first non-meta parameter or buffer device from a module."""
        if module is None:
            return None

        for getter_name in ("parameters", "buffers"):
            getter = getattr(module, getter_name, None)
            if not callable(getter):
                continue
            try:
                iterator = getter()
            except Exception:
                continue
            for tensor in iterator:
                if torch.is_tensor(tensor) and tensor.device.type != "meta":
                    return tensor.device
        return None

    @classmethod
    def infer_input_device(cls, model: nn.Module) -> torch.device:
        """Infer where model input ids should be placed."""
        base = ModelUtils.unwrap_model(model)
        dm = getattr(base, "hf_device_map", None)
        if isinstance(dm, dict) and dm:
            preferred_suffixes = ("embed_tokens", "wte", "embeddings", "tok_embeddings", "word_embeddings")
            for k, v in dm.items():
                if any(k.endswith(suf) or suf in k for suf in preferred_suffixes):
                    return cls.as_torch_device(v)
            if "" in dm:
                return cls.as_torch_device(dm[""])

        try:
            if hasattr(base, "get_input_embeddings"):
                emb = base.get_input_embeddings()
                if emb is not None:
                    dev = cls._first_real_tensor_device(emb)
                    if dev is not None:
                        return cls.as_torch_device(dev)
        except Exception:
            pass

        if isinstance(dm, dict) and dm:
            for v in dm.values():
                dv = cls.as_torch_device(v)
                if dv.type == "cuda":
                    return dv
            return cls.as_torch_device(next(iter(dm.values())))

        dev = cls._first_real_tensor_device(model)
        return cls.as_torch_device(dev) if dev is not None else torch.device("cpu")


class ExecutorDatasetBuilder:
    """Validates training rows and builds DataLoaders."""

    def __init__(self, owner: Any, methods: ExecutorMethodContext) -> None:
        """Initialize the instance and validate its configuration."""
        self.owner = owner
        self.methods = methods

    @staticmethod
    def _coerce_integer_column(df: pd.DataFrame, column: str, *, label: str) -> pd.DataFrame:
        """Return a copy with one required column normalized to integer ids."""
        out = df.copy()
        try:
            numeric = pd.to_numeric(out[column], errors="raise")
        except Exception as e:
            raise ValueError(f"{label} requires integer-like values in '{column}'.") from e

        if numeric.isna().any():
            raise ValueError(f"{label} requires non-null '{column}' for every row.")

        try:
            ints = numeric.astype("int64")
        except Exception as e:
            raise ValueError(f"{label} requires integer-like values in '{column}'.") from e

        fractional = numeric != ints
        if bool(fractional.any()):
            bad_values = out.loc[fractional, column].head(10).tolist()
            raise ValueError(
                f"{label} requires integer-like values in '{column}'. "
                f"Non-integer examples: {bad_values}"
            )

        out[column] = ints
        return out

    @staticmethod
    def _reference_balance_problems(
        df: pd.DataFrame,
        *,
        group_column: Optional[str] = None,
    ) -> List[Any]:
        """Return groups that lack positive or negative reference rows."""
        problems: List[Any] = []

        if group_column is None:
            refs = df["reference"].astype(float)
            has_pos = bool((refs == 1.0).any())
            has_neg = bool((refs == 0.0).any())
            if not has_pos or not has_neg:
                problems.append(("all rows", f"has_pos={has_pos}, has_neg={has_neg}"))
            return problems

        for group_value, sub in df.groupby(group_column, sort=False, dropna=False):
            refs = sub["reference"].astype(float)
            has_pos = bool((refs == 1.0).any())
            has_neg = bool((refs == 0.0).any())
            if not has_pos or not has_neg:
                problems.append((group_value, f"has_pos={has_pos}, has_neg={has_neg}"))
        return problems

    def _require_reference_balance(
        self,
        df: pd.DataFrame,
        *,
        method: str,
        group_column: Optional[str] = None,
    ) -> None:
        """Require both reference classes globally or within each group."""
        problems = self._reference_balance_problems(df, group_column=group_column)
        if not problems:
            return

        scope = "globally" if group_column is None else f"within each '{group_column}' group"
        preview = problems[:20]
        more = "" if len(problems) <= 20 else f" (+{len(problems) - 20} more)"
        raise ValueError(
            f"{method} training requires both positive (reference=1) and negative "
            f"(reference=0) examples {scope}. Problems: {preview}{more}"
        )

    def prepare(self, df: pd.DataFrame) -> DataLoader:
        """Validate a training frame and return a data loader."""
        if pd is None:
            raise RuntimeError("ExecutorDatasetBuilder.prepare requires pandas to be installed.")

        owner = self.owner
        df = pd.DataFrame(df)

        required = set(self.methods.definition().spec.required_columns or ("prompt", "reference"))
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if df.empty:
            raise ValueError("Training dataframe must contain at least one row.")

        if owner.task_type != TaskTypeEnum.UNSUPERVISED:
            raise NotImplementedError("No technique other than unsupervised supported")

        if not df["reference"].isin([0, 1]).all():
            raise ValueError("For UNSUPERVISED mode the 'reference' column must contain only 0 or 1.")
        if "response" not in df.columns:
            raise ValueError("UNSUPERVISED training requires a 'response' column for every row.")
        missing_response = df["response"].map(ExecutorTokenizerIO.is_missing_response_value)
        if bool(missing_response.any()):
            raise ValueError("UNSUPERVISED training requires 'response' to be present for every row.")

        if owner.method in ("cmd", "cpca", "angular", "cold_kernel", "cold_steer"):
            self._require_reference_balance(df, method=owner.method)

        elif owner.method == "mbs_cmd":
            if "mbs_layer" not in df.columns:
                raise ValueError("MBS_CMD training requires an 'mbs_layer' column.")
            if df["mbs_layer"].isna().any():
                raise ValueError("MBS_CMD training requires non-null 'mbs_layer' for every row.")

            df = self._coerce_integer_column(df, "mbs_layer", label="MBS_CMD training")
            allowed = {int(x) for x in owner.layers_to_extract}
            seen = {int(x) for x in df["mbs_layer"].dropna().tolist()}
            bad = sorted(seen - allowed)
            if bad:
                raise ValueError(
                    f"MBS_CMD df['mbs_layer'] contains layers not in layers_to_extract: {bad}. "
                    f"Allowed layers: {sorted(allowed)}"
                )

            missing_balance = []
            for layer_idx in sorted(allowed):
                sub = df[df["mbs_layer"] == int(layer_idx)]
                if sub.empty:
                    missing_balance.append((int(layer_idx), "no samples"))
                    continue
                refs = sub["reference"].astype(float)
                has_pos = bool((refs >= 0.5).any())
                has_neg = bool((refs < 0.5).any())
                if not has_pos or not has_neg:
                    missing_balance.append((int(layer_idx), f"has_pos={has_pos}, has_neg={has_neg}"))

            if missing_balance:
                raise ValueError(
                    "MBS_CMD requires every layer in layers_to_extract to have both positive "
                    f"and negative examples. Problems: {missing_balance}"
                )

        return DataLoader(
            PromptDataset(df.to_dict("records")),
            batch_size=owner.batch_size,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )


class ExecutorComponentBuilder:
    """Builds derivation, steering strategy, and runtime wrapper components."""

    def __init__(self, owner: Any, methods: ExecutorMethodContext) -> None:
        """Initialize the instance and validate its configuration."""
        self.owner = owner
        self.methods = methods

    def setup_vector_components(self) -> None:
        """Create extractor, mediator, and derivation components."""
        owner = self.owner
        try:
            layers = list(ModelUtils.find_transformer_layers(owner.model))
            if not layers:
                raise RuntimeError("No transformer layers found: check ModelUtils.find_transformer_layers().")

            training_layers = list(
                owner.method_registry.training_layers_for(
                    owner.method,
                    sorted({int(i) for i in owner.layers_to_extract}),
                    self.methods.options(n_layers=len(layers)),
                )
            )
            base = ModelUtils.unwrap_model(owner.model)

            cfg_max = int(getattr(base.config, "max_position_embeddings", 0) or 0)
            tok_max = getattr(owner.tokenizer, "model_max_length", None)
            tok_max_int = None
            try:
                if tok_max is not None:
                    tok_max_int = int(tok_max)
                    if tok_max_int >= 10 ** 9:
                        tok_max_int = None
            except Exception:
                tok_max_int = None

            candidates = [x for x in (cfg_max, tok_max_int) if isinstance(x, int) and x > 0]
            owner.max_prompt_tokens = min(candidates) if candidates else 2048

            arch = getattr(base.config, "is_encoder_decoder", False)
            if arch:
                raise NotImplementedError("Encoder-decoder architectures are not supported by this steering wrapper.")

            hidden_size = ModelUtils.get_hidden_size(base.config)
            ctx = DerivationContext(
                model=owner.model,
                tokenizer=owner.tokenizer,
                layers=training_layers,
                hidden_size=hidden_size,
                dtype=owner.dtype,
                options=self.methods.options(n_layers=len(layers)),
            )
            owner.vector_update_strategy = owner.method_registry.create_vector_strategy(owner.method, ctx)
            owner.activation_extractor = ActivationExtractor(
                model=owner.model,
                layers_to_extract=training_layers,
                offload_to_cpu=True,
                decode_chunk_max=owner.decode_chunk_max,
            )
            owner.vector_mediator = VectorMediator(
                layers_to_extract=training_layers,
                hidden_size=hidden_size,
                update_strategy=owner.vector_update_strategy,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to set up vector components: {e}") from e

    def build_steering_strategy(self) -> None:
        """Compile the finalized update strategy into runtime steering."""
        owner = self.owner
        try:
            upd = owner.vector_mediator.update_strategy if owner.vector_mediator else None
            if upd is None:
                raise ValueError("Vector mediator update strategy is not initialized")

            ctx = CompileContext(
                model=owner.model,
                tokenizer=owner.tokenizer,
                requested_layers=sorted({int(i) for i in owner.layers_to_extract}),
                dtype=owner.dtype,
                options=self.methods.options(),
            )
            owner.steering_strategy = owner.method_registry.build_runtime_strategy(owner.method, upd, ctx)
        except Exception as e:
            raise RuntimeError(f"Failed to build steering strategy: {e}") from e

    def make_runtime_wrapper(self) -> Any:
        """Create the runtime wrapper around the base model."""
        owner = self.owner
        if owner.steering_strategy is None:
            raise RuntimeError("steering_strategy is None")

        ctx = RuntimeContext(
            model=owner.model,
            tokenizer=owner.tokenizer,
            strategy=owner.steering_strategy,
            alpha=owner.alpha,
            apply_from_token=owner.apply_from_token,
            apply_from_mode=owner.apply_from_mode,
            tokens_window=owner.tokens_window,
            padding_side=owner.padding_side,
            dtype=owner.dtype,
            generation_kwargs=dict(owner.generation_kwargs),
            options=self.methods.options(),
        )
        return owner.method_registry.build_runtime_wrapper(
            owner.method,
            ctx,
            default_factory=build_residual_stream_runtime_wrapper,
        )


class ExecutorTrainingOrchestrator:
    """Coordinates training loops, finalization, and wrapper construction."""

    def __init__(
        self,
        owner: Any,
        methods: ExecutorMethodContext,
        tokenizer_io: ExecutorTokenizerIO,
        components: ExecutorComponentBuilder,
    ) -> None:
        """Initialize the instance and validate its configuration."""
        self.owner = owner
        self.methods = methods
        self.tokenizer_io = tokenizer_io
        self.components = components

    def run_standard_epoch(self, loader) -> None:
        """Run one activation-extraction epoch for non-gradient methods."""
        owner = self.owner
        if owner.activation_extractor is None:
            raise RuntimeError("activation_extractor is None")
        if owner.vector_mediator is None:
            raise RuntimeError("vector_mediator is None")

        model_device = self.tokenizer_io.infer_input_device(owner.model)
        max_len = int(getattr(owner, "max_prompt_tokens", 2048))

        for raw_batch in tqdm(loader, desc="Extracting activations"):
            batch = cast(List[Dict[str, Any]], raw_batch)
            enc = self.tokenizer_io.encode_prompts([d["prompt"] for d in batch])
            enc = {k: v.to(model_device) for k, v in enc.items()}

            if owner.task_type != TaskTypeEnum.UNSUPERVISED:
                raise NotImplementedError("No technique other than unsupervised supported")

            inp_ids = enc["input_ids"].detach().cpu()
            attn_inp = enc["attention_mask"].detach().cpu()
            prompt_lens_cpu = attn_inp.sum(1, dtype=torch.int64)

            resp_fields = [d.get("response", None) for d in batch]
            missing = [i for i, r in enumerate(resp_fields) if self.tokenizer_io.is_missing_response_value(r)]
            if missing:
                meta = [
                    {"i": i, "prompt_id": batch[i].get("prompt_id"), "task_id": batch[i].get("task_id")}
                    for i in missing[:10]
                ]
                more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
                raise ValueError(
                    "UNSUPERVISED training requires 'response' for every row (teacher forcing). "
                    f"Missing response in batch positions: {missing[:10]}{more}. Meta preview: {meta}."
                )

            concat_lists: List[List[int]] = []
            kept_batch: List[Dict[str, Any]] = []
            kept_prompt_lens: List[int] = []
            for i in range(inp_ids.size(0)):
                prompt_tokens = inp_ids[i, attn_inp[i].to(torch.bool)].tolist()
                original_prompt_len = int(prompt_lens_cpu[i].item())
                raw_response = resp_fields[i]
                if isinstance(raw_response, (list, tuple)) and all(isinstance(x, int) for x in raw_response):
                    resp_ids = list(map(int, raw_response))
                else:
                    with self.tokenizer_io.tokenizer_sides():
                        resp_ids = owner.tokenizer(
                            "" if raw_response is None else str(raw_response),
                            add_special_tokens=False,
                            return_tensors=None,
                        )["input_ids"]

                seq = prompt_tokens + self.tokenizer_io.trim_generated_tokens(list(resp_ids))
                if len(seq) > max_len:
                    dropped = len(seq) - max_len
                    pl = original_prompt_len
                    if owner.truncation_side == "left":
                        dropped_from_prompt = min(dropped, pl)
                        dropped_from_response = max(0, dropped - dropped_from_prompt)
                        if dropped_from_response > 0:
                            logger.warning(
                                "UNSUPERVISED: left-truncation removed part of the RESPONSE. "
                                "Skipping sample i=%d (prompt_len=%d, dropped=%d, "
                                "dropped_from_response=%d, max_len=%d).",
                                int(i), int(pl), int(dropped), int(dropped_from_response), int(max_len),
                            )
                            continue
                        seq = seq[-max_len:]
                        pl = max(0, pl - dropped_from_prompt)
                        prompt_lens_cpu[i] = pl
                    else:
                        logger.warning(
                            "UNSUPERVISED: right-truncation would remove part of the RESPONSE. "
                            "Skipping sample i=%d (prompt_len=%d, dropped=%d, max_len=%d).",
                            int(i), int(pl), int(dropped), int(max_len),
                        )
                        continue

                pl_now = int(prompt_lens_cpu[i].item())
                if pl_now <= 0:
                    d = batch[i]
                    logger.warning(
                        "UNSUPERVISED: prompt_len became 0 after truncation; skipping sample i=%d "
                        "(prompt_id=%s, task_id=%s, max_len=%d, truncation_side=%s).",
                        int(i), str(d.get("prompt_id")), str(d.get("task_id")), int(max_len), str(owner.truncation_side),
                    )
                    continue
                if pl_now >= len(seq):
                    d = batch[i]
                    logger.warning(
                        "UNSUPERVISED: response has no remaining tokens after truncation; "
                        "skipping sample i=%d (prompt_id=%s, task_id=%s, prompt_len=%d, seq_len=%d).",
                        int(i), str(d.get("prompt_id")), str(d.get("task_id")), int(pl_now), int(len(seq)),
                    )
                    continue
                concat_lists.append(seq)
                kept_batch.append(batch[i])
                kept_prompt_lens.append(pl_now)

            if not concat_lists:
                logger.warning(
                    "UNSUPERVISED: all samples in batch were skipped due to response truncation (max_len=%d).",
                    int(max_len),
                )
                continue

            batch = kept_batch
            prompt_lens_cpu = torch.tensor(kept_prompt_lens, dtype=torch.int64)
            enc_aug = self.tokenizer_io.pad_from_ids(concat_lists)
            enc_aug = {k: v.to(model_device) for k, v in enc_aug.items()}

            owner.activation_extractor.clear()
            _ = owner.model(
                input_ids=enc_aug["input_ids"],
                attention_mask=enc_aug["attention_mask"],
                use_cache=False,
            )
            owner.activation_extractor.assert_ok()

            acts_full = dict(owner.activation_extractor.finalize(clear_chunks=True).items())
            self.methods.require_activations(acts_full)

            device = next(iter(acts_full.values())).device
            attn_bool = enc_aug["attention_mask"].to(device=device, dtype=torch.bool)
            total_lens = enc_aug["attention_mask"].sum(1, dtype=torch.int64).to(device)
            s_pad = int(enc_aug["input_ids"].size(1))
            left_pad = (s_pad - total_lens).to(dtype=torch.int64) if owner.padding_side == "left" else torch.zeros_like(total_lens)
            prompt_lens = prompt_lens_cpu.to(device=device, dtype=torch.int64)

            if owner.apply_from_mode == ApplyFromModeEnum.FIXED:
                starts = left_pad + int(owner.apply_from_token) if owner.padding_side == "left" else torch.full_like(
                    prompt_lens,
                    int(owner.apply_from_token),
                    device=device,
                )
            else:
                starts = left_pad + prompt_lens if owner.padding_side == "left" else prompt_lens
            starts = starts.to(dtype=torch.int64)

            labels = torch.tensor([d["reference"] for d in batch], dtype=torch.float32, device=device)
            if owner.method == "mbs_cmd":
                tid_s = [d.get("mbs_layer") for d in batch]
            else:
                tid_s = [d.get("task_id") for d in batch]
            group_ids = (
                torch.tensor([int(t) for t in tid_s], dtype=torch.long, device=device)
                if tid_s and all(t is not None for t in tid_s)
                else None
            )
            self.methods.require_group_ids_if_needed(group_ids)

            owner.vector_mediator.update_vectors(
                activations=acts_full,
                labels=labels,
                starts=starts,
                tokens_window=owner.tokens_window,
                attention_mask=attn_bool,
                group_ids=group_ids,
            )

            owner.activation_extractor.clear()
            del acts_full, enc_aug, enc

    def run_cold_kernel_epoch(self, loader) -> None:
        """Run one gradient-collection epoch for COLD-Kernel methods."""
        import torch.nn.functional as F

        owner = self.owner
        if not isinstance(owner.vector_update_strategy, ColdKernelGradientMediator):
            raise RuntimeError("COLD training requires ColdKernelGradientMediator.")

        model_device = self.tokenizer_io.infer_input_device(owner.model)
        max_len = int(getattr(owner, "max_prompt_tokens", 2048))
        base = ModelUtils.unwrap_model(owner.model)
        hidden_size = ModelUtils.get_hidden_size(base.config)
        layers = ModelUtils.find_transformer_layers(owner.model)

        def make_grad_hook(layer_idx: int, store: Dict[int, torch.Tensor]):
            """Make grad hook."""
            def hook(_module, _inputs, output):
                """Forward hook used to capture or replace hidden states."""
                hidden, kind, meta = SteeredModelWrapper._extract_hidden_from_output(output)
                if hidden is None:
                    return output
                hidden = ModelUtils.ensure_bsh(hidden, hidden_size, from_layout="BSH")
                if not hidden.requires_grad:
                    hidden = hidden.detach().clone().requires_grad_(True)
                hidden.retain_grad()
                store[int(layer_idx)] = hidden
                return SteeredModelWrapper._replace_hidden_in_output(output, hidden, kind, meta)

            return hook

        for raw_batch in tqdm(loader, desc="COLD-Kernel gradients"):
            batch = cast(List[Dict[str, Any]], raw_batch)
            enc = self.tokenizer_io.encode_prompts([d["prompt"] for d in batch])
            inp_ids = enc["input_ids"].detach().cpu()
            attn_inp = enc["attention_mask"].detach().cpu()
            prompt_lens_cpu = attn_inp.sum(1, dtype=torch.int64)

            concat_lists, kept_batch, kept_prompt_lens = [], [], []
            for i in range(inp_ids.size(0)):
                prompt_tokens = inp_ids[i, attn_inp[i].to(torch.bool)].tolist()
                r = batch[i].get("response", None)
                if self.tokenizer_io.is_missing_response_value(r):
                    raise ValueError("COLD-Kernel requires 'response' for every row.")
                with self.tokenizer_io.tokenizer_sides():
                    resp_ids = owner.tokenizer(str(r), add_special_tokens=False, return_tensors=None)["input_ids"]
                seq = prompt_tokens + self.tokenizer_io.trim_generated_tokens(list(resp_ids))
                if len(seq) > max_len:
                    continue
                if int(prompt_lens_cpu[i].item()) <= 0 or int(prompt_lens_cpu[i].item()) >= len(seq):
                    continue
                concat_lists.append(seq)
                kept_batch.append(batch[i])
                kept_prompt_lens.append(int(prompt_lens_cpu[i].item()))

            if not concat_lists:
                continue

            enc_aug = self.tokenizer_io.pad_from_ids(concat_lists)
            enc_aug = {k: v.to(model_device) for k, v in enc_aug.items()}
            prompt_lens = torch.tensor(kept_prompt_lens, dtype=torch.int64, device=model_device)
            total_lens = enc_aug["attention_mask"].sum(1, dtype=torch.int64).to(model_device)
            s_pad = int(enc_aug["input_ids"].size(1))
            left_pad = (s_pad - total_lens) if owner.padding_side == "left" else torch.zeros_like(total_lens)
            starts = (left_pad + prompt_lens) if owner.padding_side == "left" else prompt_lens

            pos = torch.arange(s_pad, device=model_device).unsqueeze(0)
            response_mask = (pos >= starts.unsqueeze(1)) & enc_aug["attention_mask"].to(torch.bool)
            if owner.tokens_window > 0:
                response_mask &= pos < (starts + int(owner.tokens_window)).unsqueeze(1)

            labels_full = enc_aug["input_ids"].clone()
            labels_full[~response_mask] = -100

            grad_store: Dict[int, torch.Tensor] = {}
            handles = [
                layers[int(layer_idx)].register_forward_hook(make_grad_hook(int(layer_idx), grad_store))
                for layer_idx in owner.layers_to_extract
            ]
            try:
                owner.model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    out = owner.model(
                        input_ids=enc_aug["input_ids"],
                        attention_mask=enc_aug["attention_mask"],
                        use_cache=False,
                    )
                    logits = out.logits
                    loss_device = logits.device
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels_full[:, 1:].to(device=loss_device).contiguous()
                    ce = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(shift_labels.shape)
                    valid = shift_labels.ne(-100)
                    per_sample = ce.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
                    refs = torch.tensor(
                        [float(d["reference"]) for d in kept_batch],
                        dtype=torch.float32,
                        device=loss_device,
                    )
                    signs = torch.where(refs >= 0.5, torch.ones_like(refs), -torch.ones_like(refs))
                    objective = (signs * per_sample).mean()
                    objective.backward()

                grads = {idx: t.grad.detach() for idx, t in grad_store.items() if t.grad is not None}
                owner.vector_update_strategy.update_from_gradients(grads, response_mask=response_mask.detach())
            finally:
                for handle in handles:
                    handle.remove()
                owner.model.zero_grad(set_to_none=True)

    def representation_extractor(self) -> Any:
        """Train or reuse steering artifacts and return a steered wrapper."""
        owner = self.owner
        assert_no_forward_hooks(owner.model, where="(Executor.representation_extractor: entry)")

        if owner.steering_strategy is not None:
            return self.components.make_runtime_wrapper()

        if owner.vector_update_strategy is None or owner.activation_extractor is None or owner.vector_mediator is None:
            self.components.setup_vector_components()
        else:
            if hasattr(owner.vector_update_strategy, "reset"):
                owner.vector_update_strategy.reset()
            owner.vector_mediator.clear_errors()
            owner.activation_extractor.remove()
            owner.activation_extractor.clear()

        if owner.dataset is None:
            raise RuntimeError("Training dataset is not available and no cached steering strategy exists.")

        owner.model.eval()
        method_definition = self.methods.definition()
        if method_definition.training_runner is not None:
            method_definition.training_runner(owner, owner.dataset)
        elif self.methods.is_gradient_based():
            self.run_cold_kernel_epoch(owner.dataset)
        else:
            with attached_activation_extractor(owner.activation_extractor), torch.inference_mode():
                self.run_standard_epoch(owner.dataset)

        if hasattr(owner.vector_update_strategy, "finalize"):
            owner.vector_update_strategy.finalize()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self.components.build_steering_strategy()
        if hasattr(owner.steering_strategy, "target_layers"):
            if set(owner.steering_strategy.target_layers) != set(owner.layers_to_extract):
                raise RuntimeError(
                    f"Strategy layers {owner.steering_strategy.target_layers} != requested {owner.layers_to_extract}"
                )

        assert_no_forward_hooks(owner.model, where="(Executor.representation_extractor: before return)")
        wrapper = self.components.make_runtime_wrapper()

        if owner._free_training_artifacts_after_build:
            owner._free_training_artifacts()

        return wrapper
