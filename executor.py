"""Public facade for training activation steering artifacts and wrapping models at inference time."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from activation_manager.ActivationExtractor import ActivationExtractor
from activation_manager.VectorMediator import VectorMediator
from enums.ApplyFromModeEnum import ApplyFromModeEnum
from enums.TaskTypeEnum import TaskTypeEnum
from steering_strategy.BaseSteeringStrategy import AlphaSpec
from steering_strategy.BaseSteeringStrategy import BaseSteeringStrategy
from steering_engine.defaults import build_default_method_registry
from steering_engine.executor_services import ExecutorComponentBuilder
from steering_engine.executor_services import ExecutorDatasetBuilder
from steering_engine.executor_services import ExecutorMethodContext
from steering_engine.executor_services import ExecutorTokenizerIO
from steering_engine.executor_services import ExecutorTrainingOrchestrator
from steering_engine.registry import SteeringMethodRegistry
from steering_engine.registry import resolve_method_id
from utils.ModelUtils import ModelUtils
from vector_update_strategy.BaseVectorUpdateStrategy import BaseVectorUpdateStrategy

if TYPE_CHECKING:
    import pandas as pd
else:
    try:
        import pandas as pd
    except ImportError:
        pd = None


def _require_pandas():
    """Return pandas or raise an installation-focused error."""
    if pd is None:
        raise RuntimeError(
            "pysteer.Executor requires pandas. Install the runtime dependencies with "
            "`python -m pip install pysteer-adaptation`, or install pandas in this environment."
        )
    return pd


class Executor:
    """Train activation-steering artifacts and produce an inference wrapper.

    ``Executor`` keeps the stable public API for pysteer. It validates the
    training dataframe, extracts activations from selected transformer layers,
    delegates artifact derivation to the selected method, and returns a runtime
    wrapper from :meth:`representation_extractor`.
    """
    def __init__(
            self,
            model: nn.Module,
            tokenizer,
            train_df: pd.DataFrame,
            method: str = "cmd",
            layers_to_extract: Union[Iterable[int], int] = (0,),
            batch_size: int = 4,
            padding_side: Optional[str] = None,
            truncation_side: Optional[str] = None,
            dtype: torch.dtype = torch.float16,
            alpha: AlphaSpec = 0.5,
            generation_kwargs: Optional[Dict[str, Any]] = None,
            *,
            tokens_window: int = -1,
            method_registry: Optional[SteeringMethodRegistry] = None,
            task_type: TaskTypeEnum = TaskTypeEnum.UNSUPERVISED,
            apply_from_token: int = -1,
            apply_from_mode: ApplyFromModeEnum = ApplyFromModeEnum.PROMPT_END,
            use_last_token_for_response: bool = True,
            decode_chunk_max: int = 1,
            free_training_artifacts_after_build: bool = True,
    ) -> None:
        """Initialize the executor and prepare training components.

        Args:
            model: Causal decoder-style PyTorch model to steer.
            tokenizer: Tokenizer compatible with ``model``.
            train_df: Training rows. Built-in methods expect at least
                ``prompt``, ``response``, and ``reference`` columns.
            method: Built-in or registered steering method id.
            layers_to_extract: Transformer layer index or indexes to steer.
            batch_size: Number of dataframe rows processed per training batch.
            padding_side: Temporary tokenizer padding side, defaulting to the
                tokenizer configuration.
            truncation_side: Temporary tokenizer truncation side, defaulting to
                the tokenizer configuration.
            dtype: Runtime steering tensor dtype.
            alpha: Scalar steering strength, or method-specific per-id mapping.
            generation_kwargs: Default keyword arguments merged into wrapper
                ``generate`` calls.
            tokens_window: Number of response/decode tokens to steer; any
                non-positive value steers the full eligible span.
            method_registry: Optional custom registry for extension methods.
            task_type: Training task mode. Currently only unsupervised rows are
                implemented.
            apply_from_token: Fixed application token used when
                ``apply_from_mode`` is ``FIXED``.
            apply_from_mode: Policy for computing the first steered token.
            use_last_token_for_response: Whether derivation methods summarize
                responses with their last token instead of a mean.
            decode_chunk_max: Maximum decode-step chunk size expected by the
                activation extractor.
            free_training_artifacts_after_build: Free training-only tensors once
                the runtime wrapper has been built.

        Raises:
            RuntimeError: If transformer layers cannot be discovered.
            ValueError: If the dataframe or steering configuration is invalid.
        """
        self._generation_kwargs: Dict[str, Any] = {}
        self._vector_update_strategy: Optional[BaseVectorUpdateStrategy] = None
        self._activation_extractor: Optional[ActivationExtractor] = None
        self._vector_mediator: Optional[VectorMediator] = None
        self._steering_strategy: Optional[BaseSteeringStrategy] = None
        self.method_registry: SteeringMethodRegistry = method_registry or build_default_method_registry()
        self._method_context = ExecutorMethodContext(self)
        self._tokenizer_io = ExecutorTokenizerIO(self)
        self._dataset_builder = ExecutorDatasetBuilder(self, self._method_context)
        self._component_builder = ExecutorComponentBuilder(self, self._method_context)
        self._training_orchestrator = ExecutorTrainingOrchestrator(
            self,
            self._method_context,
            self._tokenizer_io,
            self._component_builder,
        )
        n_layers = len(ModelUtils.find_transformer_layers(model))
        if n_layers <= 0:
            raise RuntimeError("Cannot initialize steering: no transformer layers found in model.")

        self.generation_kwargs = generation_kwargs or {}

        self.use_last_token_for_response: bool = bool(use_last_token_for_response)

        self.tokenizer = tokenizer
        self.padding_side = (
            getattr(tokenizer, "padding_side", "left")
            if padding_side is None
            else padding_side
        ).lower()
        self.truncation_side = (
            getattr(tokenizer, "truncation_side", "left") if truncation_side is None else truncation_side
        ).lower()

        self.model = model
        self._ensure_tokenizer_has_pad_token()

        self.layers_to_extract = layers_to_extract
        self.batch_size = batch_size
        self.alpha = alpha
        self.apply_from_token = apply_from_token
        self.task_type = task_type
        self.apply_from_mode = apply_from_mode
        self.method = method
        self.dtype = dtype
        self.tokens_window = int(tokens_window)

        pd_module = _require_pandas()
        self.train_df: pd.DataFrame = pd_module.DataFrame(train_df)
        self.max_prompt_tokens: int = 2048
        self.decode_chunk_max: int = max(1, int(decode_chunk_max))

        self.dataset = self.prepare_dataset_to_train(self.train_df)
        self._setup_vector_components()
        self._free_training_artifacts_after_build: bool = bool(free_training_artifacts_after_build)

    def _ensure_tokenizer_has_pad_token(self) -> None:
        """Ensure tokenizer has pad token."""
        self._tokenizer_io.ensure_pad_token()

    def _with_tokenizer_sides(self):
        """With tokenizer sides."""
        return self._tokenizer_io.tokenizer_sides()

    def _free_training_artifacts(self) -> None:
        """Free training artifacts."""
        try:
            if self.activation_extractor is not None:
                self.activation_extractor.remove()
        except Exception:
            pass
        self.activation_extractor = None
        self.vector_mediator = None
        self.vector_update_strategy = None
        self.dataset = None

    def _method_definition(self):
        """Method definition."""
        return self._method_context.definition()

    def _method_options(self, *, n_layers: Optional[int] = None) -> Dict[str, Any]:
        """Method options."""
        return self._method_context.options(n_layers=n_layers)

    def _cold_enabled(self) -> bool:
        """Cold enabled."""
        return self._method_context.is_gradient_based()

    def _training_layers_to_extract(self) -> List[int]:
        """Training layers to extract."""
        return self._method_context.training_layers()

    def _require_activations(self, acts_full: Dict[int, torch.Tensor]) -> None:
        """Validate the required activations state."""
        self._method_context.require_activations(acts_full)

    def _require_group_ids_if_needed(self, group_ids: Optional[torch.Tensor]) -> None:
        """Validate the required group ids if needed state."""
        self._method_context.require_group_ids_if_needed(group_ids)

    def _encode_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode prompts values."""
        return self._tokenizer_io.encode_prompts(prompts)

    def _pad_from_ids(self, id_lists: List[List[int]]) -> Dict[str, torch.Tensor]:
        """Pad from ids values."""
        return self._tokenizer_io.pad_from_ids(id_lists)

    @staticmethod
    def _is_missing_response_value(v: Any) -> bool:
        """Return whether missing response value holds."""
        return ExecutorTokenizerIO.is_missing_response_value(v)

    def _trim_generated_tokens(self, gen_ids: List[int]) -> List[int]:
        """Trim generated tokens."""
        return self._tokenizer_io.trim_generated_tokens(gen_ids)

    @staticmethod
    def _as_torch_device(v) -> torch.device:
        """As torch device."""
        return ExecutorTokenizerIO.as_torch_device(v)

    @staticmethod
    def _infer_input_device(model: nn.Module) -> torch.device:
        """Infer input device values."""
        return ExecutorTokenizerIO.infer_input_device(model)

    @property
    def model(self) -> nn.Module:
        """Model."""
        return self._model

    @model.setter
    def model(self, m: nn.Module) -> None:
        """Set the model value after validation."""
        if not isinstance(m, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        self._model = m

    @property
    def truncation_side(self) -> str:
        """Truncation side."""
        return self._truncation_side

    @truncation_side.setter
    def truncation_side(self, value: str) -> None:
        """Set the truncation_side value after validation."""
        value = str(value).lower()
        if value not in ("left", "right"):
            raise ValueError("truncation_side must be 'left' or 'right'")
        self._truncation_side = value

    @property
    def tokenizer(self):
        """Tokenizer."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tok):
        """Set the tokenizer value after validation."""
        self._tokenizer = tok

    @property
    def padding_side(self) -> str:
        """Padding side."""
        return self._padding_side

    @padding_side.setter
    def padding_side(self, value: str) -> None:
        """Set the padding_side value after validation."""
        value = str(value).lower()
        if value not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        self._padding_side = value

    @property
    def layers_to_extract(self) -> List[int]:
        """Layers to extract."""
        return self._layers_to_extract

    @layers_to_extract.setter
    def layers_to_extract(self, value: Union[Iterable[int], int]) -> None:
        """Set the layers_to_extract value after validation."""
        if isinstance(value, int):
            value = [value]
        cleaned = sorted({int(i) for i in value})
        if not cleaned:
            raise ValueError("At least one layer must be specified for extraction")
        self._layers_to_extract = cleaned

        n = len(ModelUtils.find_transformer_layers(self.model))
        bad = [i for i in cleaned if i < 0 or i >= n]
        if bad:
            raise IndexError(f"Layer index(es) {bad} out of range for model with {n} layers.")

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size: int) -> None:
        """Set the batch_size value after validation."""
        self._batch_size = max(1, int(size))

    @property
    def dtype(self) -> torch.dtype:
        """Dtype."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        """Set the dtype value after validation."""
        self._dtype = ModelUtils.validate_steering_dtype(value)

    @property
    def alpha(self) -> AlphaSpec:
        """Alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, a: AlphaSpec) -> None:
        """Set the alpha value after validation."""
        def finite_alpha(value: Any) -> float:
            out = float(value)
            if not math.isfinite(out):
                raise ValueError("alpha values must be finite.")
            return out

        if isinstance(a, dict):
            self._alpha = {int(k): finite_alpha(v) for k, v in a.items()}
        else:
            self._alpha = finite_alpha(a)

    def _validate_apply_from_configuration(self) -> None:
        """Validate apply from configuration configuration."""
        mode = getattr(self, "_apply_from_mode", None)
        token = getattr(self, "_apply_from_token", None)

        if mode == ApplyFromModeEnum.FIXED and token is not None and int(token) < 0:
            raise ValueError(
                "apply_from_token must be >= 0 when apply_from_mode is ApplyFromModeEnum.FIXED."
            )

    @property
    def apply_from_token(self) -> int:
        """Apply from token."""
        return self._apply_from_token

    @apply_from_token.setter
    def apply_from_token(self, ind: int) -> None:
        """Set the apply_from_token value after validation."""
        self._apply_from_token = int(ind)

        if hasattr(self, "_apply_from_mode"):
            self._validate_apply_from_configuration()

    @property
    def apply_from_mode(self) -> ApplyFromModeEnum:
        """Apply from mode."""
        return self._apply_from_mode

    @apply_from_mode.setter
    def apply_from_mode(self, mode: ApplyFromModeEnum) -> None:
        """Set the apply_from_mode value after validation."""
        if not isinstance(mode, ApplyFromModeEnum):
            raise TypeError("apply_from_mode must be ApplyFromModeEnum")
        self._apply_from_mode = mode

        if hasattr(self, "_apply_from_token"):
            self._validate_apply_from_configuration()

    @property
    def task_type(self) -> TaskTypeEnum:
        """Task type."""
        return self._task_type

    @task_type.setter
    def task_type(self, t: TaskTypeEnum) -> None:
        """Set the task_type value after validation."""
        if not isinstance(t, TaskTypeEnum):
            raise TypeError("task_type must be TaskTypeEnum")
        self._task_type = t


    def _cast_strategy_tensor(self, x: Any) -> Any:
        """Cast strategy tensor values as needed."""
        work_dtype = ModelUtils.steering_work_dtype(self.dtype)
        if torch.is_tensor(x) and x.is_floating_point() and x.dtype != work_dtype:
            return x.to(dtype=work_dtype)
        return x

    def _cast_tensor_map(self, tensors: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Cast tensor map values as needed."""
        return {
            int(k): cast(torch.Tensor, self._cast_strategy_tensor(v))
            for k, v in tensors.items()
        }

    def _make_steered_wrapper(self) -> Any:
        """Create steered wrapper helper data."""
        return self._component_builder.make_runtime_wrapper()

    @property
    def method(self) -> str:
        """Registered steering method id."""
        return self._method

    @method.setter
    def method(self, m: Any) -> None:
        """Set the method value after validation."""
        method_id = resolve_method_id(m).strip()
        if not method_id:
            raise ValueError("method must be a non-empty registered method id.")
        if method_id not in self.method_registry and method_id.lower() in self.method_registry:
            method_id = method_id.lower()
        if method_id not in self.method_registry:
            raise ValueError(f"Unknown method={method_id!r}")
        self._method = method_id

    @property
    def tokens_window(self) -> int:
        """Tokens window."""
        return self._tokens_window

    @tokens_window.setter
    def tokens_window(self, value: int) -> None:
        """Set the tokens_window value after validation."""
        self._tokens_window = int(value)

    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        """Generation kwargs."""
        return self._generation_kwargs

    @generation_kwargs.setter
    def generation_kwargs(self, kw: Dict[str, Any]) -> None:
        """Set the generation_kwargs value after validation."""
        self._generation_kwargs = dict(kw or {})

    @property
    def vector_update_strategy(self) -> Optional[BaseVectorUpdateStrategy]:
        """Vector update strategy."""
        return self._vector_update_strategy

    @vector_update_strategy.setter
    def vector_update_strategy(self, v: Optional[BaseVectorUpdateStrategy]) -> None:
        """Set the vector_update_strategy value after validation."""
        self._vector_update_strategy = v

    @property
    def activation_extractor(self) -> Optional[ActivationExtractor]:
        """Activation extractor."""
        return self._activation_extractor

    @activation_extractor.setter
    def activation_extractor(self, a: Optional[ActivationExtractor]) -> None:
        """Set the activation_extractor value after validation."""
        self._activation_extractor = a

    @property
    def vector_mediator(self) -> Optional[VectorMediator]:
        """Vector mediator."""
        return self._vector_mediator

    @vector_mediator.setter
    def vector_mediator(self, vm: Optional[VectorMediator]) -> None:
        """Set the vector_mediator value after validation."""
        self._vector_mediator = vm

    @property
    def steering_strategy(self) -> Optional[BaseSteeringStrategy]:
        """Steering strategy."""
        return self._steering_strategy

    @steering_strategy.setter
    def steering_strategy(self, rs: Optional[BaseSteeringStrategy]) -> None:
        """Set the steering_strategy value after validation."""
        self._steering_strategy = rs

    def prepare_dataset_to_train(self, df: pd.DataFrame) -> DataLoader:
        """Validate training rows and return the executor data loader."""
        return self._dataset_builder.prepare(df)

    def _setup_vector_components(self) -> None:
        """Setup vector components."""
        self._component_builder.setup_vector_components()

    def _build_steering_strategy(self) -> None:
        """Build steering strategy state."""
        self._component_builder.build_steering_strategy()

    def _run_training_epoch_for_loader(self, loader) -> None:
        """Run the training epoch for loader helper path."""
        return self._training_orchestrator.run_standard_epoch(loader)

    def representation_extractor(self) -> Any:
        """Train steering artifacts when needed and return a runtime wrapper.

        The returned object is usually a ``SteeredModelWrapper``. Use it as a
        context manager while calling ``forward`` or ``generate`` so hooks are
        active only for the intended inference block.

        Returns:
            Runtime wrapper for the configured model and steering strategy.
        """
        return self._training_orchestrator.representation_extractor()


    def _run_cold_kernel_training_epoch_for_loader(self, loader) -> None:
        """Gradient-based COLD-Kernel training path. Do not wrap in inference_mode."""
        return self._training_orchestrator.run_cold_kernel_epoch(loader)
