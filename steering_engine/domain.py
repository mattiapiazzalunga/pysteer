"""Declarative domain model for steering sites, artifacts, methods, and plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple, Union

import torch


class ActivationSiteKind(str, Enum):
    """Where an intervention observes or writes model state."""

    RESIDUAL_STREAM = "residual_stream"
    ATTENTION_HEAD = "attention_head"
    MLP = "mlp"
    SAE_LATENT = "sae_latent"
    KV_CACHE = "kv_cache"
    LOGITS = "logits"
    CUSTOM = "custom"


class DerivationFamily(str, Enum):
    """How steering artifacts are derived before inference."""

    CONTRASTIVE_MEAN_DIFFERENCE = "contrastive_mean_difference"
    CONTRASTIVE_PCA = "contrastive_pca"
    PROBE_DIRECTION = "probe_direction"
    SPARSE_FEATURE = "sparse_feature"
    GRADIENT = "gradient"
    FINITE_DIFFERENCE = "finite_difference"
    ROUTED = "routed"
    ANGULAR = "angular"
    MANUAL = "manual"
    COMPOSITE = "composite"
    CUSTOM = "custom"


class RuntimeFamily(str, Enum):
    """How artifacts are selected and applied at inference time."""

    STATIC = "static"
    ROUTED = "routed"
    ADAPTIVE = "adaptive"
    CONTROLLER = "controller"
    STREAMING = "streaming"
    COMPOSITE = "composite"
    CUSTOM = "custom"


class InterventionKind(str, Enum):
    """The mathematical shape of the runtime edit."""

    ADD = "add"
    PROJECT = "project"
    ROTATE = "rotate"
    SCALE_FEATURE = "scale_feature"
    LOW_RANK = "low_rank"
    LOGIT_BIAS = "logit_bias"
    CACHE_EDIT = "cache_edit"
    CONTROLLER = "controller"
    CUSTOM = "custom"


class SteeringPhase(str, Enum):
    """Generation phases where an intervention may run."""

    PREFILL = "prefill"
    DECODE = "decode"
    TRAINING = "training"
    ANY = "any"


AlphaValue = Union[float, Mapping[int, float]]


@dataclass(frozen=True)
class ActivationSite:
    """A stable address for a model activation.

    `layer` is intentionally optional: SAE features, logits, cache entries, or
    externally provided controllers may not map cleanly to a transformer block.
    """

    kind: ActivationSiteKind = ActivationSiteKind.RESIDUAL_STREAM
    layer: Optional[int] = None
    module_path: Optional[str] = None
    head_index: Optional[int] = None
    feature_index: Optional[int] = None
    name: Optional[str] = None
    layout: str = "BSH"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenSelectorSpec:
    """Declarative token policy for runtime application."""

    mode: str = "prompt_end"
    start_token: int = -1
    window: int = -1
    phases: Tuple[SteeringPhase, ...] = (SteeringPhase.DECODE,)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterventionSpec:
    """One runtime intervention, independent from how its artifact was learned."""

    kind: InterventionKind
    sites: Tuple[ActivationSite, ...]
    alpha: AlphaValue = 1.0
    token_selector: TokenSelectorSpec = field(default_factory=TokenSelectorSpec)
    schedule_id: Optional[str] = None
    controller_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SteeringArtifact:
    """Serializable output of a derivation method.

    Existing implementations can keep exposing tensors through their current
    strategy APIs. New methods can additionally export this envelope when they
    need richer metadata, multiple sites, routing tables, SAE dictionaries, or
    controller state.
    """

    artifact_id: str
    family: DerivationFamily
    tensors: Mapping[str, Union[torch.Tensor, Mapping[int, torch.Tensor], Any]]
    sites: Tuple[ActivationSite, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass(frozen=True)
class SteeringMethodSpec:
    """Discoverable capabilities for one steering method."""

    method_id: str
    label: str
    derivation_family: DerivationFamily
    runtime_family: RuntimeFamily
    intervention_kind: InterventionKind
    description: str = ""
    required_columns: Tuple[str, ...] = ("prompt", "response", "reference")
    optional_columns: Tuple[str, ...] = ()
    training_requires_grad: bool = False
    supports_incremental_tasks: bool = False
    supports_dynamic_alpha: bool = False
    supports_routing: bool = False
    sources: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SteeringPlan:
    """Compiled high-level plan: method metadata plus runtime interventions."""

    method: SteeringMethodSpec
    interventions: Tuple[InterventionSpec, ...]
    artifact: Optional[SteeringArtifact] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
