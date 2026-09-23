"""Structural protocols and context objects for extensible steering components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from steering_engine.domain import ActivationSite
from steering_engine.domain import AlphaValue
from steering_engine.domain import InterventionSpec
from steering_engine.domain import SteeringArtifact
from steering_engine.domain import SteeringPlan


@dataclass(frozen=True)
class DerivationContext:
    """Context handed to vector/artifact derivation factories."""

    model: nn.Module
    tokenizer: Any
    layers: Sequence[int]
    hidden_size: int
    dtype: torch.dtype
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompileContext:
    """Context handed to runtime strategy builders."""

    model: nn.Module
    tokenizer: Any
    requested_layers: Sequence[int]
    dtype: torch.dtype
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeContext:
    """Context handed to runtime wrapper factories."""

    model: nn.Module
    tokenizer: Any
    strategy: Any
    alpha: AlphaValue
    apply_from_token: int
    apply_from_mode: Any
    tokens_window: int
    padding_side: str
    dtype: torch.dtype
    generation_kwargs: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ActivationReader(Protocol):
    """Port for collecting activations from any backend."""

    def attach(self) -> None:
        """Attach hooks to the configured model layers."""
        ...

    def finalize(self, *, clear_chunks: bool = True) -> Mapping[int, torch.Tensor]:
        """Finalize accumulated state into reusable artifacts."""
        ...

    def remove(self) -> None:
        """Remove hooks and clear accumulated state."""
        ...


@runtime_checkable
class SteeringArtifactDeriver(Protocol):
    """Port for pre-inference derivation methods."""

    def init_layers(self, layers: Sequence[int], hidden_size: int) -> None:
        """Initialize per-layer storage for a hidden size."""
        ...

    def reset(self) -> None:
        """Reset accumulated state while keeping configuration."""
        ...

    def finalize(self) -> None:
        """Finalize accumulated state into reusable artifacts."""
        ...


@runtime_checkable
class ArtifactExporter(Protocol):
    """Optional richer export protocol for new derivation methods."""

    def export_artifact(self) -> SteeringArtifact:
        """Export artifact."""
        ...


@runtime_checkable
class RuntimeSteeringStrategy(Protocol):
    """Port for runtime activation edits."""

    @property
    def target_layers(self) -> Sequence[int]:
        """Target layers."""
        ...

    def steer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        alpha: Any,
        mask_tok: torch.Tensor,
    ) -> torch.Tensor:
        """Apply this strategy to selected hidden-state tokens."""
        ...


@runtime_checkable
class PromptAwareRuntimeStrategy(RuntimeSteeringStrategy, Protocol):
    """Runtime strategy that also observes prompt activations for routing."""

    def ingest_prompt(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> None:
        """Observe prompt activations used by routed runtime strategies."""
        ...

    def reset_prompt_state(self) -> None:
        """Clear cached prompt-routing state."""
        ...


@runtime_checkable
class InterventionSchedule(Protocol):
    """Policy object for alpha/layer/token schedules."""

    def alpha_for(
        self,
        *,
        layer_idx: int,
        token_index: torch.Tensor,
        base_alpha: Any,
        metadata: Mapping[str, Any],
    ) -> torch.Tensor:
        """Alpha for."""
        ...


@runtime_checkable
class ActivationSiteAdapter(Protocol):
    """Maps an ActivationSite to backend-specific hook points."""

    def supports(self, site: ActivationSite) -> bool:
        """Supports."""
        ...

    def register(self, model: nn.Module, site: ActivationSite, callback: Any) -> Any:
        """Register a steering method definition."""
        ...


@runtime_checkable
class SteeringPlanCompiler(Protocol):
    """Compiles declarative plans into runtime backend objects."""

    def compile(self, plan: SteeringPlan, context: CompileContext) -> RuntimeSteeringStrategy:
        """Compile."""
        ...


@runtime_checkable
class SteeringController(Protocol):
    """Stateful controller for adaptive or feedback-driven steering."""

    def reset(self) -> None:
        """Reset accumulated state while keeping configuration."""
        ...

    def observe(self, *, site: ActivationSite, hidden_states: torch.Tensor, metadata: Mapping[str, Any]) -> None:
        """Observe."""
        ...

    def intervention_for(self, spec: InterventionSpec, metadata: Mapping[str, Any]) -> InterventionSpec:
        """Intervention for."""
        ...
