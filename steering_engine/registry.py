"""Registry and abstract factory for steering method components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from steering_engine.components import CompileContext
from steering_engine.components import DerivationContext
from steering_engine.components import RuntimeContext
from steering_engine.domain import SteeringMethodSpec


VectorFactory = Callable[[DerivationContext], Any]
StrategyBuilder = Callable[[Any, CompileContext], Any]
RuntimeWrapperFactory = Callable[[RuntimeContext], Any]
LayerPolicy = Callable[[Sequence[int], Mapping[str, Any]], Sequence[int]]
TrainingRunner = Callable[[Any, Any], None]


def resolve_method_id(method: Any) -> str:
    """Normalize enum/string method IDs for registry lookup."""

    if isinstance(method, Enum):
        return str(method.value)
    if hasattr(method, "value"):
        return str(getattr(method, "value"))
    return str(method)


def identity_layer_policy(layers: Sequence[int], _options: Mapping[str, Any]) -> Sequence[int]:
    """Return the requested layers in deterministic integer order."""
    return sorted({int(x) for x in layers})


@dataclass(frozen=True)
class MethodDefinition:
    """Abstract factory entry for a steering method."""

    spec: SteeringMethodSpec
    vector_factory: VectorFactory
    strategy_builder: StrategyBuilder
    training_layer_policy: LayerPolicy = identity_layer_policy
    training_runner: Optional[TrainingRunner] = None
    runtime_wrapper_factory: Optional[RuntimeWrapperFactory] = None

    def __post_init__(self) -> None:
        """Validate dataclass fields after initialization."""
        if not isinstance(self.spec, SteeringMethodSpec):
            raise TypeError("MethodDefinition.spec must be a SteeringMethodSpec.")
        if not self.spec.method_id:
            raise ValueError("MethodDefinition.spec.method_id must be non-empty.")
        if not callable(self.vector_factory):
            raise TypeError("MethodDefinition.vector_factory must be callable.")
        if not callable(self.strategy_builder):
            raise TypeError("MethodDefinition.strategy_builder must be callable.")
        if not callable(self.training_layer_policy):
            raise TypeError("MethodDefinition.training_layer_policy must be callable.")
        if self.training_runner is not None and not callable(self.training_runner):
            raise TypeError("MethodDefinition.training_runner must be callable when provided.")
        if self.runtime_wrapper_factory is not None and not callable(self.runtime_wrapper_factory):
            raise TypeError("MethodDefinition.runtime_wrapper_factory must be callable when provided.")


class SteeringMethodRegistry:
    """Registry plus abstract factory for derivation/runtime pairs.

    The registry is intentionally small: it knows how to instantiate components,
    but it does not know how a component learns or applies steering. That keeps
    algorithmic experiments out of `Executor` and lets future methods ship as
    add-on modules.
    """

    def __init__(self) -> None:
        """Initialize the instance and validate its configuration."""
        self._methods: Dict[str, MethodDefinition] = {}

    def register(self, definition: MethodDefinition, *, replace: bool = False) -> None:
        """Register a steering method definition."""
        if not isinstance(definition, MethodDefinition):
            raise TypeError("definition must be a MethodDefinition.")
        method_id = resolve_method_id(definition.spec.method_id)
        if method_id in self._methods and not replace:
            raise ValueError(f"Steering method already registered: {method_id}")
        self._methods[method_id] = definition

    def register_many(self, definitions: Sequence[MethodDefinition], *, replace: bool = False) -> None:
        """Register multiple steering method definitions."""
        for definition in definitions:
            self.register(definition, replace=replace)

    def merge_from(self, other: "SteeringMethodRegistry", *, replace: bool = True) -> None:
        """Copy definitions from another registry."""
        for definition in other._methods.values():
            self.register(definition, replace=replace)

    def copy(self) -> "SteeringMethodRegistry":
        """Return an independent registry with the same method definitions."""
        out = SteeringMethodRegistry()
        out.merge_from(self, replace=True)
        return out

    def get(self, method: Any) -> MethodDefinition:
        """Return a registered method definition or raise a helpful error."""
        method_id = resolve_method_id(method)
        try:
            return self._methods[method_id]
        except KeyError as e:
            known = ", ".join(sorted(self._methods)) or "<none>"
            raise KeyError(f"Unknown steering method {method_id!r}. Registered methods: {known}") from e

    def try_get(self, method: Any) -> Optional[MethodDefinition]:
        """Return a registered method definition when present."""
        return self._methods.get(resolve_method_id(method))

    def method_specs(self) -> Dict[str, SteeringMethodSpec]:
        """Return discoverable method metadata keyed by method id."""
        return {k: v.spec for k, v in self._methods.items()}

    def training_layers_for(
        self,
        method: Any,
        layers: Sequence[int],
        options: Mapping[str, Any],
    ) -> Sequence[int]:
        """Return the layers required while training a method."""
        return self.get(method).training_layer_policy(layers, options)

    def create_vector_strategy(self, method: Any, context: DerivationContext) -> Any:
        """Create the training-time vector strategy for a method."""
        return self.get(method).vector_factory(context)

    def build_runtime_strategy(self, method: Any, update_strategy: Any, context: CompileContext) -> Any:
        """Build the runtime strategy for a trained method."""
        return self.get(method).strategy_builder(update_strategy, context)

    def build_runtime_wrapper(
        self,
        method: Any,
        context: RuntimeContext,
        *,
        default_factory: RuntimeWrapperFactory,
    ) -> Any:
        """Build the runtime wrapper for a strategy."""
        factory = self.get(method).runtime_wrapper_factory or default_factory
        return factory(context)

    def __contains__(self, method: Any) -> bool:
        """Return whether the registry contains a method id."""
        return resolve_method_id(method) in self._methods
