"""Default registry entries for built-in steering methods."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from steering_engine.components import CompileContext
from steering_engine.components import DerivationContext
from steering_engine.domain import DerivationFamily
from steering_engine.domain import InterventionKind
from steering_engine.domain import RuntimeFamily
from steering_engine.domain import SteeringMethodSpec
from steering_engine.registry import MethodDefinition
from steering_engine.registry import SteeringMethodRegistry
from vector_update_strategy.AngularVectorMediator import AngularVectorMediator
from vector_update_strategy.ColdKernelGradientMediator import ColdKernelGradientMediator
from vector_update_strategy.CmdVectorMediator import CmdVectorMediator
from vector_update_strategy.CpcaVectorMediator import CpcaVectorMediator
from vector_update_strategy.MbsCmdVectorMediator import MbsCmdVectorMediator


CAA_SOURCE = "https://aclanthology.org/2024.acl-long.828/"
REPE_SOURCE = "https://arxiv.org/abs/2310.01405"
ANGULAR_SOURCE = "https://arxiv.org/abs/2510.26243"
COLD_STEER_SOURCE = "https://arxiv.org/abs/2603.06495"


def _opt(ctx: Any, name: str, default: Any = None) -> Any:
    """Opt."""
    return ctx.options.get(name, default)


def _cast_tensor_map(ctx: CompileContext, value: Dict[int, Any]) -> Dict[int, Any]:
    """Cast tensor map values as needed."""
    caster = ctx.options.get("cast_tensor_map")
    if callable(caster):
        return caster(value)
    return value


def _cmd_factory(ctx: DerivationContext) -> CmdVectorMediator:
    """Create a configured derivation strategy from context."""
    return CmdVectorMediator(
        use_last_token_for_response=bool(_opt(ctx, "use_last_token_for_response", True)),
    )


def _cpca_factory(ctx: DerivationContext) -> CpcaVectorMediator:
    """Create a configured derivation strategy from context."""
    return CpcaVectorMediator(
        use_last_token_for_response=bool(_opt(ctx, "use_last_token_for_response", True)),
    )


def _mbs_factory(ctx: DerivationContext) -> MbsCmdVectorMediator:
    """Create a configured derivation strategy from context."""
    return MbsCmdVectorMediator(
        correctness_threshold=0.5,
        incorrectness_threshold=0.5,
        use_last_token_for_response=bool(_opt(ctx, "use_last_token_for_response", True)),
    )


def _angular_factory(ctx: DerivationContext) -> AngularVectorMediator:
    """Create a configured derivation strategy from context."""
    return AngularVectorMediator(
        use_last_token_for_response=bool(_opt(ctx, "use_last_token_for_response", True)),
    )


def _cold_factory(ctx: DerivationContext) -> ColdKernelGradientMediator:
    """Create a configured derivation strategy from context."""
    return ColdKernelGradientMediator(
        normalize_to_unit_norm=bool(_opt(ctx, "normalize_to_unit_norm", True)),
    )


def _general_strategy_builder(update_strategy: Any, ctx: CompileContext) -> Any:
    """Build a runtime steering strategy from a finalized update strategy."""
    from steering_strategy.GeneralSteeringStrategy import GeneralSteeringStrategy

    correction_vectors = getattr(update_strategy, "correction_vectors", None)
    if not correction_vectors:
        raise ValueError("Correction vectors not available in update strategy")
    return GeneralSteeringStrategy(correction_vectors=_cast_tensor_map(ctx, correction_vectors))


def _mbs_strategy_builder(update_strategy: Any, ctx: CompileContext) -> Any:
    """Build a runtime steering strategy from a finalized update strategy."""
    from steering_strategy.MbsSteeringStrategy import MbsSteeringStrategy

    correction_vectors = getattr(update_strategy, "correction_vectors", None)
    if not correction_vectors:
        raise ValueError("Correction vectors not available in MBS update strategy")
    return MbsSteeringStrategy(correction_vectors=_cast_tensor_map(ctx, correction_vectors))


def _angular_strategy_builder(update_strategy: Any, ctx: CompileContext) -> Any:
    """Build a runtime steering strategy from a finalized update strategy."""
    from steering_strategy.AngularSteeringStrategy import AngularSteeringStrategy

    if not hasattr(update_strategy, "get_angular_params"):
        raise TypeError("Angular strategy builder requires update_strategy.get_angular_params().")
    return AngularSteeringStrategy(
        steering_planes=update_strategy.get_angular_params(),
        adaptive=True,
        gate_direction="first",
        gate_threshold=0.0,
        cpu_store_dtype=ctx.dtype,
    )


def _spec(
    method_id: str,
    label: str,
    derivation: DerivationFamily,
    runtime: RuntimeFamily,
    intervention: InterventionKind,
    description: str,
    *,
    required_columns: Sequence[str] = ("prompt", "response", "reference"),
    optional_columns: Sequence[str] = (),
    training_requires_grad: bool = False,
    supports_incremental_tasks: bool = False,
    supports_dynamic_alpha: bool = False,
    supports_routing: bool = False,
    sources: Sequence[str] = (),
) -> SteeringMethodSpec:
    """Spec."""
    return SteeringMethodSpec(
        method_id=str(method_id),
        label=label,
        derivation_family=derivation,
        runtime_family=runtime,
        intervention_kind=intervention,
        description=description,
        required_columns=tuple(required_columns),
        optional_columns=tuple(optional_columns),
        training_requires_grad=bool(training_requires_grad),
        supports_incremental_tasks=bool(supports_incremental_tasks),
        supports_dynamic_alpha=bool(supports_dynamic_alpha),
        supports_routing=bool(supports_routing),
        sources=tuple(sources),
    )


def build_default_method_registry() -> SteeringMethodRegistry:
    """Create a registry populated with all built-in steering methods."""
    registry = SteeringMethodRegistry()

    registry.register(
        MethodDefinition(
            spec=_spec(
                "cmd",
                "Contrastive Mean Difference",
                DerivationFamily.CONTRASTIVE_MEAN_DIFFERENCE,
                RuntimeFamily.STATIC,
                InterventionKind.ADD,
                "Learns one residual direction per layer from the difference between positive and negative "
                "response-activation means, then adds its unit direction at runtime.",
                sources=(CAA_SOURCE,),
            ),
            vector_factory=_cmd_factory,
            strategy_builder=_general_strategy_builder,
        )
    )
    registry.register(
        MethodDefinition(
            spec=_spec(
                "cpca",
                "PCA Contrastive Direction",
                DerivationFamily.CONTRASTIVE_PCA,
                RuntimeFamily.STATIC,
                InterventionKind.ADD,
                "Learns the leading principal direction of pooled class-midpoint-centered response activations "
                "and orients it toward the positive class.",
                sources=(REPE_SOURCE,),
            ),
            vector_factory=_cpca_factory,
            strategy_builder=_general_strategy_builder,
        )
    )
    registry.register(
        MethodDefinition(
            spec=_spec(
                "mbs_cmd",
                "Layer-Balanced CMD",
                DerivationFamily.CONTRASTIVE_MEAN_DIFFERENCE,
                RuntimeFamily.STATIC,
                InterventionKind.ADD,
                "pysteer extension of CMD that routes each training row to mbs_layer and enforces class balance "
                "inside every selected layer.",
                required_columns=("prompt", "response", "reference", "mbs_layer"),
                sources=(CAA_SOURCE,),
            ),
            vector_factory=_mbs_factory,
            strategy_builder=_mbs_strategy_builder,
        )
    )
    registry.register(
        MethodDefinition(
            spec=_spec(
                "angular",
                "Angular Steering",
                DerivationFamily.ANGULAR,
                RuntimeFamily.ADAPTIVE,
                InterventionKind.ROTATE,
                "Adapts Angular Steering by deriving a shared two-dimensional plane across selected layers and "
                "selectively rotating eligible residual-stream activations to a target angle.",
                supports_dynamic_alpha=True,
                sources=(ANGULAR_SOURCE,),
            ),
            vector_factory=_angular_factory,
            strategy_builder=_angular_strategy_builder,
        )
    )
    registry.register(
        MethodDefinition(
            spec=_spec(
                "cold_kernel",
                "COLD-Kernel",
                DerivationFamily.GRADIENT,
                RuntimeFamily.STATIC,
                InterventionKind.ADD,
                "Adapts unit-kernel COLD-Steer by averaging signed response-loss gradients into a static residual "
                "direction.",
                training_requires_grad=True,
                sources=(COLD_STEER_SOURCE,),
            ),
            vector_factory=_cold_factory,
            strategy_builder=_general_strategy_builder,
        )
    )
    registry.register(
        MethodDefinition(
            spec=_spec(
                "cold_steer",
                "COLD-Steer",
                DerivationFamily.GRADIENT,
                RuntimeFamily.STATIC,
                InterventionKind.ADD,
                "Exact registry alias of cold_kernel; it uses the same derivation and runtime implementation.",
                training_requires_grad=True,
                sources=(COLD_STEER_SOURCE,),
            ),
            vector_factory=_cold_factory,
            strategy_builder=_general_strategy_builder,
        )
    )
    return registry
