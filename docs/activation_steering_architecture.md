# Activation Steering Architecture

This library now treats activation steering as four orthogonal choices:

1. **Derivation**: how a steering artifact is obtained before inference.
2. **Artifact**: what is produced: dense vector, plane, probe, routing table, SAE feature, controller state, or composite bundle.
3. **Site**: where the artifact reads/writes: residual stream, attention head, MLP output, SAE latent, KV cache, or logits.
4. **Runtime policy**: when and how it is applied: static alpha, routed alpha, adaptive gating, schedules, token windows, prefill-only, decode-only, or feedback controller.

The code mirrors those axes in `steering_engine/`:

- `domain.py`: declarative vocabulary (`ActivationSite`, `InterventionSpec`, `SteeringArtifact`, `SteeringMethodSpec`).
- `components.py`: ports for derivation, runtime strategies, site adapters, schedules, controllers, and plan compilers.
- `registry.py`: a small abstract factory and registry.
- `defaults.py`: built-in method registrations.
- `runtime.py`: default residual-stream wrapper factory.
- `executor_services.py`: focused services used by the public `Executor` facade.

`Executor` accepts a registered `method` id and component selection goes through `SteeringMethodRegistry`. Existing methods remain available as default registry entries. Future methods can be added by registering a `MethodDefinition` instead of editing `Executor`.

## Method Families

Primary literature and systems work suggest these families are worth supporting:

- **Contrastive additions**: ActAdd and CAA derive dense directions from positive/negative or paired prompts and add them during the forward pass.
- **Representation engineering**: RepE generalizes reading and controlling high-level internal representations.
- **Probe/head interventions**: ITI finds directions, often on selected attention heads, using probe-like truthfulness signals.
- **Contrastive subspaces**: PCA/cPCA and angular methods derive planes or subspaces, then add, rotate, or project.
- **Routed steering**: task or prompt centroids select one of many vectors at runtime.
- **Adaptive steering**: probes, gates, classifiers, or controllers decide whether and how strongly to steer per token/batch item.
- **Gradient and finite-difference steering**: directions come from gradients or black-box perturbation estimates.
- **Sparse feature steering**: SAE/dictionary features expose interpretable latent coordinates that can be amplified, suppressed, or combined.
- **Composable steering**: multiple artifacts run together with conflict policies, schedules, and budget limits.
- **Generated steering**: hypernetworks or retrieval systems produce vectors from a steering instruction or prompt.

The important design decision: none of those requires a new `Executor` branch. They require new components.

## Extension Pattern

Register a method with:

```python
from steering_engine.domain import DerivationFamily, InterventionKind, RuntimeFamily, SteeringMethodSpec
from steering_engine.registry import MethodDefinition, SteeringMethodRegistry

registry = SteeringMethodRegistry()
registry.register(
    MethodDefinition(
        spec=SteeringMethodSpec(
            method_id="my_sae_feature_method",
            label="My SAE Feature Method",
            derivation_family=DerivationFamily.SPARSE_FEATURE,
            runtime_family=RuntimeFamily.ADAPTIVE,
            intervention_kind=InterventionKind.SCALE_FEATURE,
            required_columns=("prompt", "response", "reference"),
        ),
        vector_factory=lambda ctx: MySaeFeatureDeriver(...),
        strategy_builder=lambda deriver, ctx: MySaeRuntimeStrategy(...),
        runtime_wrapper_factory=lambda ctx: MySaeRuntimeWrapper(...),
    )
)
```

If the default activation-collection loop is enough, implement the same update/finalize contract as the existing vector mediators. If the method needs a different loop, such as SAE encoding, gradient optimization, external probes, or online controllers, provide `MethodDefinition.training_runner`.

If the method can run on transformer-block residual streams, omit `runtime_wrapper_factory` and the registry will use
`build_residual_stream_runtime_wrapper()`, which creates `SteeredModelWrapper`. Methods that steer attention heads,
SAE latents, logits, KV cache, or controller state should provide their own wrapper factory.

## Design Patterns In Play

- **Hexagonal architecture**: model hooks, tensor collection, artifact derivation, runtime steering, and persistence are ports/adapters rather than one class doing everything.
- **Strategy**: derivation and runtime steering remain interchangeable algorithm objects.
- **Abstract factory plus registry**: `MethodDefinition` creates the paired derivation/runtime/wrapper components for a method.
- **Composite**: `SteeringPlan` and `InterventionSpec` can describe multiple simultaneous interventions.
- **Policy object**: token selectors, schedules, gates, and controllers are independent of vector derivation.
- **Facade**: public `Executor` stays stable while specialized services own validation, component assembly, and training loops.

## Sources Used For The Taxonomy

This list motivates the broader extension taxonomy; it is not a list of
features currently implemented by pysteer. For the provenance and exact scope
of each built-in method, see the README's **Supported Steering Methods** table
or the corresponding `SteeringMethodSpec.sources` value.

- ActAdd: https://arxiv.org/abs/2308.10248
- ITI: https://arxiv.org/abs/2306.03341
- RepE: https://arxiv.org/abs/2310.01405
- CAA: https://arxiv.org/abs/2312.06681
- Anthropic SAE feature steering context: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Sparse activation steering examples: https://arxiv.org/abs/2503.00177 and https://arxiv.org/abs/2501.09929
- Hypernetwork-generated steering direction: https://arxiv.org/abs/2506.03292
