<p align="center">
  <img src="https://raw.githubusercontent.com/mattiapiazzalunga/pysteer/master/images/logo.png" alt="pysteer Python activation steering library logo" width="130">
</p>
<p align="center"><em>Python activation steering for LLMs and transformer language models.</em></p>

<p align="center">
  <a href="https://pypi.org/project/pysteer-adaptation/">
    <img src="https://img.shields.io/pypi/v/pysteer-adaptation?label=PyPI&logo=pypi" alt="pysteer-adaptation package on PyPI"/>
  </a>
  <a href="https://pypi.org/project/pysteer-adaptation/">
    <img src="https://img.shields.io/pypi/pyversions/pysteer-adaptation?logo=python&logoColor=white" alt="pysteer supported Python versions"/>
  </a>
  <a href="https://opensource.org/licenses/MPL-2.0">
    <img src="https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg" alt="MPL 2.0 license"/>
  </a>
  <a href="https://github.com/mattiapiazzalunga/pysteer/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/mattiapiazzalunga/pysteer/ci.yml?label=CI&logo=github" alt="pysteer continuous integration status"/>
  </a>
  <a href="https://github.com/mattiapiazzalunga/pysteer/issues">
    <img src="https://img.shields.io/github/issues/mattiapiazzalunga/pysteer?logo=github" alt="pysteer GitHub issues"/>
  </a>
</p>

# pysteer: Python Activation Steering for LLMs

`pysteer` is a lightweight Python library for activation steering,
representation engineering, and inference-time model steering in PyTorch
transformer language models. It learns steering artifacts from labeled
prompt/response examples, then applies interventions to intermediate
activations without fine-tuning or modifying model weights.

The package is designed for researchers and developers working on LLM control,
mechanistic interpretability, AI safety experiments, and activation engineering
workflows with Hugging Face-style models.

- PyPI package: <https://pypi.org/project/pysteer-adaptation/>
- Documentation: <https://mattiapiazzalunga.github.io/pysteer/>
- Source code: <https://github.com/mattiapiazzalunga/pysteer>
- Issues: <https://github.com/mattiapiazzalunga/pysteer/issues>

## Why Use pysteer

- Steer LLM behavior at inference time without retraining the model.
- Compare multiple activation-steering methods behind one `Executor` API.
- Build contrastive, angular, or gradient-derived steering workflows.
- Extend the steering engine with custom derivation and runtime strategies.
- Keep activation hooks scoped with a context-managed runtime wrapper.

## Features

- Training-time activation extraction from selected transformer layers.
- Five built-in implementations exposed through six method IDs: CMD, CPCA,
  MBS-CMD, Angular Steering, and COLD-Kernel; `cold_steer` is an alias of
  `cold_kernel`.
- A registry-based extension layer for adding new derivation/runtime methods
  without editing `Executor`.
- A context-managed runtime wrapper that keeps steering hooks scoped to the
  calls where they are intended.
- Sphinx documentation with autodoc, Napoleon docstrings, API reference pages,
  and an `open` target.

## Installation

Install from PyPI:

```bash
python -m pip install pysteer-adaptation
```

Install from a local checkout for development:

```bash
python -m pip install -e ".[dev,docs]"
```

Install only the runtime dependencies when working from source without an
editable install:

```bash
python -m pip install -r REQUIREMENTS.txt
```

Install documentation dependencies only when building the docs:

```bash
python -m pip install -r docs/requirements.txt
```

## Quick Start

The distribution is named `pysteer-adaptation`, while the Python import package
is named `pysteer`. The core entry point is `pysteer.Executor`.

The example below downloads a small instruction model, learns CMD steering
vectors, and applies them while generating text.

```python
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pysteer import Executor

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

train_df = pd.DataFrame(
    [
        {"prompt": "Question", "response": "Helpful answer", "reference": 1},
        {"prompt": "Question", "response": "Unhelpful answer", "reference": 0},
    ]
)

executor = Executor(
    model=model,
    tokenizer=tokenizer,
    train_df=train_df,
    method="cmd",
    layers_to_extract=[8, 12],
    dtype=torch.float32,
    alpha=0.5,
)

wrapper = executor.representation_extractor()
inputs = tokenizer("Question", return_tensors="pt")

with wrapper as steered_model:
    output_ids = steered_model.generate(**inputs, max_new_tokens=32)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

Model downloads and generation behavior depend on the selected Hugging Face
model. Choose `layers_to_extract` indexes that exist in that model.

## Training Data

All built-in methods currently use `TaskTypeEnum.UNSUPERVISED`; other task modes
raise `NotImplementedError`. Every row must contain:

- `prompt`: the input text.
- `response`: the candidate response whose activations are analyzed.
- `reference`: exactly `1` for a desired response or `0` for an undesired one.

Training rows are validated before hooks are attached. `reference` must contain
only `0` and `1`, and every contrastive training scope needs at least one
positive and one negative row. For standard methods the scope is the full
dataframe; MBS-CMD validates each selected `mbs_layer`.

`mbs_cmd` additionally requires an integer-like `mbs_layer` value on every row.
Every layer in `layers_to_extract` must have at least one positive and one
negative row, and `mbs_layer` values must belong to `layers_to_extract`.

For example, a two-layer MBS-CMD dataset and per-layer strengths can be defined
as follows:

```python
mbs_df = pd.DataFrame(
    [
        {"prompt": "Q1", "response": "Desired A1", "reference": 1, "mbs_layer": 8},
        {"prompt": "Q1", "response": "Undesired A1", "reference": 0, "mbs_layer": 8},
        {"prompt": "Q2", "response": "Desired A2", "reference": 1, "mbs_layer": 12},
        {"prompt": "Q2", "response": "Undesired A2", "reference": 0, "mbs_layer": 12},
    ]
)

mbs_executor = Executor(
    model=model,
    tokenizer=tokenizer,
    train_df=mbs_df,
    method="mbs_cmd",
    layers_to_extract=[8, 12],
    alpha={8: 0.4, 12: 0.7},
    dtype=torch.float32,
)
```

## Supported Steering Methods

`pysteer` ships with five implementations and six public method IDs. They are
independent adaptations of the cited ideas, not bit-for-bit reproductions of
the authors' reference code. The same links are available programmatically in
`build_default_method_registry().method_specs()[method_id].sources`.

| Method | What this library implements | Origin | `alpha` |
| --- | --- | --- | --- |
| `cmd` | Per-layer desired-minus-undesired response-activation mean; the unit direction is added at runtime. | Adapted from DiffMean/Contrastive Activation Addition ([Rimsky et al., ACL 2024](https://aclanthology.org/2024.acl-long.828/)). Unlike paired CAA datasets, rows only need global class balance. | Scalar additive strength. |
| `cpca` | First principal direction of the pooled activations centered on the midpoint of the two class means, with its sign oriented toward the desired class; the unit direction is added at runtime. | A local PCA variant inspired by representation reading in [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405), not the standard statistical cPCA algorithm or an exact paper reproduction. | Scalar additive strength. |
| `mbs_cmd` | CMD computed independently at each layer from rows whose `mbs_layer` equals that layer. | A pysteer-specific layer-routed extension of `cmd`; there is no separate external paper. | One scalar for all layers, or a complete `{layer_index: strength}` dictionary. |
| `angular` | A shared two-dimensional plane is derived across the selected layers. Positively gated residual-stream components are replaced by an equal-norm component at the target angle. | Adapted from [Angular Steering (Vu and Nguyen, 2025)](https://arxiv.org/abs/2510.26243). pysteer intervenes on transformer-block residual outputs and uses its own dataframe-based plane derivation. | Target angle in degrees, as one scalar or a complete per-layer dictionary. |
| `cold_kernel` | Signed response cross-entropy gradients are averaged into a normalized, static negative-gradient direction and added at runtime. | Adapted from the unit-kernel idea in [COLD-Steer (Sharma and Trivedi, ICLR 2026)](https://arxiv.org/abs/2603.06495). The general kernel and finite-difference variants are not implemented. | Scalar additive strength. |
| `cold_steer` | Exact alias of `cold_kernel`: same factory, training path, and runtime behavior. | Same COLD-Steer source above; this ID does not expose the full family described in the paper. | Scalar additive strength. |

CMD, CPCA, Angular Steering, and both COLD identifiers require global positive
and negative class balance. MBS-CMD requires that balance independently for
every selected layer.

COLD derivation performs backward passes through the model and therefore uses
more memory than the activation-only methods. It clears gradients after each
batch and does not update model parameters. Runtime token selection remains
controlled by `apply_from_mode` and `tokens_window`; it is not forced to match
the paper's evaluation setup.

## Important Executor Options

| Option | Meaning |
| --- | --- |
| `layers_to_extract` | Integer layer index or iterable of layer indexes to train and steer. |
| `alpha` | Steering strength, except for `angular`, where it is a target angle in degrees. MBS-CMD and Angular Steering also accept complete per-layer dictionaries. |
| `tokens_window` | Maximum number of response or decoding tokens to steer. Any non-positive value means the full eligible span. |
| `apply_from_mode` | `ApplyFromModeEnum.PROMPT_END` starts after the prompt; `FIXED` uses `apply_from_token`. |
| `apply_from_token` | Zero-based fixed start token used when `apply_from_mode` is `FIXED`; it must be non-negative in that mode. |
| `use_last_token_for_response` | If `True`, derive response representations from the last response token; otherwise use a masked mean. |
| `dtype` | Storage dtype for runtime steering tensors. Use a dtype supported by the model and device. |
| `generation_kwargs` | Default keyword arguments merged into wrapper `generate` calls. |
| `free_training_artifacts_after_build` | Release training-only extractors, mediators, and datasets after the runtime wrapper is built. |

Constructing `Executor` validates the configuration and prepares the training
components. Calling `representation_extractor()` performs derivation and returns
a `SteeredModelWrapper`. Entering the wrapper context attaches inference hooks;
leaving it removes them, so baseline model calls are not unintentionally
steered.

Current runtime boundaries are intentionally explicit:

- The model must be a causal decoder-style PyTorch model whose transformer
  blocks can be discovered by `ModelUtils`.
- The wrapper must be used as a context manager for both `forward` and
  `generate`.
- Beam search is not supported: `generate` requires `num_beams=1` because beam
  mask reordering is not implemented.
- A non-positive `tokens_window` means every eligible token from the selected
  start; positive values limit the intervention to that many tokens.
- Only `TaskTypeEnum.UNSUPERVISED` is implemented by the built-in methods.

## Architecture

The library separates steering into four concerns:

- Derivation: how an artifact is learned from activations.
- Artifact: the vector, plane, gradient-derived direction, or custom object produced.
- Site: where the artifact reads or writes model state.
- Runtime policy: when and how the intervention is applied.

The `steering_engine` package contains the extension API:

- `domain.py` defines declarative data structures such as `ActivationSite`,
  `InterventionSpec`, `SteeringArtifact`, and `SteeringMethodSpec`.
- `components.py` defines protocols for readers, derivers, runtime strategies,
  schedules, controllers, and compilers.
- `registry.py` provides `SteeringMethodRegistry` and `MethodDefinition`.
- `defaults.py` registers the built-in methods.

See `docs/activation_steering_architecture.md` for the design rationale and
taxonomy.

## Extending Methods

Register a new method with a vector factory and a runtime strategy builder:

```python
from pysteer import Executor
from steering_engine import MethodDefinition, SteeringMethodRegistry
from steering_engine.domain import DerivationFamily, InterventionKind
from steering_engine.domain import RuntimeFamily, SteeringMethodSpec

registry = SteeringMethodRegistry()
registry.register(
    MethodDefinition(
        spec=SteeringMethodSpec(
            method_id="my_method",
            label="My Method",
            derivation_family=DerivationFamily.CUSTOM,
            runtime_family=RuntimeFamily.STATIC,
            intervention_kind=InterventionKind.ADD,
        ),
        vector_factory=lambda ctx: MyVectorDeriver(...),
        strategy_builder=lambda deriver, ctx: MyRuntimeStrategy(...),
    )
)

executor = Executor(
    model=model,
    tokenizer=tokenizer,
    train_df=train_df,
    method="my_method",
    method_registry=registry,
    layers_to_extract=[8, 12],
)
```

Custom vector derivers must implement the update/finalize contract expected by
the executor, and custom runtime strategies must implement the steering
strategy protocol. See `docs/activation_steering_architecture.md` for the full
extension contracts and lifecycle.

## Documentation

Build the Sphinx HTML documentation:

```bash
make -C docs html
```

Build and open it in your default browser:

```bash
make -C docs open
```

On Windows without `make`:

```powershell
docs\make.bat html
docs\make.bat open
```

The generated site is written to `docs/_build/html/index.html`.

## Contributing

See `CONTRIBUTING.md` for development setup, local checks, and the preferred
extension path for new steering methods. Security reports should follow
`SECURITY.md`.

## Evaluation Data

`pysteer` focuses on the generic steering engine and expects callers to provide
their own training dataframes for application-specific evaluations.

## License

This project is licensed under the Mozilla Public License 2.0. See
`LICENSE.txt`.
