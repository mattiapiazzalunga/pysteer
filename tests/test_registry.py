from __future__ import annotations

from enum import Enum

import pytest

from steering_engine.defaults import build_default_method_registry
from steering_engine.domain import DerivationFamily
from steering_engine.domain import InterventionKind
from steering_engine.domain import RuntimeFamily
from steering_engine.domain import SteeringMethodSpec
from steering_engine.registry import MethodDefinition
from steering_engine.registry import SteeringMethodRegistry
from steering_engine.registry import identity_layer_policy
from steering_engine.registry import resolve_method_id


class MethodEnum(Enum):
    CMD = "cmd"


def _definition(method_id: str = "unit") -> MethodDefinition:
    return MethodDefinition(
        spec=SteeringMethodSpec(
            method_id=method_id,
            label="Unit",
            derivation_family=DerivationFamily.CUSTOM,
            runtime_family=RuntimeFamily.CUSTOM,
            intervention_kind=InterventionKind.CUSTOM,
        ),
        vector_factory=lambda ctx: {"ctx": ctx},
        strategy_builder=lambda update_strategy, ctx: {"strategy": update_strategy, "ctx": ctx},
    )


def test_resolve_method_id_accepts_strings_and_enums():
    assert resolve_method_id("cmd") == "cmd"
    assert resolve_method_id(MethodEnum.CMD) == "cmd"


def test_registry_registers_copies_and_reports_unknown_methods():
    registry = SteeringMethodRegistry()
    definition = _definition()
    registry.register(definition)

    assert "unit" in registry
    assert registry.get("unit") is definition
    assert registry.try_get("missing") is None
    assert registry.copy().get("unit") is definition

    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition)

    with pytest.raises(KeyError, match="Registered methods: unit"):
        registry.get("missing")


def test_method_definition_validates_factory_contracts():
    spec = SteeringMethodSpec(
        method_id="bad",
        label="Bad",
        derivation_family=DerivationFamily.CUSTOM,
        runtime_family=RuntimeFamily.CUSTOM,
        intervention_kind=InterventionKind.CUSTOM,
    )

    with pytest.raises(TypeError, match="vector_factory"):
        MethodDefinition(spec=spec, vector_factory=None, strategy_builder=lambda *_: None)


def test_default_registry_exposes_expected_methods_and_layer_policies():
    registry = build_default_method_registry()
    specs = registry.method_specs()

    expected = {
        "cmd",
        "cpca",
        "mbs_cmd",
        "angular",
        "cold_kernel",
        "cold_steer",
    }
    assert set(specs) == expected
    assert specs["cold_kernel"].training_requires_grad is True
    assert specs["cmd"].sources == ("https://aclanthology.org/2024.acl-long.828/",)
    assert specs["cpca"].sources == ("https://arxiv.org/abs/2310.01405",)
    assert specs["mbs_cmd"].sources == specs["cmd"].sources
    assert specs["angular"].sources == ("https://arxiv.org/abs/2510.26243",)
    assert specs["cold_kernel"].sources == ("https://arxiv.org/abs/2603.06495",)
    assert specs["cold_steer"].sources == specs["cold_kernel"].sources
    assert identity_layer_policy([2, 1, 2], {}) == [1, 2]
