from __future__ import annotations

import executor as executor_module
from pysteer import Executor
from pysteer import SteeringMethodRegistry


def test_public_facade_imports_without_constructing_executor():
    assert Executor is executor_module.Executor
    assert SteeringMethodRegistry.__name__ == "SteeringMethodRegistry"


def test_executor_pandas_dependency_error_is_actionable(monkeypatch):
    monkeypatch.setattr(executor_module, "pd", None)

    try:
        executor_module._require_pandas()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError when pandas is unavailable")

    assert "requires pandas" in message
    assert "python -m pip install pysteer-adaptation" in message
