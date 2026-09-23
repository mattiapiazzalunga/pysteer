# Contributing to pysteer

Thanks for taking the time to improve `pysteer`.

## Development Setup

Create a virtual environment and install the project with development extras:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Local Checks

Run these before opening a pull request:

```bash
python -m pytest
python -m ruff check .
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

## Adding Steering Methods

New steering methods should plug into the registry instead of editing
`Executor` directly.

The preferred path is:

1. Implement a vector or artifact deriver.
2. Implement or reuse a runtime strategy.
3. Register a `MethodDefinition` with `SteeringMethodRegistry`.
4. Add focused tests for derivation and runtime behavior.
5. Document the method in `docs/usage.rst` or the architecture guide.

The public extension API is exported from `steering_engine`.

## Pull Requests

Keep pull requests focused. Include tests for behavior changes, update docs
when public APIs change, and avoid committing local IDE files, virtual
environments, generated docs, caches, or build artifacts.
