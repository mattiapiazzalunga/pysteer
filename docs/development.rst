Development
===========

Contributor Setup
-----------------

Install the project with development and documentation extras:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"

Run the standard local checks:

.. code-block:: bash

   python -m pytest
   python -m ruff check .
   python -m sphinx -W --keep-going -b html docs docs/_build/html
   python -m build
   python -m twine check dist/*

Extension Points
----------------

New steering methods should be added through ``steering_engine`` rather than
by editing ``Executor`` directly. The main extension objects are:

- ``SteeringMethodRegistry`` for registering available methods.
- ``MethodDefinition`` for connecting metadata, derivation, and runtime logic.
- ``SteeringMethodSpec`` for declarative method identity and taxonomy.
- ``SteeringArtifactDeriver`` for train-time artifact construction.
- ``RuntimeSteeringStrategy`` for inference-time intervention behavior.

Pull Request Expectations
-------------------------

Pull requests should include focused tests for changed behavior, documentation
updates for public API changes, and no generated artifacts such as caches,
virtual environments, built distributions, or local IDE metadata.
