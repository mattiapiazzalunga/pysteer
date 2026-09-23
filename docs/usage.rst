Usage
=====

Basic Training Flow
-------------------

Training data is a dataframe with prompt, response, and reference columns. The
``reference`` value is interpreted as the positive/negative label for the
response activation span. Built-in unsupervised methods require ``reference``
to contain only ``0`` and ``1`` values, with at least one positive and one
negative row in every contrastive scope. Standard methods validate the full
dataframe; MBS-CMD validates each selected ``mbs_layer``.

.. code-block:: python

   import pandas as pd
   from pysteer import Executor

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
       layers_to_extract=[12, 16, 20],
       alpha=0.5,
   )

   wrapper = executor.representation_extractor()
   with wrapper as steered_model:
       output = steered_model.generate(**inputs, max_new_tokens=64)

Built-in Methods
----------------

The default registry exposes five implementations through six IDs:
``cmd``, ``cpca``, ``mbs_cmd``, ``angular``, ``cold_kernel``, and the exact
``cold_steer`` alias. These are independent adaptations, not bit-for-bit
reproductions of reference repositories.

- ``cmd`` adapts DiffMean/Contrastive Activation Addition and computes the
  desired-minus-undesired mean direction.
- ``cpca`` is pysteer's PCA contrastive-direction variant; it is not the
  standard statistical cPCA algorithm.
- ``mbs_cmd`` is a pysteer-specific layer-routed extension of ``cmd``.
- ``angular`` adapts Angular Steering to transformer-block residual outputs.
- ``cold_kernel`` adapts the unit-kernel COLD-Steer idea. COLD-FD and general
  kernel variants are not implemented.

The sources and exact implementation differences are documented in the
README's ``Supported Steering Methods`` table. They are also exposed through
``build_default_method_registry().method_specs()[method_id].sources``. New
methods can be added with ``SteeringMethodRegistry`` and ``MethodDefinition``
without editing ``Executor``.

Runtime Boundaries
------------------

The model must expose a supported causal-decoder transformer-block layout. Use
the returned wrapper as a context manager. ``generate`` currently requires
``num_beams=1``; beam-mask reordering is not implemented. Built-in methods only
support ``TaskTypeEnum.UNSUPERVISED``.

Documentation Commands
----------------------

Use ``html`` to build documentation and ``open`` to build and launch it in the
default browser.

.. code-block:: powershell

   docs\make.bat html
   docs\make.bat open

.. code-block:: bash

   make -C docs html
   make -C docs open
