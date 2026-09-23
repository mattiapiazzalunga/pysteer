pysteer: Python Activation Steering for LLMs
============================================

.. meta::
   :description: pysteer is a Python activation-steering library for LLMs, PyTorch transformer language models, representation engineering, and inference-time model steering.
   :keywords: pysteer, activation steering, Python activation steering, LLM steering, representation engineering, activation engineering, PyTorch transformers, mechanistic interpretability, AI safety, model steering

``pysteer`` is a lightweight Python activation-steering library for LLMs and
PyTorch transformer language models. It trains steering artifacts from
prompt/response/reference rows and applies them at inference time without
modifying model weights.

Use it for activation engineering, representation engineering, mechanistic
interpretability experiments, and model-steering workflows where the model
weights should stay unchanged.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   development
   activation_steering_architecture
   api

Build and Open
--------------

From the repository root:

.. code-block:: powershell

   python -m pip install -r docs/requirements.txt
   docs\make.bat open

On systems with ``make``:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   make -C docs open
