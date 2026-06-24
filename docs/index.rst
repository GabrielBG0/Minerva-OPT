Minerva-OPT
===========

Hyperparameter optimisation extensions for `Minerva
<https://github.com/discovery-unicamp/Minerva>`_, powered by `Ray Tune
<https://docs.ray.io/en/latest/tune/index.html>`_.

**minerva-opt** wraps Ray Tune and PyTorch Lightning into Minerva-compatible
``Pipeline`` objects so you can run distributed hyperparameter searches and
ablation studies on top of any ``LightningModule`` without boilerplate.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   getting_started
   tutorial

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index


Features
--------

- **Drop-in Minerva pipeline** — inherits from ``minerva.pipelines.base.Pipeline``,
  picking up logging, reproducibility, and run-status tracking for free.
- **Flexible search algorithms** — random, grid, or Bayesian optimisation
  (via ``HyperOptSearch``); pass any ``ray.tune.search.Searcher``.
- **ASHA early stopping** — underperforming trials are pruned automatically;
  ``grace_period`` and ``max_t`` are derived from ``max_epochs``.
- **Ablation studies** — ``AblationStudyPipeline`` runs every named condition
  across multiple seeds and aggregates results for per-component analysis.
- **Distributed training** — ``RayDDPStrategy`` + ``RayLightningEnvironment``
  make each trial distributed-ready out of the box.
- **Configurable checkpointing** — interval-based or keep-only-last strategies,
  scored on your target metric.


Quick install
-------------

.. code-block:: bash

   pip install minerva-opt

   # Bayesian optimisation support
   pip install "minerva-opt[hyperopt]"


Links
-----

- `Source code <https://github.com/gabrielbg0/Minerva-OPT>`_
- `Issue tracker <https://github.com/gabrielbg0/Minerva-OPT/issues>`_
- `PyPI <https://pypi.org/project/minerva-opt/>`_
