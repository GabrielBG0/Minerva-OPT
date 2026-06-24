Getting Started
===============

Requirements
------------

- Python 3.10 or later
- A GPU is strongly recommended for real experiments; CPU fallback works for
  development and CI.

Runtime dependencies installed automatically:

- ``minerva >= 0.3.10b0`` — base ``Pipeline`` interface, logging,
  reproducibility
- ``ray[tune] >= 2.55`` — distributed trial execution and search orchestration
- ``lightning`` (via minerva) — model training via ``LightningModule`` and
  ``LightningDataModule``


Installation
------------

Using **uv** (recommended):

.. code-block:: bash

   uv pip install minerva-opt

Using **pip**:

.. code-block:: bash

   pip install minerva-opt

For Bayesian optimisation via HyperOpt, install the optional extra:

.. code-block:: bash

   pip install "minerva-opt[hyperopt]"


Quick start — hyperparameter search
------------------------------------

Below is the minimal end-to-end example.  Each step is explained in detail in
the :doc:`tutorial`.

.. code-block:: python

   from ray import tune
   from minerva_opt import RayHyperParameterSearch

   # 1. Define the search space.
   #    Keys must match your LightningModule.__init__ parameter names.
   search_space = {
       "lr":          tune.loguniform(1e-4, 1e-1),
       "hidden_size": tune.choice([64, 128, 256]),
       "dropout":     tune.uniform(0.1, 0.5),
   }

   # 2. Build the pipeline.
   pipeline = RayHyperParameterSearch(
       model=MyModel,            # class, not instance
       search_space=search_space,
       log_dir="runs/search",
       seed=42,
   )

   # 3. Run the sweep.
   results = pipeline.run(
       data=MyDataModule(root="data/"),
       num_samples=30,
       max_epochs=50,
       tuner_metric="val_loss",
       tuner_mode="min",
   )

   # 4. Inspect the winner.
   best = results.get_best_result()
   print("Best config:", best.config["train_loop_config"])
   print("Best val_loss:", best.metrics["val_loss"])

   # 5. Evaluate the best checkpoint on the test set.
   pipeline.run(data=MyDataModule(root="data/"), task="test")


Quick start — ablation study
------------------------------

.. code-block:: python

   from minerva_opt import AblationStudyPipeline

   pipeline = AblationStudyPipeline(
       model=MyModel,
       baseline_config={"lr": 1e-3, "dropout": 0.2, "use_attention": True},
       ablations={
           "no_attention": {"use_attention": False},
           "high_dropout":  {"dropout": 0.5},
       },
       log_dir="runs/ablation",
       seed=0,
   )

   results = pipeline.run(
       data=MyDataModule(root="data/"),
       num_seeds=5,
       max_epochs=30,
       tuner_metric="val_loss",
       tuner_mode="min",
   )

   print(results.summary())
   print(results.delta_from_baseline())


What's next
-----------

- Read the full :doc:`tutorial` for advanced configuration, hardware options,
  checkpointing strategies, and troubleshooting.
- Browse the :doc:`api/index` for the complete API reference.
