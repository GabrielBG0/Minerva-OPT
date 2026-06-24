Tutorial
========

This tutorial walks you through ``minerva-opt`` from a minimal working example
to advanced configuration.  It covers both :class:`RayHyperParameterSearch`
(hyperparameter sweeps) and :class:`AblationStudyPipeline` (ablation studies).


1. Prerequisites & installation
--------------------------------

Requirements
~~~~~~~~~~~~~

- Python 3.10+
- A GPU is strongly recommended for real experiments (CPU fallback works for
  development).

Install
~~~~~~~

.. code-block:: bash

   pip install minerva-opt

For Bayesian optimisation via HyperOpt:

.. code-block:: bash

   pip install "minerva-opt[hyperopt]"

What gets installed
~~~~~~~~~~~~~~~~~~~~

- ``minerva`` — base ``Pipeline`` interface, logging, and reproducibility
- ``ray[tune]`` — distributed trial execution and search orchestration
- ``lightning`` (via minerva) — ``LightningModule`` / ``LightningDataModule``


2. How it works
---------------

When you call ``pipeline.run()``, the pipeline builds a Ray Tune
``TorchTrainer``-backed ``Tuner`` and launches trials::

   pipeline.run(data=..., num_samples=20, max_epochs=50)
          │
          ▼
     Ray Tune Tuner
          │
          ├── Trial 1: sample config → instantiate model → Lightning Trainer → report
          ├── Trial 2: sample config → instantiate model → Lightning Trainer → report
          ├── ...
          └── Trial N: (some pruned early by ASHA scheduler)
          │
          ▼
     ResultGrid  ←  stored on pipeline._last_results

Key design decisions:

- Your model **class** is passed to the pipeline, not an instance.  Each trial
  calls ``YourModel(**sampled_config)`` in an isolated Ray worker.
- Training runs inside Ray workers using ``RayDDPStrategy`` and
  ``RayLightningEnvironment``, making it distributed-ready out of the box.
- The ASHA scheduler prunes poorly-performing trials early so compute is
  concentrated on promising configs.


3. Setting up your model
-------------------------

Your model must be a ``LightningModule`` whose ``__init__`` signature accepts
**only keyword arguments that match your search-space keys**.  The pipeline
calls ``YourModel(**sampled_config)`` for every trial.

.. code-block:: python

   import lightning.pytorch as L
   import torch
   import torch.nn as nn


   class MyModel(L.LightningModule):
       def __init__(self, lr: float = 1e-3, hidden_size: int = 128, dropout: float = 0.2):
           super().__init__()
           self.save_hyperparameters()   # required for load_from_checkpoint
           self.net = nn.Sequential(
               nn.Linear(28 * 28, hidden_size),
               nn.ReLU(),
               nn.Dropout(dropout),
               nn.Linear(hidden_size, 10),
           )
           self.loss_fn = nn.CrossEntropyLoss()

       def forward(self, x):
           return self.net(x.view(x.size(0), -1))

       def training_step(self, batch, batch_idx):
           x, y = batch
           loss = self.loss_fn(self(x), y)
           self.log("train_loss", loss)
           return loss

       def validation_step(self, batch, batch_idx):
           x, y = batch
           loss = self.loss_fn(self(x), y)
           self.log("val_loss", loss, prog_bar=True)

       def test_step(self, batch, batch_idx):
           x, y = batch
           loss = self.loss_fn(self(x), y)
           self.log("test_loss", loss)

       def configure_optimizers(self):
           return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

Two requirements:

1. Call ``self.save_hyperparameters()`` in ``__init__`` so that
   ``load_from_checkpoint`` can reconstruct the model.
2. Log the target metric (e.g. ``"val_loss"``) with ``self.log()`` inside
   ``validation_step``.  The name must match ``tuner_metric``.


4. Setting up your data
-----------------------

Your data must be a ``LightningDataModule``.  The same instance is deepcopied
per trial by default (or use a :ref:`data factory <data-factory>`).

.. code-block:: python

   import lightning.pytorch as L
   from torch.utils.data import DataLoader, random_split
   from torchvision.datasets import MNIST
   from torchvision import transforms


   class MNISTDataModule(L.LightningDataModule):
       def __init__(self, root: str = "data/", batch_size: int = 32):
           super().__init__()
           self.root = root
           self.batch_size = batch_size

       def setup(self, stage=None):
           full = MNIST(self.root, train=True, transform=transforms.ToTensor(), download=True)
           self.train_ds, self.val_ds = random_split(full, [55000, 5000])
           self.test_ds = MNIST(self.root, train=False, transform=transforms.ToTensor())

       def train_dataloader(self):
           return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4)

       def val_dataloader(self):
           return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

       def test_dataloader(self):
           return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)

.. note::
   ``batch_size`` is kept out of the search space here.  If you want to tune
   it, use the :ref:`data factory <data-factory>` pattern so each trial gets a
   fresh data module configured with the sampled value.


5. Defining a search space
---------------------------

The search space is a ``dict`` whose keys match your model's ``__init__``
parameter names and whose values are Ray Tune distributions.

.. code-block:: python

   from ray import tune

   search_space = {
       # Continuous distributions
       "lr":           tune.loguniform(1e-4, 1e-1),
       "weight_decay": tune.loguniform(1e-6, 1e-2),
       "dropout":      tune.uniform(0.0, 0.5),

       # Discrete choices
       "hidden_size":  tune.choice([64, 128, 256, 512]),
       "num_layers":   tune.randint(1, 5),

       # Grid search (exhaustive, fixed set)
       "activation":   tune.grid_search(["relu", "gelu", "tanh"]),
   }

Common distributions:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Distribution
     - Use case
     - Example
   * - ``tune.loguniform(a, b)``
     - Learning rates, weight decay
     - ``tune.loguniform(1e-5, 1e-2)``
   * - ``tune.uniform(a, b)``
     - Dropout, momentum
     - ``tune.uniform(0.0, 0.5)``
   * - ``tune.choice([...])``
     - Architecture options
     - ``tune.choice([64, 128, 256])``
   * - ``tune.randint(a, b)``
     - Layer counts
     - ``tune.randint(1, 6)``
   * - ``tune.grid_search([...])``
     - Fixed set, try all
     - ``tune.grid_search(["adam", "sgd"])``

.. tip::
   Combine ``tune.grid_search`` with ``num_samples > 1`` to repeat the grid
   multiple times with different random seeds — useful for measuring variance
   across the fixed options.


6. Running a search
--------------------

Basic search
~~~~~~~~~~~~~

.. code-block:: python

   from minerva_opt import RayHyperParameterSearch

   pipeline = RayHyperParameterSearch(
       model=MyModel,
       search_space=search_space,
       log_dir="runs/my_experiment",
       seed=42,
   )

   results = pipeline.run(
       data=MNISTDataModule(root="data/"),
       num_samples=20,
       max_epochs=30,
       tuner_metric="val_loss",
       tuner_mode="min",
   )

Ray Tune artefacts (trial logs, checkpoints) are saved under ``log_dir``.
The ASHA scheduler terminates trials that fall behind the best performers, so
most trials run for fewer than ``max_epochs`` epochs.

Key ``run()`` parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 20 52

   * - Parameter
     - Default
     - Description
   * - ``data``
     - —
     - ``LightningDataModule`` instance
   * - ``task``
     - ``"search"``
     - ``"search"`` or ``"test"``
   * - ``ckpt_path``
     - ``None``
     - Warm-start all trials from this checkpoint
   * - ``data_factory``
     - ``None``
     - Callable returning a fresh data module per trial
   * - ``num_samples``
     - ``10``
     - Total number of trials to run
   * - ``max_epochs``
     - ``100``
     - Maximum epochs per trial
   * - ``tuner_metric``
     - ``"val_loss"``
     - Metric to optimise (must be logged with ``self.log``)
   * - ``tuner_mode``
     - ``"min"``
     - ``"min"`` or ``"max"``
   * - ``search_alg``
     - ``None``
     - Search algorithm; ``None`` = random
   * - ``max_concurrent``
     - ``4``
     - Max parallel trials (when using ``search_alg``)
   * - ``scheduler``
     - ASHA
     - Override the pruning scheduler
   * - ``scaling_config``
     - Auto-detected
     - Ray ``ScalingConfig`` (see :ref:`hardware`)
   * - ``resources_per_worker``
     - ``{"GPU": 1}``
     - Resources per trial worker
   * - ``run_config``
     - —
     - Ray ``RunConfig``; defaults to saving under ``log_dir``
   * - ``num_checkpoints_to_keep``
     - ``1``
     - How many checkpoints to retain per trial
   * - ``checkpoint_interval``
     - ``1``
     - Save a checkpoint every N epochs
   * - ``debug_mode``
     - ``False``
     - Disables checkpointing for fast iteration
   * - ``restore_path``
     - ``None``
     - Path to a previous experiment to resume


7. Analysing search results
----------------------------

``pipeline.run()`` returns a ``ResultGrid``.  It is also stored at
``pipeline._last_results``.

.. code-block:: python

   results = pipeline.run(data=data_module, num_samples=20, max_epochs=30)

   # Best trial
   best = results.get_best_result(metric="val_loss", mode="min")
   print("Best config:", best.config["train_loop_config"])
   print("Best val_loss:", best.metrics["val_loss"])

   # Iterate over all trials
   for result in results:
       if result.error:
           print(f"Trial {result.trial_id} failed:", result.error)
           continue
       cfg = result.config["train_loop_config"]
       print(f"  lr={cfg['lr']:.5f}  val_loss={result.metrics['val_loss']:.4f}")

   # DataFrame view
   df = results.get_dataframe()
   print(df[["train_loop_config/lr", "val_loss"]].sort_values("val_loss"))

Loading the best checkpoint:

.. code-block:: python

   import os

   best = results.get_best_result()
   with best.checkpoint.as_directory() as ckpt_dir:
       ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
       model = MyModel.load_from_checkpoint(ckpt_path)


8. Evaluating the best model
-----------------------------

After a search, call ``pipeline.run(task="test")`` to evaluate the best
checkpoint on your test set.

.. code-block:: python

   # Option A: evaluate immediately after search
   results = pipeline.run(data=data_module, num_samples=20, max_epochs=30)
   test_metrics = pipeline.run(data=data_module, task="test")
   # → [{"test_loss": 0.043, ...}]

   # Option B: evaluate from an explicit checkpoint path
   test_metrics = pipeline.run(
       data=data_module,
       task="test",
       ckpt_path="runs/my_experiment/TorchTrainer_xxx/checkpoint.ckpt",
   )

   # Option C: customise the test-time trainer
   test_metrics = pipeline.run(
       data=data_module,
       task="test",
       accelerator="gpu",
       devices=1,
       callbacks=[MyLoggingCallback()],
   )


9. Search algorithms
---------------------

Random search (default)
~~~~~~~~~~~~~~~~~~~~~~~~

The default when ``search_alg=None``.  Each trial samples independently from
the search space distributions.  Efficient and easy to parallelise.

.. code-block:: python

   results = pipeline.run(data=data_module, num_samples=50)

Bayesian optimisation with HyperOpt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian search builds a probabilistic model of the objective and picks
configs likely to improve on the current best.  Use ``max_concurrent`` to
control the parallelism.

.. code-block:: bash

   pip install "minerva-opt[hyperopt]"

.. code-block:: python

   from ray.tune.search.hyperopt import HyperOptSearch

   results = pipeline.run(
       data=data_module,
       search_alg=HyperOptSearch(metric="val_loss", mode="min"),
       max_concurrent=4,
       num_samples=30,
       max_epochs=50,
       tuner_metric="val_loss",
       tuner_mode="min",
   )

When to use Bayesian vs random:

- **Random**: large search spaces, highly parallelisable, good for initial
  exploration.
- **Bayesian**: smaller search spaces (< 10 dimensions), limited compute
  budget, want to exploit structure.

Grid search
~~~~~~~~~~~

Run every combination in the search space exactly once.

.. code-block:: python

   from ray import tune

   search_space = {
       "lr":          tune.grid_search([1e-4, 1e-3, 1e-2]),
       "hidden_size": tune.grid_search([64, 128, 256]),
   }
   # 3 × 3 = 9 trials
   pipeline = RayHyperParameterSearch(model=MyModel, search_space=search_space)
   results = pipeline.run(data=data_module, num_samples=1)

Custom schedulers
~~~~~~~~~~~~~~~~~~

The default ASHA scheduler stops underperforming trials early.  Override it
with any Ray Tune ``TrialScheduler``:

.. code-block:: python

   from ray.tune.schedulers import PopulationBasedTraining
   from ray import tune

   pbt = PopulationBasedTraining(
       time_attr="training_iteration",
       metric="val_loss",
       mode="min",
       perturbation_interval=5,
       hyperparam_mutations={"lr": tune.loguniform(1e-4, 1e-1)},
   )

   results = pipeline.run(data=data_module, scheduler=pbt, num_samples=8)


.. _hardware:

10. Hardware configuration
---------------------------

Auto-detection (default)
~~~~~~~~~~~~~~~~~~~~~~~~~

By default the pipeline detects GPU availability:

- **GPU found**: ``ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1})``
- **No GPU**: falls back to CPU and emits a ``UserWarning``

Single GPU per trial
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ray.train import ScalingConfig

   results = pipeline.run(
       data=data_module,
       scaling_config=ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}),
   )

Fractional GPU (share one GPU across multiple trials)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = pipeline.run(
       data=data_module,
       resources_per_worker={"GPU": 0.5},   # 2 trials share 1 GPU
       num_samples=10,
       max_concurrent=2,
   )

CPU-only (development / CI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ray.train import ScalingConfig

   results = pipeline.run(
       data=data_module,
       scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
       num_samples=3,
       max_epochs=2,
       debug_mode=True,
   )

Multi-worker DDP per trial
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each trial runs with 2 workers (DDP across 2 GPUs):

.. code-block:: python

   from ray.train import ScalingConfig

   results = pipeline.run(
       data=data_module,
       scaling_config=ScalingConfig(
           num_workers=2,
           use_gpu=True,
           resources_per_worker={"GPU": 1},
       ),
   )


.. _checkpointing-strategy:

11. Checkpointing strategy
---------------------------

Two callbacks control how checkpoints are saved during a trial.  The defaults
work well for most cases.

``TrainerReportOnIntervalCallback`` (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Saves a checkpoint every ``checkpoint_interval`` epochs and reports metrics to
Ray every epoch.

.. code-block:: python

   # Default: checkpoint every epoch
   results = pipeline.run(data=data_module, checkpoint_interval=1)

   # Save checkpoints every 5 epochs (reduces disk I/O for long runs)
   results = pipeline.run(data=data_module, checkpoint_interval=5)

``TrainerReportKeepOnlyLastCallback``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only keeps the most recent checkpoint, overwriting it each epoch.  Use when
disk space is tight and you only care about the final state.

.. code-block:: python

   from minerva_opt.callbacks.ray_callbacks import TrainerReportKeepOnlyLastCallback

   results = pipeline.run(
       data=data_module,
       callbacks=[TrainerReportKeepOnlyLastCallback()],
   )

Controlling how many checkpoints are retained per trial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = pipeline.run(
       data=data_module,
       num_checkpoints_to_keep=3,   # keep the top-3 checkpoints per trial
   )

Debug mode
~~~~~~~~~~~

Disables checkpointing entirely.  Useful for quickly verifying your model and
data work before committing to a full search.

.. code-block:: python

   results = pipeline.run(
       data=data_module,
       num_samples=3,
       max_epochs=2,
       debug_mode=True,
   )


12. Resuming an interrupted search
------------------------------------

If a search is interrupted (machine shutdown, OOM, etc.) Ray saves enough
state to resume.

.. code-block:: python

   # Start a long search
   results = pipeline.run(
       data=data_module,
       num_samples=50,
       max_epochs=100,
   )

If interrupted, find the experiment directory under ``log_dir`` and pass it
to ``restore_path``:

.. code-block:: python

   results = pipeline.run(
       data=data_module,
       restore_path="runs/long_search/TorchTrainer_2024-01-15_10-30-00",
   )

This resumes unfinished trials and skips completed ones.  Errored trials are
not retried by default.


.. _data-factory:

13. Using a data factory
-------------------------

By default each trial receives a ``deepcopy`` of the data module.  This can
fail if the data module holds file handles, database connections, or other
non-copyable state.  The ``data_factory`` parameter solves this:

.. code-block:: python

   # Instead of:
   results = pipeline.run(data=MNISTDataModule("data/"))

   # Use a factory (called fresh for each trial):
   results = pipeline.run(
       data=MNISTDataModule("data/"),            # still needed for task="test"
       data_factory=lambda: MNISTDataModule("data/"),
   )

.. note::
   ``data_factory`` is only called during the search (``task="search"``).
   When running ``task="test"``, the ``data`` argument is used directly.


14. Ablation studies
---------------------

:class:`AblationStudyPipeline` measures the contribution of individual model
components by training a full-model *baseline* alongside a set of named
*ablation conditions* — each one being the baseline with one or more components
removed or altered.  Every condition is trained over multiple independent seeds
so you get statistically robust estimates rather than single-run noise.

How conditions are built
~~~~~~~~~~~~~~~~~~~~~~~~~

You supply a ``baseline_config`` (the complete hyperparameter dict for your
full model) and an ``ablations`` dict that maps condition names to *override
dicts*.  The pipeline merges them automatically::

   baseline_config = {"lr": 1e-3, "dropout": 0.2, "use_attention": True, "hidden_size": 128}
   ablations       = {"no_attention": {"use_attention": False}}

   # Internally the pipeline builds:
   conditions = {
       "baseline":     {"lr": 1e-3, "dropout": 0.2, "use_attention": True,  "hidden_size": 128},
       "no_attention": {"lr": 1e-3, "dropout": 0.2, "use_attention": False, "hidden_size": 128},
   }

The key ``"baseline"`` is reserved — passing it inside ``ablations`` raises a
``ValueError``.

Minimal example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from minerva_opt import AblationStudyPipeline

   pipeline = AblationStudyPipeline(
       model=MyModel,
       baseline_config={
           "lr": 1e-3,
           "dropout": 0.2,
           "use_attention": True,
           "hidden_size": 128,
       },
       ablations={
           "no_attention": {"use_attention": False},
           "high_dropout":  {"dropout": 0.5},
           "small_model":   {"hidden_size": 64},
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

This schedules **4 conditions × 5 seeds = 20 trials**.  Unlike the
hyperparameter search pipeline, the ASHA scheduler is configured with
``grace_period=max_epochs`` so **no trial is pruned early** — every condition
always trains to completion.

Key ``run()`` parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 20 52

   * - Parameter
     - Default
     - Description
   * - ``data``
     - —
     - ``LightningDataModule`` instance
   * - ``task``
     - ``"ablate"``
     - ``"ablate"`` to run the study; ``"test"`` to evaluate a condition
   * - ``num_seeds``
     - ``5``
     - Number of independent seeds per condition
   * - ``max_epochs``
     - ``100``
     - Training epochs per trial (no early stopping)
   * - ``tuner_metric``
     - ``"val_loss"``
     - Metric logged by the model and used to rank seeds
   * - ``tuner_mode``
     - ``"min"``
     - ``"min"`` or ``"max"``
   * - ``ckpt_path``
     - ``None``
     - Optional warm-start checkpoint forwarded to ``trainer.fit``
   * - ``devices``
     - ``"auto"``
     - Forwarded to the Lightning ``Trainer``
   * - ``accelerator``
     - ``"auto"``
     - Forwarded to the Lightning ``Trainer``
   * - ``scaling_config``
     - Auto-detected
     - Ray ``ScalingConfig`` — same GPU auto-detection as the search pipeline
   * - ``run_config``
     - —
     - Ray ``RunConfig``; defaults to saving under ``log_dir``
   * - ``scheduler``
     - ASHA (no pruning)
     - Override the scheduler; default sets ``grace_period=max_epochs``
   * - ``checkpoint_interval``
     - ``1``
     - Save a checkpoint every N epochs within each trial
   * - ``num_checkpoints_to_keep``
     - ``1``
     - Number of checkpoints retained per trial
   * - ``debug_mode``
     - ``False``
     - Disables checkpointing for fast development iteration
   * - ``data_factory``
     - ``None``
     - Callable returning a fresh data module per trial
   * - ``resources_per_worker``
     - ``{"GPU": 1}``
     - GPU resource dict used when auto-creating ``ScalingConfig``

How seeding works
~~~~~~~~~~~~~~~~~~

Each trial's seed is ``(pipeline.seed or 0) + seed_offset`` where
``seed_offset`` runs from ``0`` to ``num_seeds - 1``.  Setting ``seed`` on the
pipeline makes the entire study reproducible:

.. code-block:: python

   # seed=0 → trial seeds 0, 1, 2, 3, 4
   # seed=10 → trial seeds 10, 11, 12, 13, 14
   pipeline = AblationStudyPipeline(..., seed=0)

The seed is applied via ``L.seed_everything(seed, workers=True)`` inside each
trial, covering model weight initialisation, data shuffling, and dropout.

Analysing results — ``AblationResults``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pipeline.run(task="ablate")`` returns an
:class:`~minerva_opt.results.ablation_results.AblationResults` object that
wraps Ray's ``ResultGrid`` with ablation-specific aggregation.

``summary()``
^^^^^^^^^^^^^

Returns a ``pandas.DataFrame`` with one row per condition and columns
``{metric}_mean`` / ``{metric}_std`` for every numeric metric that was logged.
Conditions appear in declaration order (baseline first).  Failed trials are
silently excluded.

.. code-block:: python

   df = results.summary()
   print(df)
   #                val_loss_mean  val_loss_std  epoch_mean  ...
   # baseline                0.21          0.01        30.0
   # no_attention            0.27          0.02        30.0
   # high_dropout            0.23          0.01        30.0
   # small_model             0.25          0.02        30.0

All metrics logged by the model appear as columns, so you can inspect
``train_loss``, ``epoch``, and any custom metrics alongside ``val_loss``.

``delta_from_baseline()``
^^^^^^^^^^^^^^^^^^^^^^^^^

Returns a ``pandas.Series`` of signed improvements relative to the baseline
mean.  **Positive means the condition is better than baseline**
(lower loss for ``mode="min"``, higher score for ``mode="max"``).

.. code-block:: python

   delta = results.delta_from_baseline()
   print(delta)
   # baseline         0.00
   # no_attention    -0.06   ← removing attention hurts: 0.06 worse
   # high_dropout    -0.02   ← higher dropout slightly hurts
   # small_model     -0.04   ← smaller model moderately hurts
   # Name: delta_val_loss_vs_baseline

   # Compare a different metric
   delta_train = results.delta_from_baseline(metric="train_loss")

The formula is:

- ``mode="min"``:  ``baseline_mean − condition_mean``  (positive = condition has lower loss)
- ``mode="max"``:  ``condition_mean − baseline_mean``  (positive = condition has higher score)

``best_checkpoint()``
^^^^^^^^^^^^^^^^^^^^^

Returns the Ray ``Checkpoint`` for the seed of a given condition that achieved
the best ``tuner_metric`` value.  Use this to load a trained model:

.. code-block:: python

   import os

   ckpt = results.best_checkpoint("no_attention")
   with ckpt.as_directory() as ckpt_dir:
       ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
       model = MyModel.load_from_checkpoint(ckpt_path)

``raw``
^^^^^^^

The underlying ``ResultGrid`` is available for direct Ray API access:

.. code-block:: python

   raw = results.raw

   # Iterate over every trial (all conditions × all seeds)
   for result in raw:
       if result.error:
           continue
       cfg  = result.config["train_loop_config"]
       name = cfg["condition_name"]
       seed = cfg["ablation_seed"]
       loss = result.metrics["val_loss"]
       print(f"{name} seed={seed}  val_loss={loss:.4f}")

   # Convert to a flat DataFrame for custom analysis
   df_all = raw.get_dataframe()

Full analysis workflow
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = pipeline.run(data=data_module, num_seeds=5, max_epochs=30)

   # 1. Check that all trials succeeded
   n_errors = sum(1 for r in results.raw if r.error)
   print(f"{n_errors} failed trials")

   # 2. Per-condition summary
   summary = results.summary()
   print(summary[["val_loss_mean", "val_loss_std"]])

   # 3. Rank ablations by impact (most harmful first)
   delta = results.delta_from_baseline()
   print(delta.sort_values())          # most negative = most important component

   # 4. Load the best baseline model for further evaluation
   ckpt = results.best_checkpoint("baseline")
   with ckpt.as_directory() as d:
       model = MyModel.load_from_checkpoint(os.path.join(d, "checkpoint.ckpt"))

Evaluating a condition on the test set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After an ablation run, call ``pipeline.run(task="test")`` to evaluate a
specific condition's best checkpoint on the test split.  Pass ``condition``
to select which condition to test (defaults to ``"baseline"``).

.. code-block:: python

   # Evaluate the baseline
   test_metrics = pipeline.run(data=data_module, task="test")
   # → [{"test_loss": 0.043, ...}]

   # Evaluate a specific ablation condition
   test_metrics = pipeline.run(
       data=data_module,
       task="test",
       condition="no_attention",
   )

   # Evaluate from an explicit checkpoint path (skips best-checkpoint lookup)
   test_metrics = pipeline.run(
       data=data_module,
       task="test",
       ckpt_path="runs/ablation/TorchTrainer_xxx/checkpoint.ckpt",
   )

Using a data factory with ablations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``data_factory`` parameter works the same as in the hyperparameter search
pipeline — it is called once per trial and takes precedence over deepcopying
``data``:

.. code-block:: python

   pipeline.run(
       data=MyDataModule("data/"),
       data_factory=lambda: MyDataModule("data/"),
       num_seeds=5,
       max_epochs=30,
   )

Combining ablations with hardware configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ablation pipeline shares the same GPU auto-detection logic as
:class:`RayHyperParameterSearch`.  All hardware options from
:ref:`section 10 <hardware>` apply directly — pass ``scaling_config``,
``resources_per_worker``, or ``debug_mode`` as keyword arguments to ``run()``.

.. code-block:: python

   from ray.train import ScalingConfig

   results = pipeline.run(
       data=data_module,
       num_seeds=5,
       max_epochs=50,
       scaling_config=ScalingConfig(
           num_workers=1,
           use_gpu=True,
           resources_per_worker={"GPU": 0.5},   # 2 trials share 1 GPU
       ),
   )


15. Minerva integration
------------------------

Both ``RayHyperParameterSearch`` and ``AblationStudyPipeline`` are full Minerva
``Pipeline`` objects, so they inherit Minerva's tracking and reproducibility
features.

``log_dir``
~~~~~~~~~~~

Ray Tune results (trial logs, checkpoints, metrics) and the pipeline status
YAML are both saved under ``log_dir``.

After a run, the directory contains::

   runs/experiment_01/
   ├── run_2024-01-15-10-30-00abc12345.yaml   # pipeline status
   └── TorchTrainer_2024-01-15_10-30-00/     # Ray Tune experiment
       ├── TorchTrainer_<trial_id>/
       │   ├── checkpoint_000001/
       │   │   └── checkpoint.ckpt
       │   └── result.json
       └── experiment_state.json

``seed``
~~~~~~~~

The seed is passed to ``L.seed_everything`` before each run, making random
search sampling and weight initialisation reproducible.

``save_run_status``
~~~~~~~~~~~~~~~~~~~~

With ``save_run_status=True`` (the default), a YAML file is saved containing
system info, installed packages, git commit hash, and run start/end times.

CLI usage
~~~~~~~~~

The ``main()`` entry point exposes each pipeline via a CLI:

.. code-block:: bash

   python -m minerva_opt.pipelines.hyperparameter_search \
       --model MyModel \
       --search_space '{"lr": {"class_path": "ray.tune.loguniform", "init_args": {"lower": 0.0001, "upper": 0.1}}}' \
       --log_dir runs/cli_exp


16. Troubleshooting
--------------------

"No GPU detected" warning on a machine with GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray workers may not see GPUs if CUDA is unavailable in the worker environment.
Check:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())   # must be True in the worker

Pass an explicit ``scaling_config`` to suppress the warning:

.. code-block:: python

   from ray.train import ScalingConfig

   results = pipeline.run(
       data=data_module,
       scaling_config=ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}),
   )

``ValueError: Unknown task 'X'. Expected 'search' or 'test'.``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``task`` parameter only accepts ``"search"`` (or ``None``) and ``"test"``.
Check the spelling.

``RuntimeError: No search results available.``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You called ``pipeline.run(task="test")`` before running a search and didn't
provide ``ckpt_path``.  Either run a search first or pass an explicit
checkpoint:

.. code-block:: python

   pipeline.run(data=data_module, task="test", ckpt_path="path/to/checkpoint.ckpt")

Trial crashes with OOM
~~~~~~~~~~~~~~~~~~~~~~

Reduce the number of concurrent trials or request fewer GPU resources:

.. code-block:: python

   results = pipeline.run(
       data=data_module,
       resources_per_worker={"GPU": 0.5},
       max_concurrent=2,
   )

``deepcopy`` of the data module fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``data_factory`` instead (see :ref:`data-factory`).

Search results not appearing in ``log_dir``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure you passed ``log_dir`` when constructing the pipeline, not when
calling ``run()``.  The ``RunConfig.storage_path`` is set from ``self.log_dir``
at search time.

.. code-block:: python

   # Correct
   pipeline = RayHyperParameterSearch(model=MyModel, search_space=..., log_dir="runs/exp")

   # Wrong: log_dir has no effect here
   pipeline.run(data=data_module, log_dir="runs/exp")

``ValueError: 'baseline' is a reserved condition name``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key ``"baseline"`` is reserved and added automatically by
``AblationStudyPipeline``.  Remove it from the ``ablations`` dict you pass to
the constructor:

.. code-block:: python

   # Wrong
   AblationStudyPipeline(
       ...,
       ablations={"baseline": {...}, "no_attention": {...}},
   )

   # Correct — baseline is defined via baseline_config, not ablations
   AblationStudyPipeline(
       ...,
       baseline_config={"lr": 1e-3, ...},
       ablations={"no_attention": {"use_attention": False}},
   )

``RuntimeError: No ablation results available``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You called ``pipeline.run(task="test")`` on an ``AblationStudyPipeline`` before
running the ablation study, and didn't provide ``ckpt_path``.  Either run the
study first or pass an explicit checkpoint:

.. code-block:: python

   pipeline.run(data=data_module, task="test", ckpt_path="path/to/checkpoint.ckpt")

``KeyError`` in ``delta_from_baseline()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metric name passed to ``delta_from_baseline(metric=...)`` was not logged by
any trial.  Check the column names in ``results.summary()`` — they follow the
``{metric}_mean`` pattern:

.. code-block:: python

   print(results.summary().columns.tolist())
   # ['val_loss_mean', 'val_loss_std', 'train_loss_mean', 'train_loss_std', ...]

   # Use the base metric name (without _mean / _std suffix)
   delta = results.delta_from_baseline(metric="val_loss")   # correct
   delta = results.delta_from_baseline(metric="val_loss_mean")  # KeyError

``summary()`` returns an empty DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All trials failed.  Check ``results.raw`` for errors:

.. code-block:: python

   for result in results.raw:
       if result.error:
           print(result.trial_id, result.error)
