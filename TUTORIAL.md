# Minerva-OPT Tutorial

This tutorial walks you through using `minerva-opt` to run hyperparameter searches on top of any Minerva-compatible model. It progresses from a minimal working example to advanced configuration.

---

## Table of Contents

1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [How It Works](#2-how-it-works)
3. [Quick Start](#3-quick-start)
4. [Setting Up Your Model](#4-setting-up-your-model)
5. [Setting Up Your Data](#5-setting-up-your-data)
6. [Defining a Search Space](#6-defining-a-search-space)
7. [Running a Search](#7-running-a-search)
8. [Analyzing Results](#8-analyzing-results)
9. [Evaluating the Best Model](#9-evaluating-the-best-model)
10. [Search Algorithms](#10-search-algorithms)
11. [Hardware Configuration](#11-hardware-configuration)
12. [Checkpointing Strategy](#12-checkpointing-strategy)
13. [Resuming an Interrupted Search](#13-resuming-an-interrupted-search)
14. [Using a Data Factory](#14-using-a-data-factory)
15. [Minerva Integration](#15-minerva-integration)
16. [Troubleshooting](#16-troubleshooting)
17. [Ablation Studies](#17-ablation-studies)

---

## 1. Prerequisites & Installation

### Requirements

- Python 3.10+
- A GPU is strongly recommended for real experiments (CPU fallback works for development)

### Install

```bash
pip install minerva-opt
```

For Bayesian optimization via HyperOpt:

```bash
pip install "minerva-opt[hyperopt]"
```

### What gets installed

`minerva-opt` depends on:
- `minerva` — the base library providing the `Pipeline` interface, logging, and reproducibility
- `ray[tune]` — distributed trial execution and search orchestration
- `lightning` (via minerva) — model training via `LightningModule` and `LightningDataModule`

---

## 2. How It Works

`RayHyperParameterSearch` is a Minerva `Pipeline` that wraps [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). Here is what happens when you call `pipeline.run()`:

```
pipeline.run(data=..., num_samples=20, max_epochs=50)
       │
       ▼
  Ray Tune Tuner
       │
       ├── Trial 1: sample config → instantiate model → Lightning Trainer → report metrics
       ├── Trial 2: sample config → instantiate model → Lightning Trainer → report metrics
       ├── ...
       └── Trial N: (some pruned early by ASHA scheduler)
       │
       ▼
  ResultGrid  ←  stored on pipeline._last_results
```

**Key design decisions:**
- Your model class is passed to the pipeline, not an instance. Each trial calls `YourModel(**sampled_config)` in an isolated worker.
- Training runs inside Ray workers using `RayDDPStrategy` and `RayLightningEnvironment`, making it distributed-ready out of the box.
- The ASHA scheduler prunes poorly-performing trials early so compute is concentrated on promising configs.

---

## 3. Quick Start

This is the minimal example. The sections below explain each piece in depth.

```python
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
import lightning.pytorch as L
import torch

from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

# 1. Define your search space
search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "hidden_size": tune.choice([64, 128, 256]),
    "dropout": tune.uniform(0.1, 0.5),
}

# 2. Create the pipeline
pipeline = RayHyperParameterSearch(
    model=MyModel,          # class, not instance
    search_space=search_space,
    log_dir="runs/search",
)

# 3. Run the search
results = pipeline.run(
    data=MyDataModule(root="data/"),
    num_samples=30,
    max_epochs=50,
    tuner_metric="val_loss",
    tuner_mode="min",
)

# 4. Inspect the winner
best = results.get_best_result()
print("Best hyperparameters:", best.config["train_loop_config"])
print("Best val_loss:", best.metrics["val_loss"])

# 5. Evaluate on the test set using the best checkpoint
pipeline.run(data=MyDataModule(root="data/"), task="test")
```

---

## 4. Setting Up Your Model

Your model must be a `LightningModule` whose `__init__` signature accepts **only keyword arguments that match your search space keys**. Minerva-OPT calls `YourModel(**sampled_config)` for every trial.

```python
import lightning.pytorch as L
import torch
import torch.nn as nn


class MyModel(L.LightningModule):
    def __init__(self, lr: float = 1e-3, hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.save_hyperparameters()  # required for load_from_checkpoint to work
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
```

**Two requirements:**
1. Call `self.save_hyperparameters()` in `__init__` — this is what makes `load_from_checkpoint` reconstruct the model correctly when you call `task="test"`.
2. Log the metric you're optimizing (e.g., `"val_loss"`) using `self.log()` during `validation_step`. The name must match `tuner_metric`.

---

## 5. Setting Up Your Data

Your data must be a `LightningDataModule`. The same instance is passed to every trial (via `deepcopy` by default, or via a factory — see [Section 14](#14-using-a-data-factory)).

```python
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
```

> **Note:** `batch_size` is deliberately kept out of the search space here. You can include it if you want, but keep it in the data module for clarity. If you include `batch_size` in the search space, use the `data_factory` pattern (see [Section 14](#14-using-a-data-factory)) so each trial gets a fresh data module configured with the sampled value.

---

## 6. Defining a Search Space

The search space is a dictionary where keys match your model's `__init__` parameter names and values are Ray Tune distributions.

```python
from ray import tune

search_space = {
    # Continuous distributions
    "lr": tune.loguniform(1e-4, 1e-1),       # log-uniform: good for learning rates
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "dropout": tune.uniform(0.0, 0.5),        # uniform: good for ratios

    # Discrete choices
    "hidden_size": tune.choice([64, 128, 256, 512]),
    "num_layers": tune.randint(1, 5),          # integer in [1, 5)

    # Grid search (fixed set, exhaustive)
    "activation": tune.grid_search(["relu", "gelu", "tanh"]),
}
```

**Common distributions:**

| Distribution | Use case | Example |
|---|---|---|
| `tune.loguniform(a, b)` | Learning rates, weight decay | `tune.loguniform(1e-5, 1e-2)` |
| `tune.uniform(a, b)` | Dropout, momentum | `tune.uniform(0.0, 0.5)` |
| `tune.choice([...])` | Architecture options | `tune.choice([64, 128, 256])` |
| `tune.randint(a, b)` | Layer counts | `tune.randint(1, 6)` |
| `tune.grid_search([...])` | Fixed set, try all | `tune.grid_search(["adam", "sgd"])` |

> **Tip:** Combine `tune.grid_search` with `num_samples > 1` to repeat the grid multiple times with random seeds — useful for measuring variance across the fixed options.

---

## 7. Running a Search

### Basic search

```python
pipeline = RayHyperParameterSearch(
    model=MyModel,
    search_space=search_space,
    log_dir="runs/my_experiment",
    seed=42,                   # reproducible trial sampling
)

results = pipeline.run(
    data=MNISTDataModule(root="data/"),
    num_samples=20,            # total number of trials
    max_epochs=30,             # max epochs per trial (ASHA may stop earlier)
    tuner_metric="val_loss",   # metric to minimize/maximize
    tuner_mode="min",
)
```

Ray Tune results (trial logs, checkpoints) are saved under `log_dir`. The ASHA scheduler automatically terminates trials that fall behind the best performers, so most trials run for fewer than `max_epochs` epochs.

### All `run()` parameters

| Parameter | Default | Description |
|---|---|---|
| `data` | — | `LightningDataModule` instance |
| `task` | `"search"` | `"search"` or `"test"` |
| `ckpt_path` | `None` | Warm-start all trials from this checkpoint |
| `data_factory` | `None` | Callable returning a fresh data module per trial (see §14) |
| `num_samples` | `10` | Total number of trials to run |
| `max_epochs` | `100` | Maximum epochs per trial |
| `tuner_metric` | `"val_loss"` | Metric to optimize (must be logged with `self.log`) |
| `tuner_mode` | `"min"` | `"min"` or `"max"` |
| `search_alg` | `None` | Search algorithm; `None` = random |
| `max_concurrent` | `4` | Max parallel trials (when using `search_alg`) |
| `scheduler` | ASHA | Override the pruning scheduler |
| `scaling_config` | Auto-detected | Ray `ScalingConfig` (see §11) |
| `resources_per_worker` | `{"GPU": 1}` | Resources per trial worker (see §11) |
| `run_config` | — | Ray `RunConfig`; defaults to saving in `log_dir` |
| `num_checkpoints_to_keep` | `1` | How many checkpoints to retain per trial |
| `checkpoint_interval` | `1` | Save a checkpoint every N epochs |
| `debug_mode` | `False` | Disables checkpointing for fast iteration |
| `restore_path` | `None` | Path to a previous experiment to resume (see §13) |

---

## 8. Analyzing Results

`pipeline.run()` returns a `ResultGrid`. The pipeline also stores it at `pipeline._last_results` so it's accessible after the run.

```python
results = pipeline.run(data=data_module, num_samples=20, max_epochs=30)

# Best trial
best = results.get_best_result(metric="val_loss", mode="min")
print("Best config:", best.config["train_loop_config"])
print("Best val_loss:", best.metrics["val_loss"])
print("Epochs run:", best.metrics["epoch"])

# Iterate over all trials
for result in results:
    if result.error:
        print(f"Trial {result.trial_id} failed:", result.error)
        continue
    cfg = result.config["train_loop_config"]
    print(f"  lr={cfg['lr']:.5f}  val_loss={result.metrics['val_loss']:.4f}")

# Convert to a DataFrame for easy analysis
df = results.get_dataframe()
print(df[["train_loop_config/lr", "val_loss"]].sort_values("val_loss"))
```

### Getting the best checkpoint path

```python
import os

best = results.get_best_result()
with best.checkpoint.as_directory() as ckpt_dir:
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")
    # Load or inspect the checkpoint here
    model = MyModel.load_from_checkpoint(ckpt_path)
```

---

## 9. Evaluating the Best Model

After a search, call `pipeline.run(task="test")` to evaluate the best checkpoint on your test set. The pipeline automatically uses `_last_results` to find the best checkpoint — no extra configuration needed.

```python
# Option A: evaluate immediately after search (uses best checkpoint from results)
results = pipeline.run(data=data_module, num_samples=20, max_epochs=30)
test_metrics = pipeline.run(data=data_module, task="test")
print(test_metrics)  # [{"test_loss": 0.043, ...}]

# Option B: evaluate from an explicit checkpoint path
test_metrics = pipeline.run(
    data=data_module,
    task="test",
    ckpt_path="runs/my_experiment/TorchTrainer_xxx/checkpoint.ckpt",
)

# Option C: configure the test-time trainer
test_metrics = pipeline.run(
    data=data_module,
    task="test",
    accelerator="gpu",
    devices=1,
    callbacks=[MyLoggingCallback()],
)
```

---

## 10. Search Algorithms

### Random search (default)

The default when `search_alg=None`. Each trial samples independently from the search space distributions. Efficient and easy to parallelize.

```python
results = pipeline.run(data=data_module, num_samples=50)
```

### Bayesian optimization with HyperOpt

Bayesian search builds a probabilistic model of the objective and picks configs likely to improve on the current best. It requires sequential evaluation per concurrent batch, so combine it with `max_concurrent` to control parallelism.

```bash
pip install "minerva-opt[hyperopt]"
```

```python
from ray.tune.search.hyperopt import HyperOptSearch

results = pipeline.run(
    data=data_module,
    search_alg=HyperOptSearch(metric="val_loss", mode="min"),
    max_concurrent=4,   # run 4 trials in parallel, then pick next batch
    num_samples=30,
    max_epochs=50,
    tuner_metric="val_loss",
    tuner_mode="min",
)
```

**When to use Bayesian vs random:**
- **Random**: large search spaces, highly parallelizable, good for initial exploration
- **Bayesian**: smaller search spaces (< 10 dimensions), limited compute budget, want to exploit structure

### Grid search

Run every combination in the search space exactly once. Combine with `num_samples` to repeat the grid.

```python
from ray import tune

search_space = {
    "lr": tune.grid_search([1e-4, 1e-3, 1e-2]),
    "hidden_size": tune.grid_search([64, 128, 256]),
}
# 3 × 3 = 9 trials
pipeline = RayHyperParameterSearch(model=MyModel, search_space=search_space)
results = pipeline.run(data=data_module, num_samples=1)
```

### Custom schedulers

The default ASHA scheduler stops underperforming trials early. You can override it:

```python
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="val_loss",
    mode="min",
    perturbation_interval=5,
    hyperparam_mutations={"lr": tune.loguniform(1e-4, 1e-1)},
)

results = pipeline.run(data=data_module, scheduler=pbt, num_samples=8)
```

---

## 11. Hardware Configuration

### Auto-detection (default)

By default, the pipeline detects whether a GPU is available:
- **GPU found**: uses `ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1})`
- **No GPU**: falls back to CPU and emits a `UserWarning`

```
UserWarning: No GPU detected. Falling back to CPU for ScalingConfig.
Pass an explicit scaling_config to suppress this warning or to configure GPU usage.
```

### Single GPU per trial (explicit)

```python
from ray.train import ScalingConfig

results = pipeline.run(
    data=data_module,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}),
)
```

### Fractional GPU (share one GPU across multiple trials)

Useful when your model is small and you want more parallel trials:

```python
results = pipeline.run(
    data=data_module,
    resources_per_worker={"GPU": 0.5},  # 2 trials share 1 GPU
    num_samples=10,
    max_concurrent=2,
)
```

### CPU-only (development / CI)

```python
from ray.train import ScalingConfig

results = pipeline.run(
    data=data_module,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    num_samples=3,
    max_epochs=2,
    debug_mode=True,   # skip checkpointing to speed things up further
)
```

### Multi-worker distributed training per trial

Each trial runs with 2 workers (DDP across 2 GPUs):

```python
from ray.train import ScalingConfig

results = pipeline.run(
    data=data_module,
    scaling_config=ScalingConfig(
        num_workers=2,
        use_gpu=True,
        resources_per_worker={"GPU": 1},
    ),
)
```

---

## 12. Checkpointing Strategy

Two callbacks control how checkpoints are saved during a trial. You normally don't need to touch these — the defaults work well — but understanding them helps when debugging or customizing.

### `TrainerReportOnIntervalCallback` (default)

Saves a checkpoint every `checkpoint_interval` epochs and reports metrics to Ray every epoch.

```python
# Default: checkpoint every epoch
results = pipeline.run(data=data_module, checkpoint_interval=1)

# Save checkpoints every 5 epochs (reduces disk I/O for long runs)
results = pipeline.run(data=data_module, checkpoint_interval=5)
```

### `TrainerReportKeepOnlyLastCallback`

Only keeps the most recent checkpoint, overwriting it each epoch. Use when disk space is tight and you only care about the final state of each trial.

```python
from minerva_opt.callbacks.ray_callbacks import TrainerReportKeepOnlyLastCallback

results = pipeline.run(
    data=data_module,
    callbacks=[TrainerReportKeepOnlyLastCallback()],
)
```

### Controlling how many checkpoints are retained per trial

By default, only the **best** checkpoint per trial is kept (scored by `tuner_metric`). To keep the top-3:

```python
results = pipeline.run(
    data=data_module,
    num_checkpoints_to_keep=3,
)
```

### Using `debug_mode`

Disables checkpointing entirely. Useful for quickly verifying your model and data work before committing to a full search:

```python
results = pipeline.run(
    data=data_module,
    num_samples=3,
    max_epochs=2,
    debug_mode=True,
)
```

---

## 13. Resuming an Interrupted Search

If a search is interrupted (machine shutdown, OOM, etc.), Ray saves enough state to resume. The results are stored under `log_dir`.

```python
# Start a search
results = pipeline.run(
    data=data_module,
    num_samples=50,
    max_epochs=100,
    log_dir="runs/long_search",   # Ray saves results here
)
```

If interrupted, find the experiment directory (it will look like `runs/long_search/TorchTrainer_YYYY-MM-DD_HH-MM-SS`) and pass it to `restore_path`:

```python
results = pipeline.run(
    data=data_module,
    restore_path="runs/long_search/TorchTrainer_2024-01-15_10-30-00",
)
```

This resumes unfinished trials and skips completed ones. Errored trials are not retried by default (pass `resume_errored=True` via a custom `run_config` to change this).

---

## 14. Using a Data Factory

By default, each trial receives a `deepcopy` of the data module you passed. This works for simple data modules but can fail if your data module holds file handles, database connections, or other non-copyable state.

The `data_factory` parameter solves this: it's a callable that creates a fresh data module for each trial.

```python
# Instead of:
results = pipeline.run(data=MNISTDataModule("data/"))

# Use a factory:
results = pipeline.run(
    data=MNISTDataModule("data/"),   # still needed for task="test"
    data_factory=lambda: MNISTDataModule("data/"),
)
```

**When you must use `data_factory`:**

1. Data module holds file handles or database connections
2. `batch_size` is part of the search space:

```python
search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
}

def make_data(config):
    # config is not passed to the factory directly — batch_size lives in the model
    # For batch_size in data, use a closure or a class
    pass

# For batch_size in the data module, create the data module inside the model
# or use a factory that reads from an environment variable set per trial.
# The cleanest pattern: keep batch_size in the model and pass it to the data module
# via trainer.datamodule, or just keep it out of the search space.
```

> **Note:** `data_factory` is only called during the search (`task="search"`). When you call `task="test"`, the `data` argument is used directly.

---

## 15. Minerva Integration

`RayHyperParameterSearch` is a full Minerva `Pipeline`, so it gets all of Minerva's tracking and reproducibility features for free.

### `log_dir` — where everything is saved

Ray Tune results (trial logs, checkpoints, metrics) are saved under `log_dir`. The pipeline's own status YAML is also saved there.

```python
pipeline = RayHyperParameterSearch(
    model=MyModel,
    search_space=search_space,
    log_dir="runs/experiment_01",   # Ray + Pipeline output both go here
)
```

After a run, you'll find:
```
runs/experiment_01/
├── run_2024-01-15-10-30-00abc12345.yaml   # pipeline status (config, git hash, etc.)
└── TorchTrainer_2024-01-15_10-30-00/     # Ray Tune experiment directory
    ├── TorchTrainer_<trial_id>/           # per-trial logs and checkpoints
    │   ├── checkpoint_000001/
    │   │   └── checkpoint.ckpt
    │   └── result.json
    └── experiment_state.json
```

### `seed` — reproducible trial sampling

```python
pipeline = RayHyperParameterSearch(
    model=MyModel,
    search_space=search_space,
    seed=42,
)
```

The seed is passed to `L.seed_everything` before each run, making random search sampling and weight initialization reproducible.

### `save_run_status` — full experiment provenance

```python
pipeline = RayHyperParameterSearch(
    model=MyModel,
    search_space=search_space,
    log_dir="runs/exp",
    save_run_status=True,    # default
)
```

With `save_run_status=True`, a YAML file is saved containing:
- All pipeline configuration (search space, hyperparameters)
- System info (Python version, installed packages)
- Git commit hash and branch
- Run start/end times and status

This makes experiment provenance traceable without any extra effort.

### CLI usage via jsonargparse

The `main()` entry point exposes the pipeline via a CLI, letting you configure and launch searches from the command line:

```bash
python -m minerva_opt.pipelines.hyperparameter_search \
    --model MyModel \
    --search_space '{"lr": {"class_path": "ray.tune.loguniform", "init_args": {"lower": 0.0001, "upper": 0.1}}}' \
    --log_dir runs/cli_exp
```

---

## 16. Troubleshooting

### "No GPU detected" warning on a machine with GPUs

Ray workers may not see GPUs if CUDA is not available in the worker environment. Check:

```python
import torch
print(torch.cuda.is_available())   # must be True in the worker
```

Pass an explicit `scaling_config` to suppress the warning and force GPU usage:

```python
from ray.train import ScalingConfig
results = pipeline.run(
    data=data_module,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}),
)
```

### `ValueError: Unknown task 'X'. Expected 'search' or 'test'.`

The `task` parameter only accepts `"search"` (or `None`) and `"test"`. Check spelling.

### `RuntimeError: No search results available.`

You called `pipeline.run(task="test")` before running a search, and didn't provide `ckpt_path`. Either run a search first or pass an explicit checkpoint:

```python
pipeline.run(data=data_module, task="test", ckpt_path="path/to/checkpoint.ckpt")
```

### `AttributeError: 'float' object has no attribute 'item'`

This was a known bug — fixed in the current version. If you see it, update `minerva-opt`.

### Trial crashes with OOM

Reduce the number of concurrent trials or request fewer GPU resources per worker:

```python
results = pipeline.run(
    data=data_module,
    resources_per_worker={"GPU": 0.5},
    max_concurrent=2,
)
```

### `deepcopy` of the data module fails

Use `data_factory` instead (see [Section 14](#14-using-a-data-factory)).

### Search results not appearing in `log_dir`

Make sure you passed `log_dir` when constructing the pipeline — not when calling `run()`. The `RunConfig.storage_path` is set from `self.log_dir` at search time.

```python
# Correct
pipeline = RayHyperParameterSearch(model=MyModel, search_space=..., log_dir="runs/exp")

# Wrong: log_dir has no effect here
pipeline.run(data=data_module, log_dir="runs/exp")  # log_dir is not a run() param
```

---

## 17. Ablation Studies

`AblationStudyPipeline` answers a different question than hyperparameter search: *how much does each component of my model contribute?* It trains a full-model **baseline** alongside a set of named **ablation conditions** (each being the baseline with one component removed or altered) over multiple random seeds, then aggregates the results so you can compare them with statistical confidence.

### How it works

```
AblationStudyPipeline.run()
       │
       ▼
  conditions × seeds  (grid — no sampling, no early stopping)
       │
       ├── baseline  × seed 0 → train to completion
       ├── baseline  × seed 1 → train to completion
       ├── ...
       ├── no_attention × seed 0 → train to completion
       ├── no_attention × seed 1 → train to completion
       └── ...
       │
       ▼
  AblationResults  ←  stored on pipeline._last_results
```

Every condition × seed pair always trains for the full `max_epochs` — there is no early stopping.

### Defining the pipeline

```python
from minerva_opt import AblationStudyPipeline

pipeline = AblationStudyPipeline(
    model=MyLightningModel,      # class, not instance
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
    log_dir="logs/ablation",
    seed=0,
)
```

Each entry in `ablations` is merged on top of `baseline_config`. The name `"baseline"` is reserved and added automatically.

### Running the study

```python
# 4 conditions × 5 seeds = 20 trials
results = pipeline.run(
    data=my_data_module,
    num_seeds=5,
    max_epochs=30,
    tuner_metric="val_loss",
    tuner_mode="min",
)
```

### Sweeping multiple values for one parameter

To test several values of a single parameter, pass a **list** as the override value. The pipeline expands it into one condition per value, named `"{key}_{value}"`:

```python
ablations = {
    "dropout": {"dropout": [0.2, 0.5, 0.8]},  # → dropout_0.2, dropout_0.5, dropout_0.8
    "no_attention": {"use_attention": False},
}
# 5 conditions × 5 seeds = 25 trials
```

This is useful for sensitivity analysis — you can see not just *whether* dropout matters, but *how much* each value affects performance.

> **Constraint**: only one key per condition entry may be a list. Mixing two list-valued keys in the same entry raises `ValueError`. For multi-parameter sweeps, use `RayHyperParameterSearch` with `tune.grid_search`.

### Analysing results

`pipeline.run()` returns an `AblationResults` object:

```python
# Mean ± std per condition across all seeds
print(results.summary())
#                val_loss_mean  val_loss_std
# baseline                0.21          0.01
# no_attention            0.27          0.02
# high_dropout            0.23          0.01
# dropout_0.2             0.22          0.01
# dropout_0.5             0.23          0.01
# dropout_0.8             0.26          0.02
# small_model             0.25          0.02

# Signed delta vs baseline (negative = worse than baseline)
print(results.delta_from_baseline())
# baseline         0.00
# no_attention    -0.06   ← removing attention hurts the most
# high_dropout    -0.02
# dropout_0.2     -0.01
# dropout_0.5     -0.02
# dropout_0.8     -0.05
# small_model     -0.04

# Rank by impact
delta = results.delta_from_baseline()
print(delta.sort_values())   # most negative = most important component

# Raw Ray ResultGrid for custom analysis
for r in results.raw:
    cfg = r.config["train_loop_config"]
    print(cfg["condition_name"], cfg["ablation_seed"], r.metrics["val_loss"])
```

### Loading a condition's best checkpoint

```python
import os

ckpt = results.best_checkpoint("no_attention")
with ckpt.as_directory() as ckpt_dir:
    model = MyLightningModel.load_from_checkpoint(
        os.path.join(ckpt_dir, "checkpoint.ckpt")
    )
```

### Testing a condition on the test set

```python
# Evaluate the baseline (default)
pipeline.run(data=my_data_module, task="test")

# Evaluate a specific condition
pipeline.run(data=my_data_module, task="test", condition="no_attention")
```

### Key `run()` parameters

| Parameter                 | Default      | Description                                                |
| ------------------------- | ------------ | ---------------------------------------------------------- |
| `data`                    | —            | `LightningDataModule` for training and testing             |
| `task`                    | `"ablate"`   | `"ablate"` to run the study, `"test"` to evaluate          |
| `num_seeds`               | `5`          | Independent seeds per condition                            |
| `max_epochs`              | `100`        | Training epochs per trial (no early stopping)              |
| `tuner_metric`            | `"val_loss"` | Metric logged by the model and used to rank seeds          |
| `tuner_mode`              | `"min"`      | `"min"` or `"max"`                                         |
| `ckpt_path`               | `None`       | Optional warm-start checkpoint forwarded to `trainer.fit`  |
| `condition`               | `"baseline"` | Which condition to evaluate when `task="test"`             |
| `data_factory`            | `None`       | Callable returning a fresh `LightningDataModule` per trial |
| `scaling_config`          | Auto-detected| Ray `ScalingConfig`                                        |
| `resources_per_worker`    | `{"GPU": 1}` | GPU resource dict used when auto-creating `ScalingConfig`  |
| `checkpoint_interval`     | `1`          | Save a checkpoint every N epochs                           |
| `num_checkpoints_to_keep` | `1`          | Checkpoints retained per trial                             |
| `debug_mode`              | `False`      | Disables checkpointing for fast iteration                  |
