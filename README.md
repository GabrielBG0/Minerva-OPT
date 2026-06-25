# Minerva-OPT

[![Auto Release](https://github.com/gabrielbg0/Minerva-OPT/actions/workflows/auto-release.yml/badge.svg)](https://github.com/gabrielbg0/Minerva-OPT/actions/workflows/auto-release.yml)
[![Deploy Docs](https://github.com/gabrielbg0/Minerva-OPT/actions/workflows/docs.yml/badge.svg)](https://gabrielbg0.github.io/Minerva-OPT)

Hyperparameter optimization and ablation study extensions for [Minerva](https://github.com/discovery-unicamp/Minerva), powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

## Description

Minerva-OPT provides two Minerva-compatible pipelines built on top of Ray Tune and PyTorch Lightning:

- **`RayHyperParameterSearch`** — runs distributed hyperparameter sweeps with support for random search, grid search, and Bayesian optimization (via HyperOpt), with early stopping through the ASHA scheduler.
- **`AblationStudyPipeline`** — trains a full-model baseline alongside a set of named ablation conditions over multiple seeds, then aggregates the results so you can measure each component's contribution with statistical confidence.

### Features

- **Drop-in Minerva pipeline**: both pipelines inherit from `minerva.pipelines.base.Pipeline`, integrating with Minerva's logging, reproducibility, and run-status tracking out of the box.
- **Flexible search algorithms**: use Ray Tune's default random/grid search or pass any `ray.tune.search.Searcher` (e.g. `HyperOptSearch` for Bayesian optimization).
- **ASHA early stopping**: trials are stopped early based on intermediate results; `grace_period` and `max_t` are derived automatically from `max_epochs`.
- **Ablation studies**: measure the contribution of individual model components by comparing named conditions against a shared baseline across multiple seeds.
- **Distributed training**: uses `RayDDPStrategy` and `RayLightningEnvironment` for multi-worker trials.
- **Configurable checkpointing**: interval-based or keep-only-last strategies, scored on the target metric.

## Installation

Requires Python 3.10+.

### With uv (recommended)

```bash
uv pip install minerva-opt
```

### With pip

```bash
pip install minerva-opt
```

To use Bayesian optimization via HyperOpt, install the optional extra:

```bash
pip install "minerva-opt[hyperopt]"
```

## Usage

### Hyperparameter search

#### Random / grid search

```python
from ray import tune
from minerva_opt import RayHyperParameterSearch

search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "hidden_size": tune.choice([64, 128, 256]),
}

pipeline = RayHyperParameterSearch(
    model=MyLightningModel,  # class, not instance — instantiated per trial as MyLightningModel(**config)
    search_space=search_space,
    log_dir="logs/",
)

results = pipeline.run(data=my_data_module, num_samples=20, max_epochs=50)
best = results.get_best_result()
print(best.config)
```

#### Bayesian optimization with HyperOpt

```python
from ray.tune.search.hyperopt import HyperOptSearch

pipeline = RayHyperParameterSearch(
    model=MyLightningModel,
    search_space=search_space,
)

results = pipeline.run(
    data=my_data_module,
    search_alg=HyperOptSearch(),
    num_samples=30,
    max_epochs=100,
    tuner_metric="val_loss",
    tuner_mode="min",
)
```

#### Testing the best model after search

```python
# Run search first, then test using the best checkpoint automatically
results = pipeline.run(data=my_data_module, num_samples=20, max_epochs=50)
pipeline.run(data=my_data_module, task="test")

# Or test from an explicit checkpoint path
pipeline.run(data=my_data_module, task="test", ckpt_path="path/to/model.ckpt")
```

#### Resuming an interrupted search

```python
pipeline.run(
    data=my_data_module,
    restore_path="logs/TorchTrainer_2024-01-01_00-00-00",
)
```

#### Key `run()` parameters

| Parameter                 | Default                        | Description                                                           |
| ------------------------- | ------------------------------ | --------------------------------------------------------------------- |
| `data`                    | —                              | `LightningDataModule` for training and testing                        |
| `task`                    | `"search"`                     | `"search"` to run the sweep, `"test"` to evaluate best checkpoint     |
| `ckpt_path`               | `None`                         | Warm-start all trials from a checkpoint (search) or eval path (test)  |
| `data_factory`            | `None`                         | Callable that returns a fresh `LightningDataModule` per trial         |
| `num_samples`             | `10`                           | Number of trials to run                                               |
| `max_epochs`              | `100`                          | Max epochs per trial                                                  |
| `tuner_metric`            | `"val_loss"`                   | Metric to optimize                                                    |
| `tuner_mode`              | `"min"`                        | `"min"` or `"max"`                                                    |
| `search_alg`              | `None`                         | Any `ray.tune.search.Searcher`; `None` = random search                |
| `max_concurrent`          | `4`                            | Max concurrent trials (when using a `search_alg`)                     |
| `scheduler`               | ASHA                           | Override the trial scheduler                                          |
| `scaling_config`          | Auto-detected (GPU or CPU)     | Override Ray `ScalingConfig`                                          |
| `resources_per_worker`    | `{"GPU": 1}` when GPU detected | Custom resource dict, e.g. `{"GPU": 0.5}` for fractional GPU         |
| `checkpoint_interval`     | `1`                            | Save a checkpoint every N epochs                                      |
| `num_checkpoints_to_keep` | `1`                            | Number of top checkpoints to retain per trial                         |
| `restore_path`            | `None`                         | Path to a Ray experiment dir to resume an interrupted search          |
| `debug_mode`              | `False`                        | Disable checkpointing for fast iteration                              |

> **GPU detection**: when `scaling_config` is not provided, the pipeline auto-detects GPU availability. A `UserWarning` is emitted if falling back to CPU. Pass an explicit `scaling_config` to suppress it.

---

### Ablation studies

`AblationStudyPipeline` trains a *baseline* condition and a set of named *ablation* conditions — each being the baseline with one or more components removed or altered — across multiple seeds, then aggregates the results for statistical comparison.

#### Defining conditions

You supply a `baseline_config` (the complete config for your full model) and an `ablations` dict mapping condition names to *override dicts*. The pipeline merges them automatically:

```python
baseline_config = {"lr": 1e-3, "dropout": 0.2, "use_attention": True, "hidden_size": 128}
ablations       = {"no_attention": {"use_attention": False}}

# Internally becomes:
# baseline     → {"lr": 1e-3, "dropout": 0.2, "use_attention": True,  "hidden_size": 128}
# no_attention → {"lr": 1e-3, "dropout": 0.2, "use_attention": False, "hidden_size": 128}
```

The key `"baseline"` is reserved and added automatically — passing it inside `ablations` raises a `ValueError`.

#### Sweeping multiple values for one parameter

To test several values of a single parameter without writing a separate condition for each, pass a **list** as the override value. The pipeline expands it into one named condition per value using the pattern `"{name}_{value}"`:

```python
ablations = {
    "dropout": {"dropout": [0.2, 0.5, 0.8]},  # expands to dropout_0.2, dropout_0.5, dropout_0.8
    "no_attention": {"use_attention": False},
}
# Produces 5 conditions total: baseline, dropout_0.2, dropout_0.5, dropout_0.8, no_attention
```

> **Constraint**: only one key per condition entry may be a list. Passing two list-valued keys in the same entry raises `ValueError` — use separate entries or `RayHyperParameterSearch` for multi-parameter grid search.

#### Running the study

```python
from minerva_opt import AblationStudyPipeline

pipeline = AblationStudyPipeline(
    model=MyLightningModel,
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

# 4 conditions × 5 seeds = 20 trials; no early stopping
results = pipeline.run(
    data=my_data_module,
    num_seeds=5,
    max_epochs=30,
    tuner_metric="val_loss",
    tuner_mode="min",
)
```

#### Analysing results

`pipeline.run()` returns an `AblationResults` object:

```python
# Mean ± std of every logged metric per condition
print(results.summary())
#                val_loss_mean  val_loss_std
# baseline                0.21          0.01
# no_attention            0.27          0.02
# high_dropout            0.23          0.01
# small_model             0.25          0.02

# Signed improvement vs baseline — positive means better than baseline
print(results.delta_from_baseline())
# baseline         0.00
# no_attention    -0.06   ← most important component
# high_dropout    -0.02
# small_model     -0.04

# Rank components by impact
delta = results.delta_from_baseline()
print(delta.sort_values())   # most negative = most important

# Access the underlying Ray ResultGrid for custom analysis
for r in results.raw:
    cfg = r.config["train_loop_config"]
    print(cfg["condition_name"], cfg["ablation_seed"], r.metrics["val_loss"])
```

#### Loading a condition's best checkpoint

```python
import os

ckpt = results.best_checkpoint("no_attention")
with ckpt.as_directory() as ckpt_dir:
    model = MyLightningModel.load_from_checkpoint(
        os.path.join(ckpt_dir, "checkpoint.ckpt")
    )
```

#### Testing a condition on the test set

```python
# Evaluate the baseline (default)
pipeline.run(data=my_data_module, task="test")

# Evaluate a specific ablation condition
pipeline.run(data=my_data_module, task="test", condition="no_attention")
```

#### Key `run()` parameters

| Parameter                 | Default     | Description                                               |
| ------------------------- | ----------- | --------------------------------------------------------- |
| `data`                    | —           | `LightningDataModule` for training and testing            |
| `task`                    | `"ablate"`  | `"ablate"` to run the study, `"test"` to evaluate         |
| `num_seeds`               | `5`         | Independent seeds per condition                           |
| `max_epochs`              | `100`       | Training epochs per trial (no early stopping)             |
| `tuner_metric`            | `"val_loss"`| Metric logged by the model and used to rank seeds         |
| `tuner_mode`              | `"min"`     | `"min"` or `"max"`                                        |
| `ckpt_path`               | `None`      | Optional warm-start checkpoint forwarded to `trainer.fit` |
| `condition`               | `"baseline"`| Which condition to evaluate when `task="test"`            |
| `data_factory`            | `None`      | Callable returning a fresh `LightningDataModule` per trial|
| `scaling_config`          | Auto-detected | Ray `ScalingConfig`                                     |
| `resources_per_worker`    | `{"GPU": 1}`| GPU resource dict used when auto-creating `ScalingConfig` |
| `checkpoint_interval`     | `1`         | Save a checkpoint every N epochs                          |
| `num_checkpoints_to_keep` | `1`         | Checkpoints retained per trial                            |
| `debug_mode`              | `False`     | Disables checkpointing for fast iteration                 |

---

## Requirements

- `minerva >= 0.3.10b0`
- `ray[tune] >= 2.55`
- `hyperopt >= 0.2.7` *(optional — only needed for `HyperOptSearch`)*

## Documentation

Full documentation including a detailed tutorial, API reference, and troubleshooting guide is available at **[gabrielbg0.github.io/Minerva-OPT](https://gabrielbg0.github.io/Minerva-OPT)**.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or bug reports, open an issue on the [GitHub issue tracker](https://github.com/gabrielbg0/Minerva-OPT/issues).
