# Minerva-OPT

[![Auto Release](https://github.com/gabrielbg0/Minerva-OPT/actions/workflows/auto-release.yml/badge.svg)](https://github.com/gabrielbg0/Minerva-OPT/actions/workflows/auto-release.yml)

Hyperparameter optimization extensions for [Minerva](https://github.com/discovery-unicamp/Minerva), powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

## Description

Minerva-OPT provides a `RayHyperParameterSearch` pipeline that wraps Ray Tune and PyTorch Lightning to run distributed hyperparameter searches on top of any Minerva-compatible model. It supports random search, grid search, and Bayesian optimization (via HyperOpt), with early stopping through the ASHA scheduler.

### Features

- **Drop-in Minerva pipeline**: inherits from `minerva.pipelines.base.Pipeline`, so it integrates with Minerva's logging, reproducibility, and run-status tracking out of the box.
- **Flexible search algorithms**: use Ray Tune's default random/grid search or pass any `ray.tune.search.Searcher` (e.g. `HyperOptSearch` for Bayesian optimization).
- **ASHA early stopping**: trials are stopped early based on intermediate results; `grace_period` and `max_t` are derived automatically from `max_epochs`.
- **Distributed training**: uses `RayDDPStrategy` and `RayLightningEnvironment` for multi-worker trials.
- **Checkpointing**: configurable number of checkpoints per trial, scored on the target metric.

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

### Random / grid search

```python
from ray import tune
from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

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

### Bayesian optimization with HyperOpt

```python
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "dropout": tune.uniform(0.1, 0.5),
}

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

### Testing the best model after search

```python
# Run search first, then test using the best checkpoint automatically
results = pipeline.run(data=my_data_module, num_samples=20, max_epochs=50)
pipeline.run(data=my_data_module, task="test")

# Or test from an explicit checkpoint path
pipeline.run(data=my_data_module, task="test", ckpt_path="path/to/model.ckpt")
```

### Resuming an interrupted search

```python
# Resume from the Ray experiment directory saved under log_dir
pipeline.run(
    data=my_data_module,
    restore_path="logs/TorchTrainer_2024-01-01_00-00-00",
)
```

### Using a data factory (recommended for long searches)

Passing a factory callable avoids deepcopying the data module for every trial, which is safer for data modules with file handles or non-picklable state:

```python
pipeline.run(
    data=my_data_module,         # still required for _test
    data_factory=lambda: MyDataModule(root="data/"),
)
```

### Key `run()` parameters

| Parameter                | Default                        | Description                                                          |
| ------------------------ | ------------------------------ | -------------------------------------------------------------------- |
| `data`                   | —                              | `LightningDataModule` for training and testing                       |
| `task`                   | `"search"`                     | `"search"` to run the sweep, `"test"` to evaluate best checkpoint    |
| `ckpt_path`              | `None`                         | Warm-start all trials from a checkpoint (search) or eval path (test) |
| `data_factory`           | `None`                         | Callable that returns a fresh `LightningDataModule` per trial        |
| `num_samples`            | `10`                           | Number of trials to run                                              |
| `max_epochs`             | `100`                          | Max epochs per trial                                                 |
| `tuner_metric`           | `"val_loss"`                   | Metric to optimize                                                   |
| `tuner_mode`             | `"min"`                        | `"min"` or `"max"`                                                   |
| `search_alg`             | `None`                         | Any `ray.tune.search.Searcher`; `None` = random search               |
| `max_concurrent`         | `4`                            | Max concurrent trials (when using a `search_alg`)                    |
| `scheduler`              | ASHA                           | Override the trial scheduler                                         |
| `scaling_config`         | Auto-detected (GPU or CPU)     | Override Ray `ScalingConfig`                                         |
| `resources_per_worker`   | `{"GPU": 1}` when GPU detected | Custom resource dict, e.g. `{"GPU": 0.5}` for fractional GPU        |
| `checkpoint_interval`    | `1`                            | Save a checkpoint every N epochs                                     |
| `num_checkpoints_to_keep`| `1`                            | Number of top checkpoints to retain per trial                        |
| `restore_path`           | `None`                         | Path to a Ray experiment dir to resume an interrupted search         |
| `debug_mode`             | `False`                        | Disable checkpointing for fast iteration                             |

> **GPU detection**: when `scaling_config` is not provided, the pipeline auto-detects GPU availability. A `UserWarning` is emitted if falling back to CPU. Pass an explicit `scaling_config` to suppress it.

## Requirements

- `minerva >= 0.3.10b0`
- `ray[tune] >= 2.55`
- `hyperopt >= 0.2.7` *(optional — only needed for `HyperOptSearch`)*

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or bug reports, open an issue on the [GitHub issue tracker](https://github.com/gabrielbg0/Minerva-OPT/issues).
