import logging
import os
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional

import lightning.pytorch as L
import torch
from lightning.pytorch.strategies import Strategy
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer
from ray.train.torch import TorchTrainer
from ray.tune.result_grid import ResultGrid
from ray.tune.schedulers import ASHAScheduler, TrialScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.searcher import Searcher

from minerva.pipelines.base import Pipeline
from minerva.utils.typing import PathLike
from minerva_opt.callbacks.ray_callbacks import TrainerReportOnIntervalCallback

logger = logging.getLogger(__name__)


class RayHyperParameterSearch(Pipeline):
    """Hyperparameter search pipeline using Ray Tune and PyTorch Lightning.

    Supports any Ray Tune search algorithm via the `search_alg` parameter in
    `_search`. When no algorithm is provided, Ray's default random/grid search
    is used. To use Bayesian optimization, pass a `HyperOptSearch` instance.
    """

    def __init__(
        self,
        model: type,
        search_space: Dict[str, Any],
        log_dir: Optional[PathLike] = None,
        save_run_status: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(log_dir=log_dir, save_run_status=save_run_status, seed=seed)
        self.model = model
        self.search_space = search_space
        self._last_results: Optional[ResultGrid] = None

    def _search(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike],
        devices: str = "auto",
        accelerator: str = "auto",
        strategy: Optional[Strategy] = None,
        callbacks: Optional[List[Any]] = None,
        plugins: Optional[List[Any]] = None,
        num_nodes: int = 1,
        debug_mode: bool = False,
        scaling_config: Optional[ScalingConfig] = None,
        run_config: Optional[RunConfig] = None,
        tuner_metric: str = "val_loss",
        tuner_mode: str = "min",
        num_samples: int = 10,
        scheduler: Optional[TrialScheduler] = None,
        search_alg: Optional[Searcher] = None,
        max_concurrent: int = 4,
        max_epochs: int = 100,
        checkpoint_interval: int = 1,
        data_factory: Optional[Callable[[], L.LightningDataModule]] = None,
        resources_per_worker: Optional[Dict[str, float]] = None,
        num_checkpoints_to_keep: int = 1,
        restore_path: Optional[PathLike] = None,
    ) -> ResultGrid:

        def _train_func(config):
            dm = data_factory() if data_factory is not None else deepcopy(data)
            model = self.model(**config)
            trainer = L.Trainer(
                max_epochs=max_epochs,
                devices=devices,
                accelerator=accelerator,
                strategy=strategy or RayDDPStrategy(find_unused_parameters=True),
                callbacks=callbacks or [TrainerReportOnIntervalCallback(checkpoint_interval)],
                plugins=plugins or [RayLightningEnvironment()],
                enable_progress_bar=False,
                num_nodes=num_nodes,
                enable_checkpointing=False if debug_mode else None,
            )
            trainer = prepare_trainer(trainer)
            trainer.fit(model, dm, ckpt_path=ckpt_path)

        scheduler = scheduler or ASHAScheduler(
            time_attr="training_iteration",
            metric=tuner_metric,
            mode=tuner_mode,
            max_t=max_epochs,
            grace_period=max(1, max_epochs // 10),
        )

        if scaling_config is None:
            if torch.cuda.is_available():
                scaling_config = ScalingConfig(
                    num_workers=1,
                    use_gpu=True,
                    resources_per_worker=resources_per_worker or {"GPU": 1},
                )
            else:
                warnings.warn(
                    "No GPU detected. Falling back to CPU for ScalingConfig. "
                    "Pass an explicit scaling_config to suppress this warning "
                    "or to configure GPU usage.",
                    UserWarning,
                    stacklevel=2,
                )
                scaling_config = ScalingConfig(num_workers=1, use_gpu=False)

        run_config = run_config or RunConfig(
            storage_path=str(self.log_dir),
            checkpoint_config=CheckpointConfig(
                num_to_keep=num_checkpoints_to_keep,
                checkpoint_score_attribute=tuner_metric,
                checkpoint_score_order=tuner_mode,
            ),
        )

        if search_alg is not None:
            search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)

        ray_trainer = TorchTrainer(
            _train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        if restore_path is not None:
            tuner = tune.Tuner.restore(
                path=str(restore_path),
                trainable=ray_trainer,
                resume_unfinished=True,
                resume_errored=False,
            )
        else:
            tuner = tune.Tuner(
                ray_trainer,
                param_space={"train_loop_config": self.search_space},
                tune_config=tune.TuneConfig(
                    metric=tuner_metric,
                    mode=tuner_mode,
                    num_samples=num_samples,
                    scheduler=scheduler,
                    search_alg=search_alg,
                ),
            )

        results = tuner.fit()
        self._last_results = results

        best = results.get_best_result()
        logger.info("Best config: %s", best.config)
        logger.info("Best %s: %s", tuner_metric, best.metrics.get(tuner_metric))
        return results

    def _test(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike] = None,
        accelerator: str = "auto",
        devices: str = "auto",
        callbacks: Optional[List[Any]] = None,
    ) -> Any:
        if ckpt_path is not None:
            model = self.model.load_from_checkpoint(ckpt_path)
            trainer = L.Trainer(accelerator=accelerator, devices=devices, callbacks=callbacks)
            return trainer.test(model, datamodule=data)

        if self._last_results is None:
            raise RuntimeError(
                "No search results available. Run _search first or provide an explicit ckpt_path."
            )

        best = self._last_results.get_best_result()
        with best.checkpoint.as_directory() as checkpoint_dir:
            best_ckpt = os.path.join(
                checkpoint_dir, TrainerReportOnIntervalCallback.CHECKPOINT_NAME
            )
            model = self.model.load_from_checkpoint(best_ckpt)

        trainer = L.Trainer(accelerator=accelerator, devices=devices, callbacks=callbacks)
        return trainer.test(model, datamodule=data)

    def _run(
        self,
        data: L.LightningDataModule,
        task: Optional[Literal["search", "test"]] = None,
        ckpt_path: Optional[PathLike] = None,
        data_factory: Optional[Callable[[], L.LightningDataModule]] = None,
        **kwargs,
    ) -> Any:
        if task == "search" or task is None:
            return self._search(data, ckpt_path, data_factory=data_factory, **kwargs)
        elif task == "test":
            return self._test(data, ckpt_path, **kwargs)
        else:
            raise ValueError(f"Unknown task {task!r}. Expected 'search' or 'test'.")


def main():
    from jsonargparse import CLI

    logger.info("Hyper Searching")
    CLI(RayHyperParameterSearch, as_positional=False)


if __name__ == "__main__":
    main()
