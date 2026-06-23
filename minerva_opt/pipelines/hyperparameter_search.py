from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

import lightning.pytorch as L
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
    ) -> ResultGrid:

        def _train_func(config):
            dm = deepcopy(data)
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

        scaling_config = scaling_config or ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1}
        )

        run_config = run_config or RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=tuner_metric,
                checkpoint_score_order=tuner_mode,
            )
        )

        if search_alg is not None:
            search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent)

        ray_trainer = TorchTrainer(
            _train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )

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
        best = results.get_best_result()
        print(f"Best config: {best.config}")
        print(f"Best {tuner_metric}: {best.metrics.get(tuner_metric)}")
        return results

    def _test(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike]) -> Any:
        raise NotImplementedError(
            "Load the best checkpoint from the ResultGrid returned by _search "
            "and call trainer.test() directly."
        )

    def _run(
        self,
        data: L.LightningDataModule,
        task: Optional[Literal["search", "test"]] = None,
        ckpt_path: Optional[PathLike] = None,
        **kwargs,
    ) -> Any:
        if task == "search" or task is None:
            return self._search(data, ckpt_path, **kwargs)
        elif task == "test":
            return self._test(data, ckpt_path)


def main():
    from jsonargparse import CLI

    print("Hyper Searching")
    CLI(RayHyperParameterSearch, as_positional=False)


if __name__ == "__main__":
    main()
