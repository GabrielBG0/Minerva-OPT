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
from ray.tune.schedulers import ASHAScheduler, TrialScheduler

from minerva.pipelines.base import Pipeline
from minerva.utils.typing import PathLike
from minerva_opt.callbacks.ray_callbacks import TrainerReportOnIntervalCallback
from minerva_opt.results.ablation_results import AblationResults

logger = logging.getLogger(__name__)


class AblationStudyPipeline(Pipeline):
    """Ablation study pipeline using Ray Tune and PyTorch Lightning.

    Runs every named condition (plus the baseline) across multiple seeds so
    each component's contribution can be measured by comparing against the
    full-model baseline.  All conditions are exhaustive — no early stopping
    by default and no sampling; every condition × seed pair always trains to
    ``max_epochs``.
    """

    def __init__(
        self,
        model: type,
        baseline_config: Dict[str, Any],
        ablations: Dict[str, Dict[str, Any]],
        log_dir: Optional[PathLike] = None,
        save_run_status: bool = True,
        seed: Optional[int] = None,
    ):
        if "baseline" in ablations:
            raise ValueError(
                "'baseline' is a reserved condition name. "
                "Remove it from ablations — it is added automatically."
            )
        super().__init__(log_dir=log_dir, save_run_status=save_run_status, seed=seed)
        self.model = model
        self.baseline_config = baseline_config
        self.ablations = ablations

        self._conditions: Dict[str, Dict[str, Any]] = {"baseline": dict(baseline_config)}
        for name, overrides in ablations.items():
            self._conditions[name] = {**baseline_config, **overrides}

        self._last_results: Optional[AblationResults] = None

    def _ablate(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike],
        num_seeds: int = 5,
        max_epochs: int = 100,
        tuner_metric: str = "val_loss",
        tuner_mode: str = "min",
        devices: str = "auto",
        accelerator: str = "auto",
        strategy: Optional[Strategy] = None,
        callbacks: Optional[List[Any]] = None,
        plugins: Optional[List[Any]] = None,
        num_nodes: int = 1,
        debug_mode: bool = False,
        scaling_config: Optional[ScalingConfig] = None,
        run_config: Optional[RunConfig] = None,
        scheduler: Optional[TrialScheduler] = None,
        checkpoint_interval: int = 1,
        num_checkpoints_to_keep: int = 1,
        data_factory: Optional[Callable[[], L.LightningDataModule]] = None,
        resources_per_worker: Optional[Dict[str, float]] = None,
    ) -> AblationResults:

        conditions = self._conditions
        base_seed = self.seed or 0

        def _train_func(config):
            condition_name = config["condition_name"]
            trial_seed = config["ablation_seed"]
            full_config = conditions[condition_name]

            L.seed_everything(base_seed + trial_seed, workers=True)

            dm = data_factory() if data_factory is not None else deepcopy(data)
            model = self.model(**full_config)
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

        # Grace period = max_epochs disables ASHA pruning so every trial runs in full.
        scheduler = scheduler or ASHAScheduler(
            time_attr="training_iteration",
            metric=tuner_metric,
            mode=tuner_mode,
            max_t=max_epochs,
            grace_period=max_epochs,
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

        param_space = {
            "train_loop_config": {
                "condition_name": tune.grid_search(list(self._conditions.keys())),
                "ablation_seed": tune.grid_search(list(range(num_seeds))),
            }
        }

        ray_trainer = TorchTrainer(
            _train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric=tuner_metric,
                mode=tuner_mode,
                num_samples=1,
                scheduler=scheduler,
            ),
        )

        result_grid = tuner.fit()
        ablation_results = AblationResults(
            result_grid,
            list(self._conditions.keys()),
            tuner_metric,
            tuner_mode,
        )
        self._last_results = ablation_results

        logger.info("Ablation complete. Conditions: %s", list(self._conditions.keys()))
        return ablation_results

    def _test(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike] = None,
        condition: str = "baseline",
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
                "No ablation results available. "
                "Run _ablate first or provide an explicit ckpt_path."
            )

        ckpt = self._last_results.best_checkpoint(condition)
        with ckpt.as_directory() as checkpoint_dir:
            best_ckpt = os.path.join(
                checkpoint_dir, TrainerReportOnIntervalCallback.CHECKPOINT_NAME
            )
            model = self.model.load_from_checkpoint(best_ckpt)

        trainer = L.Trainer(accelerator=accelerator, devices=devices, callbacks=callbacks)
        return trainer.test(model, datamodule=data)

    def _run(
        self,
        data: L.LightningDataModule,
        task: Optional[Literal["ablate", "test"]] = None,
        ckpt_path: Optional[PathLike] = None,
        data_factory: Optional[Callable[[], L.LightningDataModule]] = None,
        **kwargs,
    ) -> Any:
        if task == "ablate" or task is None:
            return self._ablate(data, ckpt_path, data_factory=data_factory, **kwargs)
        elif task == "test":
            return self._test(data, ckpt_path, **kwargs)
        else:
            raise ValueError(f"Unknown task {task!r}. Expected 'ablate' or 'test'.")


def main():
    from jsonargparse import CLI

    logger.info("Running ablation study")
    CLI(AblationStudyPipeline, as_positional=False)


if __name__ == "__main__":
    main()
