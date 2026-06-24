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

    Attributes
    ----------
    model : type
        Lightning model class to instantiate for each trial.
    baseline_config : dict
        Full hyperparameter configuration for the baseline (all-on) condition.
    ablations : dict
        Mapping of condition name → override dict applied on top of
        ``baseline_config``.
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
        """
        Parameters
        ----------
        model : type
            Lightning ``LightningModule`` subclass instantiated as
            ``model(**config)`` for each trial.
        baseline_config : dict
            Complete hyperparameter dict for the full (baseline) model.
        ablations : dict
            Mapping of condition name → partial config override.  Each
            entry's values are merged on top of ``baseline_config`` to form
            the condition's full config.  The key ``"baseline"`` is reserved
            and will raise ``ValueError`` if present.
        log_dir : path-like or None, optional
            Directory for Ray storage and run artefacts.  Passed to the base
            ``Pipeline``.
        save_run_status : bool, optional
            Whether to persist pipeline run status.  Defaults to ``True``.
        seed : int, optional
            Base seed added to the per-trial seed offset so results are
            reproducible.

        Raises
        ------
        ValueError
            If ``"baseline"`` appears as a key in ``ablations``.
        """
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
        """Run the ablation study across all conditions and seeds.

        Each condition × seed pair is scheduled as an independent Ray trial
        using ``tune.grid_search``.  The ASHA scheduler is configured with
        ``grace_period=max_epochs`` so no trial is pruned early.

        Parameters
        ----------
        data : lightning.pytorch.LightningDataModule
            Data module passed to each trial.  Deepcopied per trial unless
            ``data_factory`` is given.
        ckpt_path : path-like or None
            Optional Lightning checkpoint path forwarded to ``trainer.fit``.
        num_seeds : int, optional
            Number of random seeds to run per condition.  Seeds are computed
            as ``(self.seed or 0) + seed_offset`` for offsets
            ``0 … num_seeds-1``.  Defaults to ``5``.
        max_epochs : int, optional
            Maximum training epochs per trial.  Defaults to ``100``.
        tuner_metric : str, optional
            Metric name used for selecting the best checkpoint per condition.
            Defaults to ``"val_loss"``.
        tuner_mode : str, optional
            ``"min"`` or ``"max"`` — optimisation direction for
            ``tuner_metric``.  Defaults to ``"min"``.
        devices : str, optional
            Device specification forwarded to the Lightning ``Trainer``.
            Defaults to ``"auto"``.
        accelerator : str, optional
            Accelerator type forwarded to the Lightning ``Trainer``.
            Defaults to ``"auto"``.
        strategy : Strategy or None, optional
            Lightning training strategy.  Defaults to
            ``RayDDPStrategy(find_unused_parameters=True)``.
        callbacks : list or None, optional
            Lightning callbacks for each trial.  Defaults to
            ``[TrainerReportOnIntervalCallback(checkpoint_interval)]``.
        plugins : list or None, optional
            Lightning plugins for each trial.  Defaults to
            ``[RayLightningEnvironment()]``.
        num_nodes : int, optional
            Number of nodes forwarded to the Lightning ``Trainer``.
            Defaults to ``1``.
        debug_mode : bool, optional
            When ``True``, disables Lightning checkpointing.  Defaults to
            ``False``.
        scaling_config : ScalingConfig or None, optional
            Ray Train scaling configuration.  Auto-detected from GPU
            availability when ``None``.
        run_config : RunConfig or None, optional
            Ray Train run configuration.  Defaults to a config storing
            artefacts under ``self.log_dir``.
        scheduler : TrialScheduler or None, optional
            Ray Tune trial scheduler.  Defaults to an ``ASHAScheduler`` whose
            grace period equals ``max_epochs`` (disabling early stopping).
        checkpoint_interval : int, optional
            Epoch interval for saving checkpoints inside each trial.
            Defaults to ``1``.
        num_checkpoints_to_keep : int, optional
            Maximum number of checkpoints retained by Ray per trial.
            Defaults to ``1``.
        data_factory : callable or None, optional
            Zero-argument factory that returns a fresh ``LightningDataModule``
            for each trial.  Takes precedence over deepcopying ``data``.
        resources_per_worker : dict or None, optional
            Resource dict passed to ``ScalingConfig`` when auto-creating a
            GPU config.  Defaults to ``{"GPU": 1}``.

        Returns
        -------
        AblationResults
            Aggregated results wrapper.  Also stored as
            ``self._last_results``.
        """

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
        """Evaluate a model on the test set.

        When ``ckpt_path`` is given the model is loaded directly from that
        file.  Otherwise the best checkpoint for the requested ``condition``
        is retrieved from ``self._last_results``.

        Parameters
        ----------
        data : lightning.pytorch.LightningDataModule
            Data module that provides the test dataloader.
        ckpt_path : path-like or None, optional
            Explicit Lightning checkpoint to load.  When ``None``, the best
            seed for ``condition`` from the last ablation run is used.
        condition : str, optional
            Which ablation condition's checkpoint to use when ``ckpt_path``
            is ``None``.  Defaults to ``"baseline"``.
        accelerator : str, optional
            Accelerator type forwarded to the Lightning ``Trainer``.
            Defaults to ``"auto"``.
        devices : str, optional
            Device specification forwarded to the Lightning ``Trainer``.
            Defaults to ``"auto"``.
        callbacks : list or None, optional
            Additional Lightning callbacks for the test trainer.

        Returns
        -------
        list of dict
            Test results as returned by ``trainer.test``.

        Raises
        ------
        RuntimeError
            If ``ckpt_path`` is ``None`` and no previous ablation results are
            available.
        """
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
        """Dispatch to ``_ablate`` or ``_test`` based on ``task``.

        This is the implementation hook called by the base ``Pipeline.run``
        method.  Extra keyword arguments are forwarded verbatim to the
        selected method.

        Parameters
        ----------
        data : lightning.pytorch.LightningDataModule
            Data module for training or testing.
        task : {"ablate", "test"} or None, optional
            Which sub-task to execute.  ``None`` defaults to ``"ablate"``.
        ckpt_path : path-like or None, optional
            Checkpoint path forwarded to ``_ablate`` or ``_test``.
        data_factory : callable or None, optional
            Factory forwarded to ``_ablate``.
        **kwargs
            Additional keyword arguments forwarded to ``_ablate`` or
            ``_test``.

        Returns
        -------
        Any
            ``AblationResults`` when ``task="ablate"``; test-result list when
            ``task="test"``.

        Raises
        ------
        ValueError
            If ``task`` is not ``None``, ``"ablate"``, or ``"test"``.
        """
        if task == "ablate" or task is None:
            return self._ablate(data, ckpt_path, data_factory=data_factory, **kwargs)
        elif task == "test":
            return self._test(data, ckpt_path, **kwargs)
        else:
            raise ValueError(f"Unknown task {task!r}. Expected 'ablate' or 'test'.")


def main():
    """Entry point for the CLI interface of ``AblationStudyPipeline``.

    Parses command-line arguments via ``jsonargparse`` and runs the pipeline.
    """
    from jsonargparse import CLI

    logger.info("Running ablation study")
    CLI(AblationStudyPipeline, as_positional=False)


if __name__ == "__main__":
    main()
