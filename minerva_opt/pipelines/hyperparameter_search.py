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

    Supports any Ray Tune search algorithm via the ``search_alg`` parameter in
    ``_search``.  When no algorithm is provided, Ray's default random/grid
    search is used.  To use Bayesian optimisation, pass a ``HyperOptSearch``
    instance.

    Attributes
    ----------
    model : type
        Lightning model class to instantiate for each trial.
    search_space : dict
        Ray Tune search-space definition passed as ``param_space``.
    """

    def __init__(
        self,
        model: type,
        search_space: Dict[str, Any],
        log_dir: Optional[PathLike] = None,
        save_run_status: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        model : type
            Lightning ``LightningModule`` subclass.  Each trial instantiates it
            with ``model(**config)`` where ``config`` is sampled from
            ``search_space``.
        search_space : dict
            Mapping of hyperparameter names to Ray Tune search distributions
            (e.g. ``tune.loguniform``, ``tune.choice``).
        log_dir : path-like, optional
            Directory for Ray storage and run artefacts.  Passed to the base
            ``Pipeline``.
        save_run_status : bool, optional
            Whether to persist pipeline run status.  Defaults to ``True``.
        seed : int, optional
            Global random seed forwarded to the base ``Pipeline``.
        """
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
        """Run hyperparameter search with Ray Tune.

        Constructs a ``TorchTrainer``-backed ``tune.Tuner`` and launches
        ``num_samples`` trials sampled from ``self.search_space``.  If
        ``restore_path`` points to an existing experiment, the search resumes
        from that checkpoint instead of starting fresh.

        Parameters
        ----------
        data : lightning.pytorch.LightningDataModule
            Data module passed to each trial.  Deepcopied per trial unless
            ``data_factory`` is given.
        ckpt_path : path-like or None
            Optional Lightning checkpoint path forwarded to ``trainer.fit``.
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
            List of Lightning callbacks.  Defaults to
            ``[TrainerReportOnIntervalCallback(checkpoint_interval)]``.
        plugins : list or None, optional
            List of Lightning plugins.  Defaults to
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
            Ray Train run configuration.  Defaults to a config that stores
            artefacts under ``self.log_dir``.
        tuner_metric : str, optional
            Metric name used by the scheduler and for selecting the best
            trial.  Defaults to ``"val_loss"``.
        tuner_mode : str, optional
            ``"min"`` or ``"max"`` — optimisation direction for
            ``tuner_metric``.  Defaults to ``"min"``.
        num_samples : int, optional
            Number of hyperparameter configurations to evaluate.
            Defaults to ``10``.
        scheduler : TrialScheduler or None, optional
            Ray Tune trial scheduler.  Defaults to ``ASHAScheduler``.
        search_alg : Searcher or None, optional
            Ray Tune search algorithm.  ``None`` uses random/grid search.
            Any provided searcher is automatically wrapped in a
            ``ConcurrencyLimiter``.
        max_concurrent : int, optional
            Maximum number of trials that may run simultaneously when a
            ``search_alg`` is used.  Defaults to ``4``.
        max_epochs : int, optional
            Maximum training epochs per trial.  Defaults to ``100``.
        checkpoint_interval : int, optional
            Epoch interval for saving checkpoints inside each trial.
            Defaults to ``1``.
        data_factory : callable or None, optional
            Zero-argument factory that returns a fresh ``LightningDataModule``
            for each trial.  Takes precedence over deepcopying ``data``.
        resources_per_worker : dict or None, optional
            Resource dict passed to ``ScalingConfig`` when auto-creating a
            GPU config.  Defaults to ``{"GPU": 1}``.
        num_checkpoints_to_keep : int, optional
            Maximum number of checkpoints retained by Ray.  Defaults to ``1``.
        restore_path : path-like or None, optional
            Path to an existing Ray experiment to resume.  When provided,
            unfinished trials are resumed and errored ones are skipped.

        Returns
        -------
        ray.tune.result_grid.ResultGrid
            Full grid of trial results.  Also stored as
            ``self._last_results``.
        """

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
        """Evaluate a model on the test set.

        When ``ckpt_path`` is given the model is loaded directly from that
        file.  Otherwise the best checkpoint from the most recent search is
        used, which requires ``_search`` to have been called first.

        Parameters
        ----------
        data : lightning.pytorch.LightningDataModule
            Data module that provides the test dataloader.
        ckpt_path : path-like or None, optional
            Explicit Lightning checkpoint to load.  When ``None``, the best
            checkpoint from ``self._last_results`` is used.
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
            If ``ckpt_path`` is ``None`` and no previous search results are
            available.
        """
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
        """Dispatch to ``_search`` or ``_test`` based on ``task``.

        This is the implementation hook called by the base ``Pipeline.run``
        method.  Extra keyword arguments are forwarded verbatim to the
        selected method.

        Parameters
        ----------
        data : lightning.pytorch.LightningDataModule
            Data module for training or testing.
        task : {"search", "test"} or None, optional
            Which sub-task to execute.  ``None`` defaults to ``"search"``.
        ckpt_path : path-like or None, optional
            Checkpoint path forwarded to ``_search`` or ``_test``.
        data_factory : callable or None, optional
            Factory forwarded to ``_search``.
        **kwargs
            Additional keyword arguments forwarded to ``_search`` or
            ``_test``.

        Returns
        -------
        Any
            ``ResultGrid`` when ``task="search"``; test-result list when
            ``task="test"``.

        Raises
        ------
        ValueError
            If ``task`` is not ``None``, ``"search"``, or ``"test"``.
        """
        if task == "search" or task is None:
            return self._search(data, ckpt_path, data_factory=data_factory, **kwargs)
        elif task == "test":
            return self._test(data, ckpt_path, **kwargs)
        else:
            raise ValueError(f"Unknown task {task!r}. Expected 'search' or 'test'.")


def main():
    """Entry point for the CLI interface of ``RayHyperParameterSearch``.

    Parses command-line arguments via ``jsonargparse`` and runs the pipeline.
    """
    from jsonargparse import CLI

    logger.info("Hyper Searching")
    CLI(RayHyperParameterSearch, as_positional=False)


if __name__ == "__main__":
    main()
