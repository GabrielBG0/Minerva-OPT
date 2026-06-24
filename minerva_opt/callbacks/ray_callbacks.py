import os
import shutil
import tempfile
from pathlib import Path

import lightning.pytorch as L
from ray import train
from ray.train import Checkpoint

try:
    from ray._common.usage.usage_lib import TagKey, record_extra_usage_tag

    def _record_usage():
        """Record Ray Train Lightning callback usage tag for telemetry."""
        record_extra_usage_tag(TagKey.TRAIN_LIGHTNING_RAYTRAINREPORTCALLBACK, "1")

except ImportError:

    def _record_usage():
        """No-op fallback when Ray usage tracking is unavailable."""
        pass


class TrainerReportOnIntervalCallback(L.Callback):
    """PyTorch Lightning callback that reports metrics to Ray Train every N epochs.

    At each epoch whose index is divisible by ``interval`` a full checkpoint is
    saved and reported to Ray; all other epochs report metrics only.  The
    per-epoch temporary directory is removed after reporting to keep disk usage
    bounded.

    Attributes
    ----------
    CHECKPOINT_NAME : str
        Fixed filename used when saving the Lightning checkpoint inside the
        temporary directory.
    trial_name : str
        Name of the current Ray Train trial, retrieved from the train context.
    local_rank : int
        Local rank of the current worker inside the trial.
    tmpdir_prefix : str
        Root path under which per-epoch checkpoint directories are created.
    interval : int
        How often (in epochs) a checkpoint is included in the Ray report.
    step : int
        Internal counter incremented after each ``on_train_epoch_end`` call.
    """

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(self, interval: int = 1) -> None:
        """
        Parameters
        ----------
        interval : int, optional
            Number of epochs between checkpoint saves. A value of ``1`` saves a
            checkpoint every epoch; ``2`` saves every other epoch, etc.
            Defaults to ``1``.
        """
        super().__init__()
        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()
        self.tmpdir_prefix = Path(tempfile.gettempdir(), self.trial_name).as_posix()
        self.interval = interval
        self.step = 0
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

        _record_usage()

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Report metrics (and optionally a checkpoint) to Ray Train.

        Called automatically by the Lightning ``Trainer`` at the end of every
        training epoch.  A checkpoint is included in the report only when
        ``self.step % self.interval == 0``; otherwise only metrics are sent.
        The temporary epoch directory is deleted on rank 0 after reporting.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            The active Lightning trainer instance.
        pl_module : lightning.pytorch.LightningModule
            The model being trained (unused directly; provided by the callback
            protocol).
        """
        metrics = trainer.callback_metrics
        metrics = {k: v.item() if hasattr(v, "item") else float(v) for k, v in metrics.items()}
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        tmpdir = Path(self.tmpdir_prefix, str(trainer.current_epoch)).as_posix()
        os.makedirs(tmpdir, exist_ok=True)

        if self.step % self.interval == 0:
            ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
            trainer.save_checkpoint(ckpt_path, weights_only=False)
            checkpoint = Checkpoint.from_directory(tmpdir)
            train.report(metrics=metrics, checkpoint=checkpoint)
        else:
            train.report(metrics=metrics)

        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)

        self.step += 1


class TrainerReportKeepOnlyLastCallback(L.Callback):
    """PyTorch Lightning callback that always reports the most recent checkpoint.

    Every epoch overwrites the single ``last/`` checkpoint directory so only
    the latest weights are kept on disk.  This minimises storage at the cost of
    not being able to restore from earlier epochs.

    Attributes
    ----------
    CHECKPOINT_NAME : str
        Fixed filename used when saving the Lightning checkpoint inside the
        temporary directory.
    trial_name : str
        Name of the current Ray Train trial, retrieved from the train context.
    local_rank : int
        Local rank of the current worker inside the trial.
    tmpdir_prefix : str
        Root path under which the ``last/`` checkpoint directory is created.
    """

    CHECKPOINT_NAME = "checkpoint.ckpt"

    def __init__(self) -> None:
        """Initialise the callback and clean up any leftover trial directory."""
        super().__init__()
        self.trial_name = train.get_context().get_trial_name()
        self.local_rank = train.get_context().get_local_rank()
        self.tmpdir_prefix = Path(tempfile.gettempdir(), self.trial_name).as_posix()
        if os.path.isdir(self.tmpdir_prefix) and self.local_rank == 0:
            shutil.rmtree(self.tmpdir_prefix)

        _record_usage()

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Overwrite the last checkpoint and report metrics to Ray Train.

        Called automatically by the Lightning ``Trainer`` at the end of every
        training epoch.  The previous epoch's checkpoint directory is removed
        before saving the new one, so at most one checkpoint exists at any time.
        The temporary directory is deleted on rank 0 after reporting.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            The active Lightning trainer instance.
        pl_module : lightning.pytorch.LightningModule
            The model being trained (unused directly; provided by the callback
            protocol).
        """
        metrics = trainer.callback_metrics
        metrics = {k: v.item() if hasattr(v, "item") else float(v) for k, v in metrics.items()}
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        tmpdir = Path(self.tmpdir_prefix, "last").as_posix()

        # Delete previous epoch's checkpoint before writing the new one
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)

        ckpt_path = Path(tmpdir, self.CHECKPOINT_NAME).as_posix()
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        checkpoint = Checkpoint.from_directory(tmpdir)
        train.report(metrics=metrics, checkpoint=checkpoint)

        trainer.strategy.barrier()

        if self.local_rank == 0:
            shutil.rmtree(tmpdir)
