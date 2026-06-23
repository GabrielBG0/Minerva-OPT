import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_context(trial_name="trial-0", local_rank=0):
    ctx = MagicMock()
    ctx.get_trial_name.return_value = trial_name
    ctx.get_local_rank.return_value = local_rank
    return ctx


def _make_trainer(tmp_path, epoch=0):
    trainer = MagicMock()
    trainer.callback_metrics = {
        "val_loss": MagicMock(item=MagicMock(return_value=0.42))
    }
    trainer.current_epoch = epoch
    trainer.global_step = epoch * 10
    trainer.save_checkpoint = MagicMock(
        side_effect=lambda path, **kw: Path(path).touch()
    )
    trainer.strategy = MagicMock()
    return trainer


@pytest.fixture(autouse=True)
def _patch_ray(tmp_path):
    ctx = _make_context(trial_name=tmp_path.name)
    with (
        patch("ray.train.get_context", return_value=ctx),
        patch("ray.train.report"),
    ):
        yield


class TestTrainerReportOnIntervalCallback:
    def test_reports_with_checkpoint_on_interval(self, tmp_path):
        import ray.train as rt

        from minerva_opt.callbacks.ray_callbacks import TrainerReportOnIntervalCallback

        cb = TrainerReportOnIntervalCallback(interval=2)
        trainer = _make_trainer(tmp_path, epoch=0)

        # step=0 → 0 % 2 == 0 → checkpoint
        cb.on_train_epoch_end(trainer, MagicMock())
        _, kwargs = rt.report.call_args
        assert kwargs.get("checkpoint") is not None

    def test_reports_without_checkpoint_between_intervals(self, tmp_path):
        import ray.train as rt

        from minerva_opt.callbacks.ray_callbacks import TrainerReportOnIntervalCallback

        cb = TrainerReportOnIntervalCallback(interval=2)
        trainer = _make_trainer(tmp_path, epoch=0)

        # step=0 → checkpoint; step=1 → no checkpoint
        cb.on_train_epoch_end(trainer, MagicMock())
        rt.report.reset_mock()
        trainer.current_epoch = 1
        cb.on_train_epoch_end(trainer, MagicMock())
        _, kwargs = rt.report.call_args
        assert kwargs.get("checkpoint") is None

    def test_metrics_include_epoch_and_step(self, tmp_path):
        import ray.train as rt

        from minerva_opt.callbacks.ray_callbacks import TrainerReportOnIntervalCallback

        cb = TrainerReportOnIntervalCallback(interval=1)
        trainer = _make_trainer(tmp_path, epoch=3)

        cb.on_train_epoch_end(trainer, MagicMock())
        reported_metrics = rt.report.call_args[1]["metrics"]
        assert reported_metrics["epoch"] == 3
        assert "val_loss" in reported_metrics


    def test_non_tensor_metrics_do_not_crash(self, tmp_path):
        from minerva_opt.callbacks.ray_callbacks import TrainerReportOnIntervalCallback

        import ray.train as rt

        cb = TrainerReportOnIntervalCallback(interval=1)
        trainer = _make_trainer(tmp_path, epoch=0)
        # Override with a plain Python float (no .item())
        trainer.callback_metrics = {"val_loss": 0.99}

        cb.on_train_epoch_end(trainer, MagicMock())
        reported = rt.report.call_args[1]["metrics"]
        assert reported["val_loss"] == 0.99


class TestTrainerReportKeepOnlyLastCallback:
    def test_always_reports_with_checkpoint(self, tmp_path):
        import ray.train as rt

        from minerva_opt.callbacks.ray_callbacks import (
            TrainerReportKeepOnlyLastCallback,
        )

        cb = TrainerReportKeepOnlyLastCallback()
        trainer = _make_trainer(tmp_path, epoch=0)

        cb.on_train_epoch_end(trainer, MagicMock())
        _, kwargs = rt.report.call_args
        assert kwargs.get("checkpoint") is not None

    def test_replaces_previous_checkpoint(self, tmp_path):
        import ray.train as rt

        from minerva_opt.callbacks.ray_callbacks import (
            TrainerReportKeepOnlyLastCallback,
        )

        cb = TrainerReportKeepOnlyLastCallback()
        trainer = _make_trainer(tmp_path, epoch=0)

        cb.on_train_epoch_end(trainer, MagicMock())
        trainer.current_epoch = 1
        cb.on_train_epoch_end(trainer, MagicMock())

        assert rt.report.call_count == 2

    def test_non_tensor_metrics_do_not_crash(self, tmp_path):
        import ray.train as rt

        from minerva_opt.callbacks.ray_callbacks import (
            TrainerReportKeepOnlyLastCallback,
        )

        cb = TrainerReportKeepOnlyLastCallback()
        trainer = _make_trainer(tmp_path, epoch=0)
        trainer.callback_metrics = {"val_loss": 0.42}

        cb.on_train_epoch_end(trainer, MagicMock())
        reported = rt.report.call_args[1]["metrics"]
        assert reported["val_loss"] == 0.42
