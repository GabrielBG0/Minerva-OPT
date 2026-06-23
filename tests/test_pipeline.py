import warnings
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import lightning.pytorch as L
import pytest
import torch


class _ToyModel(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        return cls()


class TestRayHyperParameterSearchInit:
    def test_stores_model_and_search_space(self):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        search_space = {"lr": 1e-3}
        pipeline = RayHyperParameterSearch(model=_ToyModel, search_space=search_space)
        assert pipeline.model is _ToyModel
        assert pipeline.search_space == search_space

    def test_last_results_starts_as_none(self):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        pipeline = RayHyperParameterSearch(model=_ToyModel, search_space={})
        assert pipeline._last_results is None


class TestRayHyperParameterSearchTest:
    def _make_pipeline(self):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        return RayHyperParameterSearch(model=_ToyModel, search_space={})

    def test_raises_without_search_or_ckpt(self):
        pipeline = self._make_pipeline()
        with pytest.raises(RuntimeError, match="No search results available"):
            pipeline._test(MagicMock())

    def test_uses_provided_ckpt_path(self, tmp_path):
        pipeline = self._make_pipeline()
        ckpt = tmp_path / "model.ckpt"
        ckpt.touch()

        mock_trainer = MagicMock()
        with patch("lightning.pytorch.Trainer", return_value=mock_trainer):
            pipeline._test(MagicMock(), ckpt_path=str(ckpt))

        mock_trainer.test.assert_called_once()

    def test_uses_best_checkpoint_from_results(self, tmp_path):
        pipeline = self._make_pipeline()
        ckpt_file = tmp_path / "checkpoint.ckpt"
        ckpt_file.touch()

        mock_checkpoint = MagicMock()
        mock_checkpoint.__enter__ = MagicMock(return_value=str(tmp_path))
        mock_checkpoint.__exit__ = MagicMock(return_value=False)

        mock_best = MagicMock()
        mock_best.checkpoint.as_directory.return_value = mock_checkpoint

        mock_results = MagicMock()
        mock_results.get_best_result.return_value = mock_best
        pipeline._last_results = mock_results

        mock_trainer = MagicMock()
        with patch("lightning.pytorch.Trainer", return_value=mock_trainer):
            pipeline._test(MagicMock())

        mock_trainer.test.assert_called_once()


class TestScalingConfigAutoDetect:
    def _make_pipeline(self):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        return RayHyperParameterSearch(model=_ToyModel, search_space={})

    def test_warns_on_cpu(self):
        pipeline = self._make_pipeline()

        mock_results = MagicMock()
        mock_results.get_best_result.return_value = MagicMock(
            config={}, metrics={"val_loss": 0.5}
        )

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("minerva_opt.pipelines.hyperparameter_search.TorchTrainer"),
            patch("minerva_opt.pipelines.hyperparameter_search.tune") as mock_tune,
        ):
            mock_tune.Tuner.return_value.fit.return_value = mock_results

            with pytest.warns(UserWarning, match="No GPU detected"):
                pipeline._search(MagicMock(), ckpt_path=None)

    def test_no_warning_on_gpu(self):
        pipeline = self._make_pipeline()

        mock_results = MagicMock()
        mock_results.get_best_result.return_value = MagicMock(
            config={}, metrics={"val_loss": 0.5}
        )

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("minerva_opt.pipelines.hyperparameter_search.TorchTrainer"),
            patch("minerva_opt.pipelines.hyperparameter_search.tune") as mock_tune,
        ):
            mock_tune.Tuner.return_value.fit.return_value = mock_results

            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                pipeline._search(MagicMock(), ckpt_path=None)

    def test_no_warning_with_explicit_scaling_config(self):
        from ray.train import ScalingConfig

        pipeline = self._make_pipeline()
        explicit = ScalingConfig(num_workers=1, use_gpu=False)

        mock_results = MagicMock()
        mock_results.get_best_result.return_value = MagicMock(
            config={}, metrics={"val_loss": 0.5}
        )

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("minerva_opt.pipelines.hyperparameter_search.TorchTrainer"),
            patch("minerva_opt.pipelines.hyperparameter_search.tune") as mock_tune,
        ):
            mock_tune.Tuner.return_value.fit.return_value = mock_results

            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                pipeline._search(MagicMock(), ckpt_path=None, scaling_config=explicit)


class TestDataFactory:
    def test_data_factory_forwarded_to_search(self):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        pipeline = RayHyperParameterSearch(model=_ToyModel, search_space={})
        factory = MagicMock(return_value=MagicMock(spec=L.LightningDataModule))
        mock_results = MagicMock()

        with patch.object(
            pipeline, "_search", return_value=mock_results
        ) as mock_search:
            pipeline._run(MagicMock(), task="search", data_factory=factory)

        _, kwargs = mock_search.call_args
        assert kwargs["data_factory"] is factory


class TestInvalidTask:
    def test_invalid_task_raises_value_error(self):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        pipeline = RayHyperParameterSearch(model=_ToyModel, search_space={})
        with pytest.raises(ValueError, match="Unknown task"):
            pipeline._run(MagicMock(), task="typo")


class TestStoragePath:
    def test_default_run_config_uses_log_dir(self, tmp_path):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        pipeline = RayHyperParameterSearch(model=_ToyModel, search_space={}, log_dir=tmp_path)

        mock_results = MagicMock()
        mock_results.get_best_result.return_value = MagicMock(
            config={}, metrics={"val_loss": 0.5}
        )

        captured = {}

        def fake_torch_trainer(fn, scaling_config, run_config):
            captured["run_config"] = run_config
            return MagicMock()

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "minerva_opt.pipelines.hyperparameter_search.TorchTrainer",
                side_effect=fake_torch_trainer,
            ),
            patch("minerva_opt.pipelines.hyperparameter_search.tune") as mock_tune,
            pytest.warns(UserWarning),
        ):
            mock_tune.Tuner.return_value.fit.return_value = mock_results
            pipeline._search(MagicMock(), ckpt_path=None)

        assert captured["run_config"].storage_path == str(tmp_path)


class TestTestConfigurability:
    def _make_pipeline(self):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        return RayHyperParameterSearch(model=_ToyModel, search_space={})

    def test_custom_accelerator_and_devices_passed_to_trainer(self, tmp_path):
        pipeline = self._make_pipeline()
        ckpt = tmp_path / "model.ckpt"
        ckpt.touch()

        mock_trainer = MagicMock()
        with patch(
            "minerva_opt.pipelines.hyperparameter_search.L.Trainer",
            return_value=mock_trainer,
        ) as mock_trainer_cls:
            pipeline._test(MagicMock(), ckpt_path=str(ckpt), accelerator="cpu", devices=2)

        _, kwargs = mock_trainer_cls.call_args
        assert kwargs["accelerator"] == "cpu"
        assert kwargs["devices"] == 2

    def test_test_kwargs_flow_through_run(self, tmp_path):
        pipeline = self._make_pipeline()
        ckpt = tmp_path / "model.ckpt"
        ckpt.touch()

        mock_trainer = MagicMock()
        with patch(
            "minerva_opt.pipelines.hyperparameter_search.L.Trainer",
            return_value=mock_trainer,
        ) as mock_trainer_cls:
            pipeline._run(
                MagicMock(), task="test", ckpt_path=str(ckpt), accelerator="cpu"
            )

        _, kwargs = mock_trainer_cls.call_args
        assert kwargs["accelerator"] == "cpu"


class TestFlexibilityParams:
    def _make_pipeline(self):
        from minerva_opt.pipelines.hyperparameter_search import RayHyperParameterSearch

        return RayHyperParameterSearch(model=_ToyModel, search_space={})

    def _run_search(self, pipeline, **extra):
        mock_results = MagicMock()
        mock_results.get_best_result.return_value = MagicMock(
            config={}, metrics={"val_loss": 0.5}
        )
        captured = {}

        def fake_torch_trainer(fn, scaling_config, run_config):
            captured["scaling_config"] = scaling_config
            captured["run_config"] = run_config
            return MagicMock()

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "minerva_opt.pipelines.hyperparameter_search.TorchTrainer",
                side_effect=fake_torch_trainer,
            ),
            patch("minerva_opt.pipelines.hyperparameter_search.tune") as mock_tune,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", UserWarning)
            mock_tune.Tuner.return_value.fit.return_value = mock_results
            pipeline._search(MagicMock(), ckpt_path=None, **extra)

        return captured

    def test_num_checkpoints_to_keep_forwarded(self):
        pipeline = self._make_pipeline()
        captured = self._run_search(pipeline, num_checkpoints_to_keep=3)
        assert captured["run_config"].checkpoint_config.num_to_keep == 3

    def test_resources_per_worker_forwarded_on_gpu(self):
        pipeline = self._make_pipeline()
        mock_results = MagicMock()
        mock_results.get_best_result.return_value = MagicMock(
            config={}, metrics={"val_loss": 0.5}
        )
        captured = {}

        def fake_torch_trainer(fn, scaling_config, run_config):
            captured["scaling_config"] = scaling_config
            return MagicMock()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "minerva_opt.pipelines.hyperparameter_search.TorchTrainer",
                side_effect=fake_torch_trainer,
            ),
            patch("minerva_opt.pipelines.hyperparameter_search.tune") as mock_tune,
        ):
            mock_tune.Tuner.return_value.fit.return_value = mock_results
            pipeline._search(
                MagicMock(), ckpt_path=None, resources_per_worker={"GPU": 0.5}
            )

        assert captured["scaling_config"].resources_per_worker == {"GPU": 0.5}

    def test_restore_path_calls_tuner_restore(self):
        pipeline = self._make_pipeline()
        mock_results = MagicMock()
        mock_results.get_best_result.return_value = MagicMock(
            config={}, metrics={"val_loss": 0.5}
        )

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("minerva_opt.pipelines.hyperparameter_search.TorchTrainer"),
            patch("minerva_opt.pipelines.hyperparameter_search.tune") as mock_tune,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", UserWarning)
            mock_tune.Tuner.restore.return_value.fit.return_value = mock_results
            pipeline._search(MagicMock(), ckpt_path=None, restore_path="/some/path")

        mock_tune.Tuner.restore.assert_called_once()
        call_kwargs = mock_tune.Tuner.restore.call_args[1]
        assert call_kwargs["path"] == "/some/path"
        assert call_kwargs["resume_unfinished"] is True
        mock_tune.Tuner.assert_not_called()
