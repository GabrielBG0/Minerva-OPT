import warnings
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import lightning.pytorch as L
import pytest
import torch
from ray import tune


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class _ToyModel(L.LightningModule):
    def __init__(self, lr: float = 1e-3, dropout: float = 0.0, use_bn: bool = True):
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


_BASELINE = {"lr": 1e-3, "dropout": 0.2, "use_bn": True}
_ABLATIONS = {
    "no_dropout": {"dropout": 0.0},
    "no_bn": {"use_bn": False},
}


def _make_pipeline(**kwargs):
    from minerva_opt.pipelines.ablation_study import AblationStudyPipeline

    return AblationStudyPipeline(
        model=_ToyModel,
        baseline_config=_BASELINE,
        ablations=_ABLATIONS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Mock ResultGrid helpers
# ---------------------------------------------------------------------------

class _MockResult:
    def __init__(self, condition_name: str, seed: int, metrics: Dict[str, Any], error=None, checkpoint=None):
        self.config = {
            "train_loop_config": {
                "condition_name": condition_name,
                "ablation_seed": seed,
            }
        }
        self.metrics = metrics
        self.error = error
        self.checkpoint = checkpoint


class _MockResultGrid:
    def __init__(self, results):
        self._results = results

    def __iter__(self):
        return iter(self._results)

    def get_best_result(self, metric=None, mode=None):
        return self._results[0]


# ---------------------------------------------------------------------------
# TestAblationStudyPipelineInit
# ---------------------------------------------------------------------------

class TestAblationStudyPipelineInit:
    def test_baseline_always_included(self):
        p = _make_pipeline()
        assert "baseline" in p._conditions
        assert p._conditions["baseline"] == _BASELINE

    def test_ablations_merged_with_baseline(self):
        p = _make_pipeline()
        assert p._conditions["no_dropout"] == {**_BASELINE, "dropout": 0.0}
        assert p._conditions["no_bn"] == {**_BASELINE, "use_bn": False}

    def test_condition_order_baseline_first(self):
        p = _make_pipeline()
        assert list(p._conditions.keys())[0] == "baseline"

    def test_last_results_starts_as_none(self):
        p = _make_pipeline()
        assert p._last_results is None

    def test_model_stored(self):
        p = _make_pipeline()
        assert p.model is _ToyModel

    def test_baseline_key_reserved(self):
        from minerva_opt.pipelines.ablation_study import AblationStudyPipeline

        with pytest.raises(ValueError, match="reserved"):
            AblationStudyPipeline(
                model=_ToyModel,
                baseline_config=_BASELINE,
                ablations={"baseline": {"lr": 1e-2}},
            )


# ---------------------------------------------------------------------------
# TestAblationResultsSummary
# ---------------------------------------------------------------------------

class TestAblationResultsSummary:
    def _make_grid(self):
        results = [
            _MockResult("baseline", 0, {"val_loss": 0.10}),
            _MockResult("baseline", 1, {"val_loss": 0.12}),
            _MockResult("no_dropout", 0, {"val_loss": 0.18}),
            _MockResult("no_dropout", 1, {"val_loss": 0.16}),
            _MockResult("no_bn", 0, {"val_loss": 0.15}),
            _MockResult("no_bn", 1, {"val_loss": 0.13}),
        ]
        return _MockResultGrid(results)

    def _make_ablation_results(self):
        from minerva_opt.results.ablation_results import AblationResults

        grid = self._make_grid()
        return AblationResults(grid, ["baseline", "no_dropout", "no_bn"], "val_loss", "min")

    def test_summary_has_correct_conditions(self):
        ar = self._make_ablation_results()
        df = ar.summary()
        assert set(df.index) == {"baseline", "no_dropout", "no_bn"}

    def test_summary_baseline_first(self):
        ar = self._make_ablation_results()
        df = ar.summary()
        assert df.index[0] == "baseline"

    def test_summary_mean_values(self):
        import pandas as pd

        ar = self._make_ablation_results()
        df = ar.summary()
        assert "val_loss_mean" in df.columns
        assert abs(df.loc["baseline", "val_loss_mean"] - 0.11) < 1e-9
        assert abs(df.loc["no_dropout", "val_loss_mean"] - 0.17) < 1e-9

    def test_summary_std_values(self):
        import pandas as pd

        ar = self._make_ablation_results()
        df = ar.summary()
        assert "val_loss_std" in df.columns
        # std of [0.10, 0.12] ≈ 0.01414
        assert abs(df.loc["baseline", "val_loss_std"] - 0.01414) < 1e-4

    def test_summary_skips_errored_results(self):
        from minerva_opt.results.ablation_results import AblationResults

        results = [
            _MockResult("baseline", 0, {"val_loss": 0.10}),
            _MockResult("baseline", 1, {"val_loss": 0.10}, error=RuntimeError("boom")),
        ]
        grid = _MockResultGrid(results)
        ar = AblationResults(grid, ["baseline"], "val_loss", "min")
        df = ar.summary()
        # Only one valid result; std should be NaN (single sample)
        assert abs(df.loc["baseline", "val_loss_mean"] - 0.10) < 1e-9

    def test_summary_empty_on_all_errors(self):
        from minerva_opt.results.ablation_results import AblationResults

        results = [_MockResult("baseline", 0, {}, error=RuntimeError("boom"))]
        grid = _MockResultGrid(results)
        ar = AblationResults(grid, ["baseline"], "val_loss", "min")
        df = ar.summary()
        assert df.empty


# ---------------------------------------------------------------------------
# TestAblationResultsDelta
# ---------------------------------------------------------------------------

class TestAblationResultsDelta:
    def _make_results(self, mode="min"):
        from minerva_opt.results.ablation_results import AblationResults

        # baseline: 0.10, no_dropout: 0.18, no_bn: 0.14
        results = [
            _MockResult("baseline", 0, {"val_loss": 0.10}),
            _MockResult("no_dropout", 0, {"val_loss": 0.18}),
            _MockResult("no_bn", 0, {"val_loss": 0.14}),
        ]
        grid = _MockResultGrid(results)
        return AblationResults(grid, ["baseline", "no_dropout", "no_bn"], "val_loss", mode)

    def test_delta_min_positive_means_worse(self):
        ar = self._make_results(mode="min")
        delta = ar.delta_from_baseline()
        # no_dropout 0.10 - 0.18 = -0.08 (worse, negative delta)
        assert abs(delta["no_dropout"] - (-0.08)) < 1e-9

    def test_delta_max_positive_means_better(self):
        from minerva_opt.results.ablation_results import AblationResults

        results = [
            _MockResult("baseline", 0, {"val_acc": 0.90}),
            _MockResult("no_dropout", 0, {"val_acc": 0.80}),
        ]
        grid = _MockResultGrid(results)
        ar = AblationResults(grid, ["baseline", "no_dropout"], "val_acc", "max")
        delta = ar.delta_from_baseline("val_acc")
        # 0.80 - 0.90 = -0.10 (worse)
        assert abs(delta["no_dropout"] - (-0.10)) < 1e-9

    def test_delta_baseline_is_zero(self):
        ar = self._make_results(mode="min")
        delta = ar.delta_from_baseline()
        assert abs(delta["baseline"]) < 1e-9

    def test_delta_raises_on_missing_metric(self):
        ar = self._make_results(mode="min")
        with pytest.raises(KeyError, match="nonexistent"):
            ar.delta_from_baseline("nonexistent")

    def test_delta_series_name(self):
        ar = self._make_results()
        delta = ar.delta_from_baseline()
        assert delta.name == "delta_val_loss_vs_baseline"


# ---------------------------------------------------------------------------
# TestAblationResultsBestCheckpoint
# ---------------------------------------------------------------------------

class TestAblationResultsBestCheckpoint:
    def test_selects_best_seed_for_condition(self):
        from minerva_opt.results.ablation_results import AblationResults

        ckpt_good = MagicMock()
        ckpt_bad = MagicMock()
        results = [
            _MockResult("baseline", 0, {"val_loss": 0.08}, checkpoint=ckpt_good),
            _MockResult("baseline", 1, {"val_loss": 0.15}, checkpoint=ckpt_bad),
        ]
        grid = _MockResultGrid(results)
        ar = AblationResults(grid, ["baseline"], "val_loss", "min")
        assert ar.best_checkpoint("baseline") is ckpt_good

    def test_raises_on_unknown_condition(self):
        from minerva_opt.results.ablation_results import AblationResults

        results = [_MockResult("baseline", 0, {"val_loss": 0.10}, checkpoint=MagicMock())]
        grid = _MockResultGrid(results)
        ar = AblationResults(grid, ["baseline"], "val_loss", "min")
        with pytest.raises(ValueError, match="no_dropout"):
            ar.best_checkpoint("no_dropout")

    def test_raises_when_all_errored(self):
        from minerva_opt.results.ablation_results import AblationResults

        results = [_MockResult("baseline", 0, {}, error=RuntimeError("boom"))]
        grid = _MockResultGrid(results)
        ar = AblationResults(grid, ["baseline"], "val_loss", "min")
        with pytest.raises(ValueError, match="baseline"):
            ar.best_checkpoint("baseline")

    def test_max_mode_selects_highest(self):
        from minerva_opt.results.ablation_results import AblationResults

        ckpt_better = MagicMock()
        ckpt_worse = MagicMock()
        results = [
            _MockResult("baseline", 0, {"val_acc": 0.70}, checkpoint=ckpt_worse),
            _MockResult("baseline", 1, {"val_acc": 0.95}, checkpoint=ckpt_better),
        ]
        grid = _MockResultGrid(results)
        ar = AblationResults(grid, ["baseline"], "val_acc", "max")
        assert ar.best_checkpoint("baseline") is ckpt_better


# ---------------------------------------------------------------------------
# TestAblationRawProperty
# ---------------------------------------------------------------------------

class TestAblationRawProperty:
    def test_raw_returns_underlying_grid(self):
        from minerva_opt.results.ablation_results import AblationResults

        grid = _MockResultGrid([])
        ar = AblationResults(grid, [], "val_loss", "min")
        assert ar.raw is grid


# ---------------------------------------------------------------------------
# TestAblationScalingConfig
# ---------------------------------------------------------------------------

class TestAblationScalingConfig:
    def _run_ablate(self, pipeline, **extra):
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))
        captured = {}

        def fake_torch_trainer(fn, scaling_config, run_config):
            captured["scaling_config"] = scaling_config
            captured["run_config"] = run_config
            return MagicMock()

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "minerva_opt.pipelines.ablation_study.TorchTrainer",
                side_effect=fake_torch_trainer,
            ),
            patch("minerva_opt.pipelines.ablation_study.tune") as mock_tune,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", UserWarning)
            mock_tune.Tuner.return_value.fit.return_value = mock_results
            mock_tune.grid_search = tune.grid_search
            pipeline._ablate(MagicMock(), ckpt_path=None, **extra)

        return captured

    def test_warns_on_cpu(self):
        p = _make_pipeline()
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("minerva_opt.pipelines.ablation_study.TorchTrainer"),
            patch("minerva_opt.pipelines.ablation_study.tune") as mock_tune,
        ):
            mock_tune.Tuner.return_value.fit.return_value = mock_results
            mock_tune.grid_search = tune.grid_search
            with pytest.warns(UserWarning, match="No GPU detected"):
                p._ablate(MagicMock(), ckpt_path=None)

    def test_no_warning_on_gpu(self):
        p = _make_pipeline()
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("minerva_opt.pipelines.ablation_study.TorchTrainer"),
            patch("minerva_opt.pipelines.ablation_study.tune") as mock_tune,
        ):
            mock_tune.Tuner.return_value.fit.return_value = mock_results
            mock_tune.grid_search = tune.grid_search
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                p._ablate(MagicMock(), ckpt_path=None)

    def test_no_warning_with_explicit_scaling_config(self):
        from ray.train import ScalingConfig

        p = _make_pipeline()
        explicit = ScalingConfig(num_workers=1, use_gpu=False)
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("minerva_opt.pipelines.ablation_study.TorchTrainer"),
            patch("minerva_opt.pipelines.ablation_study.tune") as mock_tune,
        ):
            mock_tune.Tuner.return_value.fit.return_value = mock_results
            mock_tune.grid_search = tune.grid_search
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                p._ablate(MagicMock(), ckpt_path=None, scaling_config=explicit)


# ---------------------------------------------------------------------------
# TestAblationEarlyStopping
# ---------------------------------------------------------------------------

class TestAblationEarlyStopping:
    def test_default_scheduler_grace_period_equals_max_epochs(self):
        p = _make_pipeline()
        mock_results = MagicMock()
        mock_results.__iter__ = MagicMock(return_value=iter([]))

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("minerva_opt.pipelines.ablation_study.TorchTrainer"),
            patch("minerva_opt.pipelines.ablation_study.tune") as mock_tune,
            patch("minerva_opt.pipelines.ablation_study.ASHAScheduler") as mock_asha,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", UserWarning)
            mock_tune.Tuner.return_value.fit.return_value = mock_results
            mock_tune.TuneConfig = tune.TuneConfig
            mock_tune.grid_search = tune.grid_search
            p._ablate(MagicMock(), ckpt_path=None, max_epochs=50)

        _, kwargs = mock_asha.call_args
        assert kwargs["grace_period"] == 50
        assert kwargs["max_t"] == 50


# ---------------------------------------------------------------------------
# TestAblationInvalidTask
# ---------------------------------------------------------------------------

class TestAblationInvalidTask:
    def test_invalid_task_raises(self):
        p = _make_pipeline()
        with pytest.raises(ValueError, match="Unknown task"):
            p._run(MagicMock(), task="typo")


# ---------------------------------------------------------------------------
# TestAblationTestDispatch
# ---------------------------------------------------------------------------

class TestAblationTestDispatch:
    def test_raises_without_results_or_ckpt(self):
        p = _make_pipeline()
        with pytest.raises(RuntimeError, match="No ablation results available"):
            p._test(MagicMock())

    def test_uses_provided_ckpt_path(self, tmp_path):
        p = _make_pipeline()
        ckpt = tmp_path / "model.ckpt"
        ckpt.touch()

        mock_trainer = MagicMock()
        with patch("lightning.pytorch.Trainer", return_value=mock_trainer):
            p._test(MagicMock(), ckpt_path=str(ckpt))

        mock_trainer.test.assert_called_once()

    def test_data_factory_forwarded(self):
        p = _make_pipeline()
        factory = MagicMock(return_value=MagicMock(spec=L.LightningDataModule))
        mock_results = MagicMock()

        with patch.object(p, "_ablate", return_value=mock_results) as mock_ablate:
            p._run(MagicMock(), task="ablate", data_factory=factory)

        _, kwargs = mock_ablate.call_args
        assert kwargs["data_factory"] is factory
