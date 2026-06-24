from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    import pandas as pd
    from ray.tune.result_grid import ResultGrid


class AblationResults:
    """Wrapper around Ray's ResultGrid providing ablation-specific analysis.

    Aggregates results across seeds per condition, computes deltas vs baseline,
    and provides checkpoint access per condition.
    """

    def __init__(
        self,
        result_grid: "ResultGrid",
        condition_names: List[str],
        metric: str,
        mode: str,
    ):
        self._grid = result_grid
        self._condition_names = condition_names
        self._metric = metric
        self._mode = mode

    @property
    def raw(self) -> "ResultGrid":
        """Underlying ResultGrid for direct Ray API access."""
        return self._grid

    def summary(self) -> "pd.DataFrame":
        """DataFrame of mean ± std per condition across seeds.

        Rows are conditions in declaration order (baseline first).
        Columns are ``{metric}_mean`` and ``{metric}_std`` for all logged metrics.
        """
        import pandas as pd

        rows = []
        for result in self._grid:
            if result.error:
                continue
            cfg = result.config["train_loop_config"]
            condition = cfg["condition_name"]
            rows.append({"condition": condition, **(result.metrics or {})})

        if not rows:
            return pd.DataFrame()

        df_raw = pd.DataFrame(rows)
        numeric_cols = df_raw.select_dtypes(include="number").columns
        agg = df_raw.groupby("condition")[numeric_cols].agg(["mean", "std"])
        agg.columns = ["_".join(col) for col in agg.columns]
        return agg.reindex(self._condition_names)

    def delta_from_baseline(self, metric: Optional[str] = None) -> "pd.Series":
        """Per-condition improvement vs baseline on ``metric``.

        Positive values mean the condition is *better* than baseline
        (lower loss for ``mode='min'``, higher score for ``mode='max'``).
        """
        metric = metric or self._metric
        df = self.summary()
        mean_col = f"{metric}_mean"
        if mean_col not in df.columns:
            raise KeyError(
                f"Metric {metric!r} not found. Available: {list(df.columns)}"
            )
        baseline_val = df.loc["baseline", mean_col]
        if self._mode == "min":
            delta = baseline_val - df[mean_col]
        else:
            delta = df[mean_col] - baseline_val
        delta.name = f"delta_{metric}_vs_baseline"
        return delta

    def best_checkpoint(self, condition: str) -> Any:
        """Best Ray Checkpoint for ``condition`` (seed with best tuner metric)."""
        condition_results = [
            r
            for r in self._grid
            if not r.error
            and r.config["train_loop_config"]["condition_name"] == condition
        ]
        if not condition_results:
            raise ValueError(f"No successful results for condition {condition!r}")

        if self._mode == "min":
            best = min(
                condition_results,
                key=lambda r: r.metrics.get(self._metric, float("inf")),
            )
        else:
            best = max(
                condition_results,
                key=lambda r: r.metrics.get(self._metric, float("-inf")),
            )
        return best.checkpoint
