from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    import pandas as pd
    from ray.tune.result_grid import ResultGrid


class AblationResults:
    """Wrapper around Ray's ``ResultGrid`` providing ablation-specific analysis.

    Aggregates trial results across seeds per condition, computes metric deltas
    relative to the baseline condition, and provides checkpoint access per
    condition.

    Attributes
    ----------
    _grid : ResultGrid
        Underlying Ray ``ResultGrid`` containing all trial results.
    _condition_names : list of str
        Ordered list of condition names (baseline first).
    _metric : str
        Primary metric name used for ranking and delta computation.
    _mode : str
        ``"min"`` or ``"max"`` — optimisation direction for ``_metric``.
    """

    def __init__(
        self,
        result_grid: "ResultGrid",
        condition_names: List[str],
        metric: str,
        mode: str,
    ):
        """
        Parameters
        ----------
        result_grid : ray.tune.result_grid.ResultGrid
            Full grid of trial results produced by ``tune.Tuner.fit()``.
        condition_names : list of str
            Ordered list of condition names used to index the results
            (baseline must be first).
        metric : str
            Name of the primary evaluation metric.
        mode : str
            ``"min"`` if lower values of ``metric`` are better; ``"max"``
            otherwise.
        """
        self._grid = result_grid
        self._condition_names = condition_names
        self._metric = metric
        self._mode = mode

    @property
    def raw(self) -> "ResultGrid":
        """Underlying ``ResultGrid`` for direct Ray API access.

        Returns
        -------
        ray.tune.result_grid.ResultGrid
            The unmodified result grid produced by ``tune.Tuner.fit()``.
        """
        return self._grid

    def summary(self) -> "pd.DataFrame":
        """Compute mean and standard deviation of all metrics per condition.

        Each row corresponds to one condition in declaration order (baseline
        first).  Columns are named ``{metric}_mean`` and ``{metric}_std`` for
        every numeric metric that was logged by at least one trial.  Failed
        trials are excluded from the aggregation.

        Returns
        -------
        pandas.DataFrame
            Index: condition names.  Columns: ``{metric}_mean`` and
            ``{metric}_std``.  An empty ``DataFrame`` is returned when no
            successful trial results are available.
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
        """Compute per-condition improvement relative to the baseline.

        Positive values mean the condition performs *better* than baseline —
        lower mean loss for ``mode='min'``, higher mean score for
        ``mode='max'``.

        Parameters
        ----------
        metric : str or None, optional
            Metric to use for the comparison.  Defaults to the metric
            supplied at construction time (``self._metric``).

        Returns
        -------
        pandas.Series
            Index: condition names.  Values: signed delta versus the baseline
            mean.  Series name is ``"delta_{metric}_vs_baseline"``.

        Raises
        ------
        KeyError
            If the requested metric is not present in ``summary()`` columns.
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
        """Retrieve the best checkpoint for a given condition.

        Among all seeds for the condition, the seed whose final reported
        ``self._metric`` value is optimal (min or max, per ``self._mode``)
        is selected and its checkpoint is returned.

        Parameters
        ----------
        condition : str
            Name of the ablation condition (e.g. ``"baseline"`` or a key
            from the ``ablations`` dict).

        Returns
        -------
        ray.train.Checkpoint
            Ray checkpoint object for the winning seed of ``condition``.

        Raises
        ------
        ValueError
            If no successful trial results exist for ``condition``.
        """
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
