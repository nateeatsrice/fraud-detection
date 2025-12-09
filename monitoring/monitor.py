"""
Monitoring utilities for the fraud detection project.

This module uses the [Evidently](https://evidently.ai/) library to generate
data drift and performance reports.  The primary function
`generate_monitoring_report` accepts a CSV file, optionally a baseline
dataset, and produces an HTML report that can be visualised in Grafana or
any other dashboarding tool.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metrics import (
        DataDriftPreset,
        ClassificationPerformancePreset,
        RegressionPerformancePreset,
    )
except ImportError:
    # Evidently is an optional dependency; this module will not function without it.
    Report = None  # type: ignore


def generate_monitoring_report(
    data_path: Path,
    baseline_path: Optional[Path] = None,
    target_col: str = "fraud_label",
) -> Optional[Path]:
    """Generate an HTML monitoring report comparing current and baseline data.

    If no baseline is provided, the dataset is split in half and the first
    half is treated as baseline.  The report includes data drift metrics and
    performance metrics for the specified target variable.

    Parameters
    ----------
    data_path : Path
        Path to the CSV file containing the dataset to analyse.
    baseline_path : Path, optional
        Path to a baseline CSV file.  If omitted, the data in `data_path`
        will be split to derive a baseline.
    target_col : str
        The name of the column containing the target variable for performance
        analysis.

    Returns
    -------
    Path or None
        Path to the generated HTML report, or None if Evidently is not installed.
    """
    if Report is None:
        raise ImportError("Evidently is not installed; cannot generate monitoring reports")

    current_data = pd.read_csv(data_path)
    if baseline_path and baseline_path.exists():
        reference_data = pd.read_csv(baseline_path)
    else:
        # Split the provided data into two halves: reference and current
        mid = len(current_data) // 2
        reference_data = current_data.iloc[:mid].copy()
        current_data = current_data.iloc[mid:].copy()

    # Determine numeric and categorical columns
    numerical_features = [c for c in current_data.columns if c.startswith("feature_")]
    classification_targets = {"fraud_label", "chargeback_label", "takeover_label"}
    column_mapping = ColumnMapping(
        prediction=None,
        numerical_features=numerical_features,
        target=target_col,
    )

    # Choose performance preset based on target type
    if target_col in classification_targets:
        performance_metric = ClassificationPerformancePreset()
    else:
        performance_metric = RegressionPerformancePreset()

    report = Report(metrics=[DataDriftPreset(), performance_metric])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    output_dir = Path("monitoring_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = output_dir / f"report_{target_col}_{timestamp}.html"
    report.save_html(str(report_path))
    return report_path