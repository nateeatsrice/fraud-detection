"""
Prefect orchestration for the fraud detection pipeline.

This module defines a Prefect flow that orchestrates dataset generation,
model training and monitoring.  It provides a simple example of how to
compose the tasks defined in other modules into a reusable workflow.  The
flow can be run locally or deployed to a Prefect server/cloud for
scheduling and observability.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from prefect import flow, task, get_run_logger

from data.generate_data import generate_dataset
from src.train import main as train_main
from monitoring.monitor import generate_monitoring_report


@task
def generate_data_task(n_samples: int, output_path: Path) -> str:
    logger = get_run_logger()
    logger.info(f"Generating dataset with {n_samples} samples to {output_path}")
    df = generate_dataset(n_samples=n_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path)


@task
def train_models_task(data_path: str, test_size: float, random_state: int, models: str) -> None:
    logger = get_run_logger()
    logger.info("Training models")
    # Invoke the training script via its main() function.  We build a fake
    # argument list to simulate CLI invocation.
    import sys
    argv_backup = sys.argv
    try:
        sys.argv = [
            "train.py",
            "--data",
            data_path,
            "--test_size",
            str(test_size),
            "--random_state",
            str(random_state),
            "--models",
            models,
        ]
        train_main()
    finally:
        sys.argv = argv_backup


@task
def monitoring_task(data_path: str) -> Optional[str]:
    logger = get_run_logger()
    logger.info("Running monitoring checks")
    # Generate a basic monitoring report comparing the new data against a baseline.
    report_path = generate_monitoring_report(Path(data_path))
    logger.info(f"Monitoring report saved to {report_path}")
    return str(report_path)


@flow(name="fraud_detection_flow")
def fraud_detection_flow(
    *,
    n_samples: int = 10000,
    test_size: float = 0.2,
    random_state: int = 42,
    models: str = "all",
) -> None:
    """Orchestrate the fraud detection pipeline.

    Parameters
    ----------
    n_samples : int
        Number of rows to generate for the synthetic dataset.
    test_size : float
        Proportion of the dataset reserved for the test split during training.
    random_state : int
        Seed for reproducibility.
    models : str
        Comma‑separated list of models to train, or 'all' for all supported models.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "transactions.csv"
        data_path = generate_data_task(n_samples=n_samples, output_path=data_file)
        train_models_task(data_path=data_path, test_size=test_size, random_state=random_state, models=models)
        monitoring_task(data_path=data_path)


if __name__ == "__main__":
    fraud_detection_flow()