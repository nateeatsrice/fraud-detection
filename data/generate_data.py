"""
Synthetic dataset generator for the fraud‑detection project.

This script uses scikit‑learn to create a set of numeric features and several
correlated outcome variables.  You can adjust the number of samples
produced and the output location on disk using command‑line arguments.

Usage:
    python generate_data.py --n_samples 10000 --output data/transactions.csv

The resulting CSV contains 20 feature columns (`feature_0` … `feature_19`) and
four target columns: `fraud_label`, `chargeback_label`, `takeover_label`,
and `anomaly_score`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def generate_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic transaction dataset with multiple risk indicators.

    Parameters
    ----------
    n_samples : int
        Number of samples (rows) to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing feature columns and target variables.
    """
    rng = np.random.RandomState(random_state)

    # Generate informative features and a base fraud label
    X, fraud = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        flip_y=0.03,
        random_state=random_state,
    )

    # Create chargeback risk as a noisy version of fraud
    # Fraudulent transactions tend to result in chargebacks, but not always
    chargeback_noise = rng.binomial(1, 0.1, n_samples)
    chargeback_label = ((fraud + chargeback_noise) > 0).astype(int)

    # Account takeover risk; some frauds are takeovers, and some takeovers are legitimate
    takeover_noise = rng.binomial(1, 0.08, n_samples)
    takeover_offset = rng.binomial(1, 0.05, n_samples)
    takeover_label = ((fraud + takeover_noise - takeover_offset) > 0).astype(int)

    # Anomaly score from IsolationForest (unsupervised)
    iso = IsolationForest(contamination=0.1, random_state=random_state)
    iso.fit(X)
    # decision_function yields higher values for normal points; invert to get anomaly intensity
    raw_scores = -iso.decision_function(X)
    scaler = MinMaxScaler()
    anomaly_score = scaler.fit_transform(raw_scores.reshape(-1, 1)).ravel()

    # Assemble dataframe
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["fraud_label"] = fraud
    df["chargeback_label"] = chargeback_label
    df["takeover_label"] = takeover_label
    df["anomaly_score"] = anomaly_score
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic fraud‑detection dataset.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of rows to generate.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/transactions.csv",
        help="Path to the CSV file to save the generated data.",
    )
    args = parser.parse_args()

    df = generate_dataset(n_samples=args.n_samples)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated dataset with {len(df)} rows and saved to {output_path}")


if __name__ == "__main__":
    main()
