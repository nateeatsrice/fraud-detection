"""
Integration tests for the overall pipeline.

These tests run the Prefect flow on a very small dataset to ensure that the
orchestration layer correctly calls the data generator, training script and
monitoring components.  The purpose of this test is to catch integration
issues between modules rather than validate model quality.
"""

from orchestration.flow import fraud_detection_flow


def test_prefect_flow_runs():
    # Run the flow with a small number of samples and a single model to speed up execution.
    # If this call raises no exception the test passes.
    fraud_detection_flow(n_samples=200, test_size=0.3, random_state=123, models="RandomForest")