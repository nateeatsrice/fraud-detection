"""
AWS Lambda handler for the fraud detection project.

This is a minimal example of a Lambda function that could be wired up to an
SQS queue or API Gateway.  It simply logs the incoming event and returns
a 200 response.  Extend this function to trigger background training jobs
or other asynchronous tasks.
"""

from __future__ import annotations

import json


def lambda_handler(event, context):
    """Entry point for AWS Lambda."""
    # In a real implementation you would parse the incoming SQS message or
    # HTTP request and perform work such as triggering a training job.
    print("Received event:", json.dumps(event))
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Hello from fraud-detection Lambda"}),
    }