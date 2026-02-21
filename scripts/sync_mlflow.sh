#!/usr/bin/env bash
# =============================================================================
# scripts/sync_mlflow.sh
# =============================================================================
# Syncs the local MLflow SQLite database (metadata) to/from S3.
#
# Why this exists:
#   Your mlflow.db contains experiment metadata (params, metrics, run IDs).
#   Since Codespaces are ephemeral, this file could be lost when your
#   Codespace is rebuilt.  This script lets you push/pull the metadata
#   to the same long-lived S3 bucket that stores your artifacts.
#
# Usage:
#   ./scripts/sync_mlflow.sh push    # Upload local mlflow.db to S3
#   ./scripts/sync_mlflow.sh pull    # Download mlflow.db from S3 to local
#   ./scripts/sync_mlflow.sh status  # Check if local and remote are in sync
#
# The artifacts (model files) are already in S3 — this script only handles
# the metadata database that the tracking store uses.
# =============================================================================

set -euo pipefail

# Configuration — these match your train.py defaults
BUCKET="${MLFLOW_S3_BUCKET:-nateeatsrice-mlflow}"
PREFIX="${MLFLOW_S3_PREFIX:-fraud-detection}"
S3_PATH="s3://${BUCKET}/${PREFIX}/mlflow.db"

# Local database path — same as what train.py creates
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_DB="${PROJECT_ROOT}/mlflow.db"

usage() {
    echo "Usage: $0 {push|pull|status}"
    echo ""
    echo "  push    Upload local mlflow.db → S3"
    echo "  pull    Download S3 mlflow.db → local"
    echo "  status  Compare local and remote timestamps"
    exit 1
}

push() {
    if [[ ! -f "$LOCAL_DB" ]]; then
        echo "ERROR: No local mlflow.db found at ${LOCAL_DB}"
        echo "       Run training first to create it."
        exit 1
    fi
    echo "Uploading ${LOCAL_DB} → ${S3_PATH} ..."
    aws s3 cp "$LOCAL_DB" "$S3_PATH"
    echo "Done. Metadata synced to S3."
}

pull() {
    echo "Downloading ${S3_PATH} → ${LOCAL_DB} ..."
    if aws s3 cp "$S3_PATH" "$LOCAL_DB" 2>/dev/null; then
        echo "Done. Metadata restored from S3."
    else
        echo "No remote mlflow.db found at ${S3_PATH}."
        echo "This is expected if you haven't pushed yet."
    fi
}

status() {
    echo "Local:  ${LOCAL_DB}"
    if [[ -f "$LOCAL_DB" ]]; then
        local_size=$(wc -c < "$LOCAL_DB" | tr -d ' ')
        local_mod=$(date -r "$LOCAL_DB" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || stat -c "%y" "$LOCAL_DB" 2>/dev/null)
        echo "  Size: ${local_size} bytes"
        echo "  Modified: ${local_mod}"
    else
        echo "  Not found"
    fi

    echo ""
    echo "Remote: ${S3_PATH}"
    if aws s3 ls "$S3_PATH" 2>/dev/null; then
        : # aws s3 ls already prints size and date
    else
        echo "  Not found"
    fi
}

# --- Main ---
if [[ $# -ne 1 ]]; then
    usage
fi

case "$1" in
    push)   push   ;;
    pull)   pull   ;;
    status) status ;;
    *)      usage  ;;
esac


