#!/usr/bin/env bash
# What the Spark is doing right now: services, memory, GPU, model endpoints.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
set -a; . "$ROOT/config/spark.env"; set +a
ssh "${SPARK_USER}@${SPARK_HOST}" "bash ~/${REMOTE_DIR:-Development/clone}/spark/preflight.sh; echo '== services'; bash ~/${REMOTE_DIR:-Development/clone}/spark/service.sh status; systemctl --user --no-pager status vllm.service 2>/dev/null | grep -E 'Active' || true"
