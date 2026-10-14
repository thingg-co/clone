#!/usr/bin/env bash
# Forward the Spark's model ports and the control page to localhost, so the
# agent and Claude Code can use http://localhost:8080 from the Mac and the
# panel is at http://localhost:8088. Runs until Ctrl-C.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
set -a; . "$ROOT/config/spark.env"; set +a
ports=(-L "${PORT:-8080}:127.0.0.1:${PORT:-8080}")
[ -n "${SIDEKICK_PORT:-}" ] && ports+=(-L "${SIDEKICK_PORT}:127.0.0.1:${SIDEKICK_PORT}")
ports+=(-L "${PANEL_PORT:-8088}:127.0.0.1:${PANEL_PORT:-8088}")
echo "tunnel to ${SPARK_USER}@${SPARK_HOST}: localhost:${PORT:-8080}${SIDEKICK_PORT:+ and :$SIDEKICK_PORT}, panel on http://localhost:${PANEL_PORT:-8088}"
exec ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes "${ports[@]}" "${SPARK_USER}@${SPARK_HOST}"
