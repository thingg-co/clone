#!/usr/bin/env bash
# Sync this tree to the Spark and (optionally) run the bootstrap there.
#   scripts/deploy.sh                 # rsync only
#   scripts/deploy.sh --bootstrap     # rsync, then spark/bootstrap.sh with the remaining flags
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$ROOT/config/spark.env" ] || { echo "copy config/spark.env.example to config/spark.env first" >&2; exit 1; }
set -a; . "$ROOT/config/spark.env"; set +a
target="${SPARK_USER}@${SPARK_HOST}"
remote="${REMOTE_DIR:-Development/frank}"
run_bootstrap=0; flags=()
for a in "$@"; do case "$a" in --bootstrap) run_bootstrap=1 ;; *) flags+=("$a") ;; esac; done

echo "syncing to $target:~/$remote/"
ssh "$target" "mkdir -p ~/$remote"
rsync -az --delete \
    --exclude /.git --exclude /models/ --exclude /llama.cpp/ --exclude /.tools/ --exclude /run/ --exclude /logs/ --exclude /third_party/ \
    --exclude /agent/.venv/ --exclude /agent/.frank/ --exclude __pycache__/ --exclude .pytest_cache/ \
    --exclude /config/spark.env \
    "$ROOT/" "$target:~/$remote/"
# spark.env is per-machine state: the control page and vllm.sh switch rewrite the
# Spark's copy. Send ours only when the Spark has none yet.
rsync -az --ignore-existing "$ROOT/config/spark.env" "$target:~/$remote/config/spark.env"
if [ "$run_bootstrap" = 1 ]; then
    ssh -t "$target" "bash ~/$remote/spark/bootstrap.sh ${flags[*]:-}"
fi
