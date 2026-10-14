#!/usr/bin/env bash
# Print the GGUF path for a model key (the first shard if the file is split).
# Exit 1 with a hint if the model has not been downloaded.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
load_model "${1:?usage: resolve-gguf.sh <model-key>}"

dir="$MODELS_DIR/$MODEL_KEY"
[ -d "$dir" ] || die "$dir does not exist; run spark/pull-model.sh $MODEL_KEY"

mapfile -t files < <(find "$dir" -type f -iname '*.gguf' -ipath "*${GGUF_MATCH}*" 2>/dev/null | sort)
[ "${#files[@]}" -gt 0 ] || die "no .gguf matching '$GGUF_MATCH' under $dir; run spark/pull-model.sh $MODEL_KEY"

for f in "${files[@]}"; do
    case "$f" in *-00001-of-*) echo "$f"; exit 0 ;; esac
done
if [ "${#files[@]}" -eq 1 ]; then echo "${files[0]}"; exit 0; fi
# several unsplit candidates: ambiguous, make the user tighten GGUF_MATCH
printf 'candidates:\n' >&2; printf '  %s\n' "${files[@]}" >&2
die "several files match GGUF_MATCH='$GGUF_MATCH'; make it more specific in config/models/$MODEL_KEY.env"
