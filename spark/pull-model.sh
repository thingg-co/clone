#!/usr/bin/env bash
# Download a model profile's GGUF files into $MODELS_DIR/<key>.
# Resumable and idempotent: rerun it after an interruption.
#
#   spark/pull-model.sh gpt-oss-120b
#   spark/pull-model.sh gpt-oss-120b gpt-oss-20b     # several
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
command -v hf >/dev/null || die "hf CLI missing; run spark/bootstrap.sh (it installs it under .tools/)"

keys=("$@"); [ "${#keys[@]}" -gt 0 ] || keys=("$MODEL")

for key in "${keys[@]}"; do
    load_model "$key"
    code=$(curl -s -o /dev/null -w '%{http_code}' "https://huggingface.co/api/models/$HF_REPO")
    [ "$code" = 200 ] || die "HF repo $HF_REPO returned HTTP $code; fix HF_REPO in config/models/$key.env (browse https://huggingface.co/$HF_REPO)"

    # size of what we are about to fetch, from the repo tree listing
    gb=$(curl -s "https://huggingface.co/api/models/$HF_REPO/tree/main?recursive=true" | python3 -I -c '
import sys, json, fnmatch
pat = sys.argv[1]
items = json.load(sys.stdin)
tot = sum(i.get("size", 0) for i in items if i.get("type") == "file" and fnmatch.fnmatch(i["path"].split("/")[-1], pat))
print(f"{tot/1e9:.1f}")' "$HF_INCLUDE" 2>/dev/null || echo "?")
    dest="$MODELS_DIR/$key"
    mkdir -p "$dest"
    log "$key: $HF_REPO ($HF_INCLUDE, ~${gb} GB) -> $dest"
    hf download "$HF_REPO" --include "$HF_INCLUDE" --local-dir "$dest"
    gguf=$("$ROOT/spark/resolve-gguf.sh" "$key")
    log "$key ready: $gguf"
done
