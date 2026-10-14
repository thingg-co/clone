#!/usr/bin/env bash
# Show what the box is doing and whether a profile fits. Read-only.
#   spark/preflight.sh [model-key]
set -uo pipefail
. "$(dirname "$0")/common.sh"
load_env
key=${1:-$MODEL}

echo "== memory"
printf '  total %s GB, available %s GB\n' "$(mem_total_gb)" "$(mem_available_gb)"
echo "== GPU compute processes"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  (nvidia-smi unavailable)"
echo "== biggest resident processes"
ps -eo pid,user,rss,etime,comm --sort=-rss | head -6 | awk 'NR==1{print "  "$0;next}{printf "  %s %s %.1fGB %s %s\n",$1,$2,$3/1048576,$4,$5}'
echo "== listening ports"
ss -ltn 2>/dev/null | awk 'NR>1{print $4}' | sort -u | tr '\n' ' '; echo
echo "== profile $key"
if load_model "$key" 2>/dev/null; then
    printf '  repo %s, ctx %s, kv %s, wants %s GB free\n' "$HF_REPO" "$CTX" "$KV_TYPE" "$MIN_FREE_GB"
    if g=$("$ROOT/spark/resolve-gguf.sh" "$key" 2>/dev/null); then
        printf '  weights: %s (%s)\n' "$g" "$(du -ch "$(dirname "$g")"/*.gguf | tail -1 | cut -f1)"
    else
        echo "  weights: not downloaded"
    fi
    if [ "$(mem_available_gb)" -ge "$MIN_FREE_GB" ]; then echo "  fits: yes"; else echo "  fits: NO"; fi
fi
