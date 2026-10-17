#!/usr/bin/env bash
# Build llama.cpp with CUDA for the GB10 (compute capability 12.1).
# No sudo: DGX OS ships CUDA 13 at /usr/local/cuda, gcc 13 and cmake 3.28.
# The checkout and build stay under $ROOT/llama.cpp.
#
#   LLAMA_REF=b7500 spark/build-llama.sh    # pin a release tag instead of master
#   CUDA_ARCHS=native spark/build-llama.sh  # if cmake rejects 121
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
LLAMA_REF=${LLAMA_REF:-master}
CUDA_ARCHS=${CUDA_ARCHS:-121}

command -v nvcc >/dev/null || die "nvcc not on PATH and not in /usr/local/cuda/bin; is the CUDA toolkit installed?"
log "nvcc: $(nvcc --version | tail -1)"
log "cmake: $(cmake --version | head -1)"

if [ ! -d "$LLAMA_DIR/.git" ]; then
    log "cloning llama.cpp into $LLAMA_DIR"
    git clone --quiet https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --quiet --tags origin
git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_REF"
if [ "$LLAMA_REF" = master ]; then git -C "$LLAMA_DIR" pull --quiet --ff-only origin master; fi
log "llama.cpp at $(git -C "$LLAMA_DIR" describe --tags --always)"

gen=()
if command -v ninja >/dev/null; then gen=(-G Ninja); fi

# cmake refuses a build directory that was configured somewhere else, as after the tree moved.
cache="$LLAMA_DIR/build/CMakeCache.txt"
if [ -f "$cache" ] && ! grep -qxF "CMAKE_CACHEFILE_DIR:INTERNAL=$LLAMA_DIR/build" "$cache"; then
    warn "llama.cpp/build was configured in another directory; configuring it again"
    rm -rf "$cache" "$LLAMA_DIR/build/CMakeFiles"
fi

# BUILD_RPATH_USE_ORIGIN: the binaries find their libraries next to themselves, so the tree can move.
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" "${gen[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    || die "cmake configure failed. If it complains about CUDA architectures, retry with CUDA_ARCHS=native"

log "building with $(nproc) jobs (expect 10 to 20 minutes the first time)"
if ! cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-server llama-bench llama-cli; then
    warn "targeted build failed, building everything"
    cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)"
fi

for b in llama-server llama-bench llama-cli; do
    [ -x "$LLAMA_DIR/build/bin/$b" ] && log "built $LLAMA_DIR/build/bin/$b"
done
"$LLAMA_DIR/build/bin/llama-server" --version 2>&1 | head -2 || true

# Record whether this build speaks the Anthropic Messages API (matters for claude-code/).
if grep -rqs 'v1/messages' "$LLAMA_DIR/tools/server/"; then
    log "this build has /v1/messages: Claude Code can talk to it directly (see claude-code/README.md)"
    touch "$RUN_DIR/has-anthropic-api"
else
    log "no /v1/messages in this build: use the LiteLLM route in claude-code/README.md"
    rm -f "$RUN_DIR/has-anthropic-api"
fi
