#!/usr/bin/env bash
set -euo pipefail

# Ensure files written to the mounted /app/experimental_results directory have
# liberal read/write permissions so the host user on Fedora 41 / macOS can
# inspect, modify, or clean up results without root ownership issues.
umask 0000

RESULTS_DIR="${RESULTS_DIR:-/app/experimental_results}"
mkdir -p "${RESULTS_DIR}/tensorboard" "${RESULTS_DIR}/checkpoints" "${RESULTS_DIR}/.hf_cache"
chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true

# Launch TensorBoard in the background on port 6006 pointing at the mounted volume
if [ "${ENABLE_TENSORBOARD:-1}" = "1" ]; then
    pixi run --locked tensorboard \
        --logdir "${RESULTS_DIR}/tensorboard" \
        --host 0.0.0.0 \
        --port 6006 \
        >/dev/null 2>&1 &
fi

# Run the adversarial co-evolution experiment, forwarding any extra CLI arguments
exec pixi run --locked python -m src.main --output-dir "${RESULTS_DIR}" "$@"
