#!/usr/bin/env bash
# Full collection. Edit config/collect.json for the per-bucket count and paths.
# This script does not submit a Slurm job; run it on a node that already has
# the GPUs and a Python environment with this vLLM tree.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
# Keep the installed v0.32 wheel. Do not put a source checkout ahead of it.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export MOE_PREDICTION_CAPTURE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$PYTHON" -m data_generate.collect run --config "${CONFIG:-config/collect.json}"
