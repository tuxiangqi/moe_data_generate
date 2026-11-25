#!/usr/bin/env bash
# Four short requests: max_new_tokens, EOG, repetition, and one prefill/decode pair.
# Needs a Python that can import the upstream vLLM wheel, and 2 GPUs (see config tensor_parallel_size).
# Example:
#   srun -p pos --gres=gpu:2 --cpus-per-task=16 --mem=64G --time=01:00:00 \
#     env PYTHON=/path/to/python bash scripts/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
# Keep the installed v0.32 wheel. Do not put a source checkout ahead of it.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Greedy collection does not need FlashInfer's sampler. Without nvcc it
# tries to JIT that kernel during warmup and aborts engine startup.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$PYTHON" -m data_generate.collect smoke --config "${CONFIG:-config/collect.json}"
