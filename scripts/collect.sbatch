#!/usr/bin/env bash
# Six TP2 instances (12 GPUs): sbatch --ntasks=6 scripts/collect.sbatch
# Ten TP2 instances (20 GPUs): sbatch --ntasks=10 scripts/collect.sbatch
# TP4 override: sbatch --ntasks=2 --gpus-per-task=4 scripts/collect.sbatch
#SBATCH --job-name=moe-collect
#SBATCH --partition=pos
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/cognitive_comp/tuxiangqi/workspace/data_generate
#SBATCH --output=slurm-collect-%j.out

set -euo pipefail
export PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
export CONFIG="${CONFIG:-$PWD/config/collect.json}"
export TP_SIZE="${TP_SIZE:-${SLURM_GPUS_PER_TASK:-2}}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MOE_PREDICTION_CAPTURE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ "$TP_SIZE" != 2 && "$TP_SIZE" != 4 ]]; then
    echo 'TP_SIZE must be 2 or 4' >&2
    exit 1
fi
if [[ "$TP_SIZE" != "${SLURM_GPUS_PER_TASK:-2}" ]]; then
    echo 'TP_SIZE must match the requested --gpus-per-task' >&2
    exit 1
fi
COLLECT_OUTPUT=$("$PYTHON" -m data_generate.collect check --config "$CONFIG")
mkdir -p "$COLLECT_OUTPUT"
# Prevent overlapping submissions from writing the same dataset.
exec 9>"$COLLECT_OUTPUT/.collection.lock"
flock -n 9 || { echo 'Another collection job holds this output directory' >&2; exit 1; }
export PLAN="${PLAN:-$COLLECT_OUTPUT/selection.jsonl}"
"$PYTHON" -m data_generate.collect prepare --config "$CONFIG" --plan "$PLAN"
echo "nodes=$SLURM_NNODES instances=$SLURM_NTASKS TP=$TP_SIZE plan=$PLAN"
# Each task selects a local TP group from this node's allocated GPUs.
srun --ntasks="$SLURM_NTASKS" --gpus-per-task="$TP_SIZE" --gpu-bind=none --kill-on-bad-exit=1 \
    "$PYTHON" scripts/collection_node.py
