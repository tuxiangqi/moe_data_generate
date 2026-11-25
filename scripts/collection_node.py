"""Run one collector per Slurm task, using one local TP group per task."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from data_generate.collect import load_config


def task_devices(devices: list[str], tp: int, local_rank: int) -> list[str]:
    if tp not in (2, 4) or not all(devices):
        raise ValueError("require allocated GPUs and TP_SIZE 2 or 4")
    if len(devices) == tp:
        return devices
    # --gpu-bind=none exposes all allocated GPUs on this node.
    if len(devices) % tp or (local_rank + 1) * tp > len(devices):
        raise ValueError(f"cannot assign local task {local_rank} TP{tp} from {devices}")
    return devices[local_rank * tp:(local_rank + 1) * tp]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tp = int(os.environ["TP_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    tasks = int(os.environ["SLURM_NTASKS"])
    visible = ",".join(task_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","), tp,
        int(os.environ["SLURM_LOCALID"]),
    ))
    config_path = Path(os.environ["CONFIG"]).resolve()
    config = load_config(config_path)
    log_root = Path(config["output"]) / "logs" / os.environ["SLURM_JOB_ID"]
    command = [sys.executable, "-m", "data_generate.collect", "run",
               "--config", str(config_path), "--plan", os.environ["PLAN"],
               "--tp-size", str(tp), "--shard-index", str(rank), "--num-shards", str(tasks)]
    if os.environ.get("BATCH_SIZE"):
        command.extend(["--batch-size", os.environ["BATCH_SIZE"]])
    print(json.dumps({"shard": rank, "num_shards": tasks, "gpus": visible,
                      "command": command}), flush=True)
    if args.dry_run:
        return

    def interrupted(signum, frame):
        raise RuntimeError(f"collection interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / f"shard-{rank}.log").open("w") as log:
        proc = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible),
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait()
            if code:
                raise RuntimeError(f"collector exited with {code}; inspect {log_root}/shard-{rank}.log")
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


if __name__ == "__main__":
    main()
