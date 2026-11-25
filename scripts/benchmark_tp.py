"""Compare two TP2 collectors with one TP4 collector on eight allocated GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def worker(args):
    from data_generate.collect import (
        first_short_prompt, generate_one, load_config, make_engine, sampling_params,
    )

    folder = args.output / args.worker
    folder.mkdir(parents=True)
    config = load_config(ROOT / "config/collect.json")
    tp = 4 if args.worker == "tp4" else 2
    config.update(tensor_parallel_size=tp, max_context=4096)
    started = time.perf_counter()
    llm = make_engine(config, folder / "captures")
    engine_seconds = time.perf_counter() - started
    prompt = first_short_prompt(config, llm.get_tokenizer())
    # Force equal decode lengths and token trajectories, as in the smoke test.
    # Routing/forward/capture use real model computations; these are not answers.
    trace = list(range(100, 117))

    def request():
        return generate_one(
            llm, config, folder / "captures", prompt,
            sampling_params(config, 16, trace=trace), 16,
        )

    request()
    write_json(folder / "ready.json", {
        "tp": tp, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "engine_seconds": engine_seconds,
        "startup_with_warmup_seconds": time.perf_counter() - started,
        "prompt_token_ids": prompt, "decode_token_ids": trace[:16],
    })
    count = 8 if tp == 4 else 4
    for round_id in range(3):
        while not (args.output / f"go-{round_id}").exists():
            time.sleep(0.02)
        start = time.perf_counter()
        results = []
        for _ in range(count):
            tick = time.perf_counter()
            meta = request()
            assert meta["prompt_len"] == 32 and meta["generated_len"] == 16
            results.append({"seconds": time.perf_counter() - tick, **meta})
        write_json(folder / f"round-{round_id}.json", {
            "elapsed_seconds": time.perf_counter() - start,
            "requests": results,
        })
    print(f"{args.worker} done", flush=True)


def coordinate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpu_ids) != 8:
        raise RuntimeError(f"Expected eight allocated GPUs, got {gpu_ids}")
    workers = {"tp2-a": gpu_ids[:2], "tp2-b": gpu_ids[2:4], "tp4": gpu_ids[4:]}
    processes = {}
    logs = []
    start = time.monotonic()

    def wait_for(paths):
        while not all(path.exists() for path in paths):
            for name, proc in processes.items():
                if proc.poll() is not None and proc.returncode != 0:
                    raise RuntimeError(f"{name} failed ({proc.returncode}); see its log")
                if proc.poll() == 0 and not all(path.exists() for path in paths
                                               if path.parent.name == name):
                    raise RuntimeError(f"{name} exited before writing results")
            if time.monotonic() - start > 1800:
                raise TimeoutError("benchmark exceeded 30 minutes")
            time.sleep(0.1)

    try:
        for name, devices in workers.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices),
                       PYTHONUNBUFFERED="1", VLLM_USE_FLASHINFER_SAMPLER="0")
            log = (args.output / f"{name}.log").open("w")
            logs.append(log)
            processes[name] = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--output",
                 str(args.output), "--worker", name], env=env,
                stdout=log, stderr=subprocess.STDOUT,
            )
        print(f"Benchmark output: {args.output}", flush=True)
        wait_for([args.output / name / "ready.json" for name in workers])
        print("All three collectors warmed up", flush=True)
        report = {"setup": "8 GPUs on one node; 2xTP2 and 1xTP4 simultaneously",
                  "workload": "3 rounds; each scheme processes 8 requests per round; "
                  "32 prefill + 16 forced decode tokens per request; real capture + validation",
                  "workers": {name: json.loads((args.output / name / "ready.json").read_text())
                              for name in workers}, "rounds": []}
        for round_id in range(3):
            (args.output / f"go-{round_id}").touch()
            wait_for([args.output / name / f"round-{round_id}.json" for name in workers])
            timings = {name: json.loads((args.output / name / f"round-{round_id}.json").read_text())
                       ["elapsed_seconds"] for name in workers}
            pair = max(timings["tp2-a"], timings["tp2-b"])
            single = timings["tp4"]
            result = {"round": round_id, "worker_seconds": timings,
                      "two_tp2_seconds": pair, "one_tp4_seconds": single,
                      "two_tp2_requests_per_second": 8 / pair,
                      "one_tp4_requests_per_second": 8 / single,
                      "two_tp2_speedup": single / pair}
            report["rounds"].append(result)
            write_json(args.output / "summary.json", report)
            print(json.dumps(result), flush=True)
        for name, proc in processes.items():
            code = proc.wait(timeout=120)
            if code:
                raise RuntimeError(f"{name} exited with {code}")
        print("benchmark ok", flush=True)
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["tp2-a", "tp2-b", "tp4"])
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        coordinate(args)
