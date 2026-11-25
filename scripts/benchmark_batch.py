"""Measure TP2 capture throughput at request batch sizes 1, 2, 4 and 8."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_tp import write_json


def worker(args):
    from data_generate.collect import (
        encode_prompt, generate_batch, iter_bucket_files, load_config,
        make_engine, sampling_params,
    )
    folder = args.output / f"batch-{args.batch}"
    folder.mkdir()
    config = load_config(ROOT / "config/collect.json")
    config.update(max_num_seqs=args.batch, tensor_parallel_size=2, max_context=4096)
    start = time.perf_counter()
    llm = make_engine(config, folder / "captures")
    source = iter_bucket_files(Path(config["source"]), "1-8k")[0]
    with source.open() as handle:
        tokens = encode_prompt(llm.get_tokenizer(), json.loads(handle.readline())["prompt"])
    prompts = [tokens[:[32, 64, 96, 128][i % 4]] for i in range(16)]

    def run(indices):
        params = [sampling_params(config, 32, trace=list(range(100 + i, 133 + i)))
                  for i in indices]
        metas = generate_batch(llm, config, folder / "captures",
                               [prompts[i] for i in indices], params, [32] * len(indices))
        for i, meta in zip(indices, metas):
            assert meta["prompt_len"] == len(prompts[i]) and meta["generated_len"] == 32
        return metas

    run(list(range(args.batch)))
    # Mixed lengths and early EOG check that concurrent requests cannot mix rows.
    check = generate_batch(
        llm, config, folder / "captures", prompts[:3],
        [sampling_params(config, 8, trace=[248046]),
         sampling_params(config, 3, trace=[110, 111, 112, 113]),
         sampling_params(config, 5, trace=[120, 121, 122, 123, 124, 125])],
        [None, 3, 5],
    )
    assert [m["generated_len"] for m in check] == [0, 3, 5]
    write_json(folder / "ready.json", {"startup_seconds": time.perf_counter() - start,
                                      "mixed_stop_validation": "passed"})
    for round_id in range(3):
        while not (args.output / f"go-{round_id}").exists():
            time.sleep(0.02)
        start = time.perf_counter()
        metas = []
        for offset in range(0, 16, args.batch):
            metas.extend(run(list(range(offset, offset + args.batch))))
        elapsed = time.perf_counter() - start
        write_json(folder / f"round-{round_id}.json", {
            "batch_size": args.batch, "seconds": elapsed,
            "requests_per_second": 16 / elapsed,
            "captured_tokens_per_second": sum(len(p) + 32 for p in prompts) / elapsed,
            "requests": metas,
        })


def coordinate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(gpus) == 8
    procs, logs = {}, []
    start = time.monotonic()

    def wait(paths):
        while not all(p.exists() for p in paths):
            if any(p.poll() not in (None, 0) for p in procs.values()):
                raise RuntimeError("batch benchmark failed; inspect worker logs")
            if time.monotonic() - start > 1800:
                raise TimeoutError("batch benchmark timeout")
            time.sleep(0.1)

    try:
        for index, batch in enumerate((1, 2, 4, 8)):
            log = (args.output / f"batch-{batch}.log").open("w")
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus[2*index:2*index+2]),
                       PYTHONUNBUFFERED="1", VLLM_USE_FLASHINFER_SAMPLER="0")
            procs[batch] = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                 "--batch", str(batch)], env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        wait([args.output / f"batch-{b}" / "ready.json" for b in procs])
        print("All four TP2 collectors ready; mixed-stop validation passed", flush=True)
        report = {"workload": "TP2; 16 requests/round; prefill lengths 32/64/96/128; "
                  "32 forced decode tokens; capture, disk writes and validation included",
                  "rounds": []}
        for round_id in range(3):
            (args.output / f"go-{round_id}").touch()
            paths = [args.output / f"batch-{b}" / f"round-{round_id}.json" for b in procs]
            wait(paths)
            result = {str(b): json.loads(path.read_text()) for b, path in zip(procs, paths)}
            report["rounds"].append(result)
            write_json(args.output / "summary.json", report)
            print(json.dumps({b: v["seconds"] for b, v in result.items()}), flush=True)
        for proc in procs.values():
            assert proc.wait(timeout=120) == 0
        print("batch benchmark ok", flush=True)
    finally:
        for proc in procs.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 4, 8))
    args = parser.parse_args()
    if args.batch:
        worker(args)
    else:
        coordinate(args)
