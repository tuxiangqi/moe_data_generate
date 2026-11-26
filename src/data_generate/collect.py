"""Collect MoE expert-prediction records with vLLM.

The per-bucket sample count lives in config/collect.json. `smoke` runs a few
short requests. `run` collects per_bucket records from each length bucket.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
from importlib.metadata import PackageNotFoundError, version
from itertools import chain
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np

from .capture_writer import wait_for_capture

NUM_LAYERS = 40
TOP_K = 8
NUM_EXPERTS = 256
HIDDEN_SIZE = 2048


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    for key in ("model", "source", "output"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty path")
        config[key] = str(Path(value).expanduser().resolve())
    buckets = config["buckets"]
    if (not isinstance(buckets, list) or not buckets
            or any(not isinstance(bucket, str) or not bucket.strip()
                   or bucket == ".." or Path(bucket).name != bucket for bucket in buckets)
            or len(set(buckets)) != len(buckets)):
        raise ValueError("buckets must contain unique directory names")
    config["per_bucket"] = int(config["per_bucket"])
    config["max_new_tokens"] = int(config["max_new_tokens"])
    config["max_context"] = int(config["max_context"])
    config["seed"] = int(config["seed"])
    if config["per_bucket"] < 1:
        raise ValueError("per_bucket must be positive")
    if not 0 < config["max_new_tokens"] < config["max_context"]:
        raise ValueError("max_new_tokens must be positive and less than max_context")
    layers = config["capture_layers"]
    if (not isinstance(layers, list) or not layers
            or any(type(layer) is not int or not 0 <= layer < NUM_LAYERS for layer in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError(f"capture_layers must contain unique layer IDs in 0..{NUM_LAYERS - 1}")
    collection_concurrency(config)
    return config


def require_model_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required model file missing: {path}")
    with path.open("rb") as handle:
        if not handle.read(1):
            raise ValueError(f"empty model file: {path}")


def validate_model(model: Path) -> None:
    if not model.is_dir():
        raise FileNotFoundError(f"model directory missing: {model}")
    model_config = model / "config.json"
    require_model_file(model_config)
    try:
        settings = json.loads(model_config.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"invalid model config: {model_config}: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError(f"model config must be a JSON object: {model_config}")
    tokenizer_files = [
        ("tokenizer.json",), ("tokenizer.model",), ("vocab.json", "merges.txt"),
    ]
    for names in tokenizer_files:
        if all((model / name).is_file() for name in names):
            for name in names:
                require_model_file(model / name)
            break
    else:
        raise FileNotFoundError(f"tokenizer files missing under {model}")

    for name in ("model.safetensors", "pytorch_model.bin"):
        index = model / f"{name}.index.json"
        if index.exists() or index.is_symlink():
            require_model_file(index)
            try:
                manifest = json.loads(index.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise ValueError(f"invalid model weight index: {index}: {exc}") from exc
            weights = manifest.get("weight_map") if isinstance(manifest, dict) else None
            if (not isinstance(weights, dict) or not weights
                    or any(not isinstance(shard, str) or not shard.strip()
                           or Path(shard).is_absolute() or ".." in Path(shard).parts
                           for shard in weights.values())):
                raise ValueError(f"invalid weight_map in model weight index: {index}")
            for shard in sorted(set(weights.values())):
                require_model_file(model / shard)
            return
        checkpoint = model / name
        if checkpoint.exists() or checkpoint.is_symlink():
            require_model_file(checkpoint)
            return
    raise FileNotFoundError(f"model weights missing under {model}")


def validate_inputs(config: dict) -> None:
    validate_model(Path(config["model"]))
    source = Path(config["source"])
    files = [path for bucket in config["buckets"] for path in iter_bucket_files(source, bucket)]
    if not files:
        raise FileNotFoundError(f"no natural.json files in configured buckets under {source}")
    for path in files:
        with path.open("rb"):
            pass
    output = Path(config["output"])
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output is not a directory: {output}")


def collection_concurrency(config: dict) -> int:
    limit = config.get("max_num_seqs", 8)
    if "max_num_seqs" not in config and "batch_size_by_bucket" in config:
        sizes = config["batch_size_by_bucket"]
        if (not isinstance(sizes, dict) or not sizes or set(sizes) != set(config["buckets"])
                or any(type(size) is not int or size < 1 for size in sizes.values())):
            raise ValueError("batch_size_by_bucket must specify a positive limit for each bucket")
        limit = max(sizes.values())
    if type(limit) is not int or limit < 1:
        raise ValueError("max_num_seqs must be a positive integer")
    return limit


def iter_completed_requests(llm, config: dict, records, params):
    """Refill the vLLM queue as requests finish, reserving their full KV budgets."""
    engine = llm.llm_engine
    cache = engine.vllm_config.cache_config
    capacity = cache.kv_cache_size_tokens
    block = cache.effective_attention_block_size
    if not capacity or not block:
        raise RuntimeError("cannot verify request budget: profiled KV capacity is unavailable")
    usable = int(capacity * 0.9)
    limit = collection_concurrency(config)
    print(f"context_budget profiled_kv_capacity={capacity} usable_with_margin={usable} "
          f"block_size={block}", flush=True)

    def with_reservations():
        for record in records:
            length = record["prompt_len"] + config["max_new_tokens"]
            if not prompt_fits(record["prompt_len"], config):
                raise RuntimeError(f"{record['bucket']}/{record['id']}: request must leave one "
                                   f"context slot after generation allowance ({length} tokens)")
            reserve = ((length + block - 1) // block) * block
            if reserve > usable:
                raise RuntimeError(f"{record['bucket']}/{record['id']} needs {reserve} KV tokens "
                                   f"including generation; capacity with 10% margin is {usable}")
            yield record, reserve

    pending = with_reservations()
    next_request = next(pending, None)
    in_flight = {}
    reserved = 0
    completed = []
    while next_request is not None or in_flight or completed:
        while next_request is not None and len(in_flight) < limit:
            record, reserve = next_request
            if reserved + reserve > usable:
                break
            (internal_id,) = llm.enqueue(
                {"prompt_token_ids": record["token_ids"]}, params, use_tqdm=False
            )
            # vLLM 0.32 returns the internal ID from enqueue and the numeric
            # external ID from step. The capture directory uses the internal ID.
            external_id = internal_id.partition("-")[0]
            if external_id in in_flight:
                raise RuntimeError(f"duplicate vLLM request ID {external_id}")
            in_flight[external_id] = record, reserve, internal_id
            reserved += reserve
            next_request = next(pending, None)
        yield from completed
        completed.clear()
        if not in_flight:
            break
        if not engine.has_unfinished_requests():
            raise RuntimeError(f"vLLM lost unfinished requests: {list(in_flight)}")
        for output in engine.step():
            if not output.finished:
                continue
            record, reserve, internal_id = in_flight.pop(output.request_id)
            reserved -= reserve
            if list(output.prompt_token_ids) != record["token_ids"]:
                raise RuntimeError(f"vLLM output prompt mismatch for {record['bucket']}/{record['id']}")
            completed.append((record, output, internal_id))


def prompt_fits(prompt_len: int, config: dict) -> bool:
    # vLLM reserves one slot for the sample discarded after the final forward.
    return prompt_len + config["max_new_tokens"] < config["max_context"]


def iter_bucket_files(source: Path, bucket: str) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"dataset source is not a directory: {source}")
    folder = source / bucket
    if folder.exists() and not folder.is_dir():
        raise NotADirectoryError(folder)
    return sorted(folder.rglob("natural.json"))


def read_at(path: Path, offset: int) -> dict:
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    return json.loads(line)


def record_domain(record: dict) -> str:
    domain = (record.get("meta") or {}).get("domain")
    return (domain.strip() or "unknown") if isinstance(domain, str) else "unknown"


def index_domains(files: list[Path]) -> dict[str, list[tuple[int, int]]]:
    offsets = defaultdict(list)
    for file_index, path in enumerate(files):
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets[record_domain(json.loads(line))].append((file_index, offset))
    return offsets


def encode_prompt(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def count_eligible(config: dict, tokenizer, bucket: str) -> tuple[int, int]:
    files = iter_bucket_files(Path(config["source"]), bucket)
    total = 0
    eligible = 0
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                record = json.loads(line)
                length = record.get("meta", {}).get("length_tokens")
                if isinstance(length, int) and not prompt_fits(length, config):
                    continue
                token_ids = encode_prompt(tokenizer, record["prompt"])
                if prompt_fits(len(token_ids), config):
                    eligible += 1
    return eligible, total


def select_records(config: dict, tokenizer, bucket: str) -> list[dict]:
    files = iter_bucket_files(Path(config["source"]), bucket)
    by_domain = index_domains(files)
    rng = random.Random(f"{config['seed']}:{bucket}")
    domains = sorted(by_domain)
    rng.shuffle(domains)
    for domain in domains:
        rng.shuffle(by_domain[domain])

    chosen = []
    seen = set()
    limit = config["per_bucket"]
    # Each domain gets one draw per round until its eligible records run out.
    remaining_domains = deque(by_domain[domain] for domain in domains)
    while remaining_domains and len(chosen) < limit:
        offsets = remaining_domains.popleft()
        while offsets:
            file_index, offset = offsets.pop()
            record = read_at(files[file_index], offset)
            if record["id"] in seen:
                continue
            token_ids = encode_prompt(tokenizer, record["prompt"])
            if not prompt_fits(len(token_ids), config):
                continue
            seen.add(record["id"])
            chosen.append({
                "id": record["id"],
                "prompt": record["prompt"],
                "token_ids": token_ids,
                "meta": record.get("meta", {}),
                "prompt_len": len(token_ids),
            })
            if offsets:
                remaining_domains.append(offsets)
            break
    return chosen


def capture_row_bytes(capture_layers: list[int]) -> dict[str, int]:
    return {
        "expert_ids.bin": NUM_LAYERS * TOP_K * 4,
        "expert_scores.bin": NUM_LAYERS * NUM_EXPERTS * 4,
        "router_weights_topk.bin": NUM_LAYERS * TOP_K * 4,
        "hidden_states.bin": len(capture_layers) * HIDDEN_SIZE * 2,
    }


def estimate_prefill_bytes(records: list[dict], capture_layers: list[int]) -> int:
    bytes_per_token = sum(capture_row_bytes(capture_layers).values()) + 9
    return sum(item["prompt_len"] for item in records) * bytes_per_token


def _load_array(path: Path, dtype, tail: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype=dtype)
    width = int(np.prod(tail))
    if raw.size % width != 0:
        raise ValueError(f"{path.name} size {raw.size} is not a multiple of {width}")
    rows = raw.size // width
    return raw.reshape((rows, *tail))


def validate_capture(
    folder: Path,
    prompt_ids: list[int],
    output_ids: list[int],
    eog_ids: set[int],
    stop_reason: str,
    max_new_tokens: int | None,
    capture_layers: list[int],
) -> None:
    layers = json.loads((folder / "layers.json").read_text(encoding="utf-8"))
    if layers["capture_layers"] != capture_layers:
        raise ValueError(
            f"{folder.name}: captured layers {layers['capture_layers']}, expected {capture_layers}"
        )
    expert_ids = _load_array(folder / "expert_ids.bin", np.int32, (NUM_LAYERS, TOP_K))
    scores = _load_array(
        folder / "expert_scores.bin", np.float32, (NUM_LAYERS, NUM_EXPERTS)
    )
    weights = _load_array(
        folder / "router_weights_topk.bin", np.float32, (NUM_LAYERS, TOP_K)
    )
    hidden = _load_array(
        folder / "hidden_states.bin", np.uint16, (len(capture_layers), HIDDEN_SIZE)
    )
    rows = expert_ids.shape[0]
    token_ids = np.asarray(prompt_ids + output_ids, dtype=np.int32)
    if not (
        rows == scores.shape[0] == weights.shape[0] == hidden.shape[0]
    ):
        raise ValueError(f"row counts differ in {folder}")
    if rows != token_ids.shape[0]:
        raise ValueError(
            f"{folder.name}: captured {rows} rows, tokens {token_ids.shape[0]}"
        )
    if not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise ValueError(f"non-finite router values in {folder}")
    if expert_ids.min() < 0 or expert_ids.max() >= NUM_EXPERTS:
        raise ValueError(f"expert id out of range in {folder}")
    weight_sum = weights.sum(axis=2)
    if not np.allclose(weight_sum, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError(f"router weights do not sum to 1 in {folder}")
    if any(token in eog_ids for token in output_ids):
        raise ValueError(f"decode tokens include an EOG id in {folder}")
    if list(token_ids[: len(prompt_ids)]) != list(prompt_ids):
        raise ValueError(f"prompt tokens were changed in {folder}")
    if stop_reason == "max_new_tokens" and max_new_tokens is not None:
        if len(output_ids) != max_new_tokens:
            raise ValueError(
                f"expected {max_new_tokens} decode tokens, got {len(output_ids)}"
            )
    phase = np.zeros(rows, dtype=np.uint8)
    phase[len(prompt_ids) :] = 1
    position_ids = np.arange(rows, dtype=np.int32)
    np.save(folder / "token_ids.npy", token_ids)
    np.save(folder / "position_ids.npy", position_ids)
    np.save(folder / "phase.npy", phase)


def sampling_params(config: dict, max_new_tokens: int, **extra):
    from vllm import SamplingParams

    repeat = extra.pop("moe_repeat", None)
    trace = extra.pop("trace", None)
    args = {"moe_eog_ids": list(config["eog_token_ids"])}
    if repeat:
        args["moe_repeat"] = repeat
    return SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
        stop_token_ids=list(config["eog_token_ids"]),
        extra_args=args,
        trace_decode_token_ids=trace,
    )


def make_engine(config: dict, collect_dir: Path):
    try:
        vllm_version = version("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError("Install the upstream vLLM 0.32.0 wheel in the collection environment") from exc
    if vllm_version.partition("+")[0] != "0.32.0":
        raise RuntimeError(f"MoE collection supports upstream vLLM 0.32.0, found {vllm_version}")
    if not config.get("enforce_eager", True):
        raise ValueError("The MoE collection plugin requires enforce_eager=true")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["MOE_COLLECT_DIR"] = str(collect_dir)
    os.environ["MOE_CAPTURE_LAYERS"] = json.dumps(config["capture_layers"])
    os.environ.setdefault("MOE_REPEAT_PERIOD_MAX", "512")
    os.environ.setdefault("MOE_REPEAT_MIN_TOKENS", "96")
    os.environ.setdefault("MOE_REPEAT_DIVERSITY_WINDOW", "2048")
    os.environ.setdefault("MOE_REPEAT_DIVERSITY_MAX", "64")
    from vllm import LLM, envs

    if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
        raise RuntimeError("Buffered MoE capture requires vLLM engine multiprocessing")

    return LLM(
        model=config["model"],
        tensor_parallel_size=int(config["tensor_parallel_size"]),
        max_model_len=int(config["max_context"]),
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        max_num_seqs=int(config.get("max_num_seqs", 1)),
        max_num_batched_tokens=int(config["max_num_batched_tokens"]),
        trust_remote_code=True,
        enforce_eager=True,
        worker_cls="data_generate.vllm_worker.CaptureWorker",
        scheduler_cls="data_generate.vllm_scheduler.CaptureScheduler",
        enable_trace_replay=True,
    )


def locate_capture_dir(collect_dir: Path, request_id: str) -> Path:
    direct = collect_dir / request_id
    if (direct / "stop.json").is_file():
        return direct
    pending = [
        path
        for path in collect_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "stop.json").is_file()
        and not (path / "meta.json").is_file()
    ]
    prefixed = [path for path in pending if path.name.startswith(f"{request_id}-")]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(pending) == 1:
        return pending[0]
    raise FileNotFoundError(
        f"no capture directory for request {request_id} in {collect_dir}"
    )


def finish_request(config: dict, collect_dir: Path, output, prompt_ids: list[int],
                   max_new_tokens: int | None, *, capture_id: str | None = None):
    folder = collect_dir / capture_id if capture_id is not None else locate_capture_dir(
        collect_dir, output.request_id
    )
    wait_for_capture(folder)
    stop = json.loads((folder / "stop.json").read_text(encoding="utf-8"))
    generated = list(output.outputs[0].token_ids)
    validate_capture(
        folder,
        prompt_ids,
        generated,
        set(config["eog_token_ids"]),
        stop["stop_reason"],
        max_new_tokens if stop["stop_reason"] == "max_new_tokens" else None,
        config["capture_layers"],
    )
    meta = {
        "request_id": folder.name,
        "stop_reason": stop["stop_reason"],
        "repeat_kind": stop.get("repeat_kind"),
        "repeat_period": stop.get("repeat_period"),
        "repeat_unique": stop.get("repeat_unique"),
        "prompt_len": len(prompt_ids),
        "generated_len": len(generated),
        "capture_layers": config["capture_layers"],
        "model": config["model"],
    }
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def generate_one(llm, config: dict, collect_dir: Path, token_ids: list[int], params, max_new_tokens: int | None):
    return generate_batch(llm, config, collect_dir, [token_ids], params, [max_new_tokens])[0]


def generate_batch(llm, config: dict, collect_dir: Path,
                   prompts: list[list[int]], params, limits: list[int | None]):
    if len(prompts) != len(limits):
        raise ValueError("each prompt requires its own generation limit")
    outputs = llm.generate(
        [{"prompt_token_ids": token_ids} for token_ids in prompts],
        params,
        use_tqdm=False,
    )
    if len(outputs) != len(prompts):
        raise RuntimeError("generation returned a different number of requests")
    results = []
    for output, prompt, limit in zip(outputs, prompts, limits):
        if list(output.prompt_token_ids) != prompt:
            raise RuntimeError("generation output does not match the submitted prompt")
        results.append(finish_request(config, collect_dir, output, prompt, limit))
    return results


def first_short_prompt(config: dict, tokenizer) -> list[int]:
    source = Path(config["source"])
    for path in iter_bucket_files(source, "1-8k"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                token_ids = encode_prompt(tokenizer, record["prompt"])
                if len(token_ids) >= 8:
                    return token_ids[:32]
    raise ValueError(f"no smoke prompt with at least 8 tokens under {source / '1-8k'}")


def run_smoke(config: dict) -> None:
    from transformers import AutoTokenizer

    config = dict(config)
    # Short prompts only. Keep the context small so the smoke engine does not
    # reserve the full 256k KV cache.
    config["max_context"] = 4096
    config["max_num_seqs"] = 1
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    prompt = first_short_prompt(config, tokenizer)
    smoke_root = Path(config["output"]) / "_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke_dir = Path(tempfile.mkdtemp(prefix="run-", dir=smoke_root))
    print(f"smoke output: {smoke_dir}", flush=True)
    llm = make_engine(config, smoke_dir)
    eog = int(config["eog_token_ids"][0])
    filler = 100
    while filler in prompt or filler in config["eog_token_ids"]:
        filler += 1
    repeated = 200
    while repeated in prompt or repeated in config["eog_token_ids"] or repeated == filler:
        repeated += 1

    print("smoke max_new_tokens")
    meta = generate_one(
        llm,
        config,
        smoke_dir,
        prompt,
        sampling_params(config, 4, trace=[filler, filler + 1, filler + 2, filler + 3, filler + 4]),
        4,
    )
    if meta["stop_reason"] != "max_new_tokens" or meta["generated_len"] != 4:
        raise RuntimeError(f"max_new_tokens smoke failed: {meta}")

    print("smoke eog")
    meta = generate_one(
        llm,
        config,
        smoke_dir,
        prompt,
        sampling_params(config, 8, trace=[eog]),
        None,
    )
    if meta["stop_reason"] != "eog" or meta["generated_len"] != 0:
        raise RuntimeError(f"eog smoke failed: {meta}")

    print("smoke repetition")
    meta = generate_one(
        llm,
        config,
        smoke_dir,
        prompt,
        sampling_params(
            config,
            16,
            trace=[repeated] * 6,
            moe_repeat={
                "period_max": 4,
                "min_tokens": 4,
                "diversity_window": 0,
                "diversity_max": 0,
            },
        ),
        None,
    )
    if meta["stop_reason"] != "repetition" or meta["generated_len"] < 1:
        raise RuntimeError(f"repetition smoke failed: {meta}")

    print("smoke prefill/decode")
    split = prompt[:16]
    held = split[-1]
    if held in config["eog_token_ids"]:
        raise RuntimeError("smoke split token is an EOG id")
    prefill = generate_one(
        llm,
        config,
        smoke_dir,
        split,
        sampling_params(config, 1, trace=[filler, filler + 1]),
        1,
    )
    decode = generate_one(
        llm,
        config,
        smoke_dir,
        split[:-1],
        sampling_params(config, 1, trace=[held, filler]),
        1,
    )
    prefill_ids = _load_array(
        smoke_dir / prefill["request_id"] / "expert_ids.bin",
        np.int32,
        (NUM_LAYERS, TOP_K),
    )
    decode_ids = _load_array(
        smoke_dir / decode["request_id"] / "expert_ids.bin",
        np.int32,
        (NUM_LAYERS, TOP_K),
    )
    # BF16 prefill and single-token decode use different matrix kernels.
    # Near-tied logits can change top-k membership, even for aligned tokens.
    # Each request's IDs/logits are already checked by validate_capture.
    row = len(split) - 1
    report = {"different_expert_positions": int(np.count_nonzero(
        prefill_ids[row] != decode_ids[row]
    )), "comparisons": {}}
    for name, dtype, tail in (
        ("expert_scores", np.float32, (NUM_LAYERS, NUM_EXPERTS)),
        ("hidden_states", np.uint16, (len(config["capture_layers"]), HIDDEN_SIZE)),
    ):
        x = _load_array(smoke_dir / prefill["request_id"] / f"{name}.bin", dtype, tail)[row]
        y = _load_array(smoke_dir / decode["request_id"] / f"{name}.bin", dtype, tail)[row]
        if dtype == np.uint16:
            x = (x.astype(np.uint32) << 16).view(np.float32)
            y = (y.astype(np.uint32) << 16).view(np.float32)
        norm_x = np.linalg.norm(x, axis=1)
        norm_y = np.linalg.norm(y, axis=1)
        cosine = np.sum(x * y, axis=1) / np.maximum(norm_x * norm_y, 1e-20)
        relative_error = np.linalg.norm(x - y, axis=1) / np.maximum(norm_x, 1e-20)
        report["comparisons"][name] = {
            "cosine_per_layer": cosine.tolist(),
            "relative_error_per_layer": relative_error.tolist(),
        }
        # Smoke sanity bounds, not a claim of bitwise inference equivalence.
        if (not np.isfinite(cosine).all() or not np.isfinite(relative_error).all()
                or np.min(cosine) < 0.99 or np.max(relative_error) > 0.1):
            raise RuntimeError(f"prefill/decode numerical mismatch: {report}")
    (smoke_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    print(f"prefill/decode comparison: {report}", flush=True)
    print("smoke ok")


def collection_settings(config: dict) -> dict:
    return {key: config[key] for key in (
        "model", "max_new_tokens", "max_context", "seed", "capture_layers", "eog_token_ids"
    )}


def prepare_plan(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _prepare_plan(config, path)


def _prepare_plan(config: dict, path: Path) -> None:
    selection_settings = {key: config[key] for key in ("source", "per_bucket", "buckets")}
    selection_settings["sampling"] = "balanced_domains"
    settings = {"collection_settings": collection_settings(config),
                "selection_settings": selection_settings}
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    counts = dict.fromkeys(config["buckets"], 0)
    if path.exists():
        if not summary_path.is_file():
            raise RuntimeError(f"plan has no domain-balanced selection summary: {path}; use a new plan path")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(summary.get(key) != value for key, value in settings.items()):
            raise RuntimeError(f"existing plan has different settings: {path}")
        seen = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if any(record.get(key) != value for key, value in settings.items()):
                    raise RuntimeError(f"existing plan has different settings: {path}")
                key = (record["bucket"], record["id"])
                if key in seen:
                    raise RuntimeError(f"duplicate request in plan: {key}")
                seen.add(key)
                counts[record["bucket"]] += 1
        if counts != summary.get("counts") or any(
            count > config["per_bucket"] for count in counts.values()
        ):
            raise RuntimeError(f"incomplete plan: {counts}")
        print(f"reusing verified plan: {path}, counts={counts}", flush=True)
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    domains = {}
    with tempfile.TemporaryDirectory(prefix=".selection-", dir=path.parent) as directory:
        temporary = Path(directory) / "selection.jsonl"
        with temporary.open("w", encoding="utf-8") as handle:
            for bucket in config["buckets"]:
                records = select_records(config, tokenizer, bucket)
                counts[bucket] = len(records)
                domains[bucket] = dict(sorted(Counter(map(record_domain, records)).items()))
                for record in records:
                    record.pop("prompt")
                    record.update(settings, bucket=bucket)
                    handle.write(json.dumps(record) + "\n")
                print(f"prepared {bucket}: {len(records)}/{config['per_bucket']} requests "
                      f"domains={domains[bucket]}", flush=True)
        summary = Path(directory) / "summary.json"
        summary.write_text(json.dumps({**settings, "counts": counts, "domains": domains}, indent=2)
                           + "\n", encoding="utf-8")
        summary.replace(summary_path)
        temporary.replace(path)


def iter_plan(path: Path, shard_index: int, num_shards: int):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % num_shards == shard_index:
                yield json.loads(line)


def completed_record(dest: Path, record: dict, config: dict) -> bool:
    if not dest.exists():
        return False
    meta = json.loads((dest / "meta.json").read_text())
    if (meta.get("corpus_id") != record["id"]
            or meta.get("collection_settings") != collection_settings(config)):
        raise RuntimeError(f"existing result has incompatible metadata: {dest}")
    layers = json.loads((dest / "layers.json").read_text(encoding="utf-8"))
    if (layers.get("capture_layers") != config["capture_layers"]
            or meta.get("capture_layers") != config["capture_layers"]):
        raise RuntimeError(f"existing result has incompatible capture layers: {dest}")
    rows = meta["prompt_len"] + meta["generated_len"]
    for name, width in capture_row_bytes(config["capture_layers"]).items():
        if (dest / name).stat().st_size != rows * width:
            raise RuntimeError(f"incomplete existing result: {dest / name}")
    for name in ("token_ids.npy", "phase.npy", "position_ids.npy"):
        if np.load(dest / name, mmap_mode="r").shape != (rows,):
            raise RuntimeError(f"incomplete existing result: {dest / name}")
    return True


def run_collection(config: dict, plan: Path | None = None,
                   shard_index: int = 0, num_shards: int = 1) -> None:
    if not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    config = dict(config)
    config["max_num_seqs"] = collection_concurrency(config)
    print(f"shard={shard_index}/{num_shards} TP={config['tensor_parallel_size']} "
          f"max_num_seqs={config['max_num_seqs']}", flush=True)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if plan is None:
        if num_shards != 1:
            raise ValueError("parallel collection requires a prepared --plan")
        plan = output / "selection.jsonl"
        prepare_plan(config, plan)

    def pending_records():
        for record in iter_plan(plan, shard_index, num_shards):
            if record["collection_settings"] != collection_settings(config):
                raise RuntimeError("plan does not match collection settings")
            if completed_record(output / record["bucket"] / record["id"], record, config):
                print(f"skip completed {record['bucket']}/{record['id']}", flush=True)
            else:
                yield record

    records = pending_records()
    first = next(records, None)
    if first is None:
        return
    active_root = output / "_active"
    active_root.mkdir(exist_ok=True)
    active = Path(tempfile.mkdtemp(prefix=f"shard-{shard_index}-", dir=active_root))
    llm = make_engine(config, active)
    run_id = os.environ.get("SLURM_JOB_ID", active.name)

    def publish(record, result, internal_id):
        meta = finish_request(config, active, result, record["token_ids"],
                              config["max_new_tokens"], capture_id=internal_id)
        bucket = record["bucket"]
        dest = output / bucket / record["id"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = active / internal_id
        meta.update(corpus_id=record["id"], source_meta=record["meta"],
                    collection_settings=collection_settings(config))
        (src / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        src.rename(dest)
        manifest = output / bucket / "manifests" / f"{run_id}-shard-{shard_index}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"corpus_id": record["id"], "stop_reason": meta["stop_reason"]}) + "\n")
        print(bucket, record["id"], meta["stop_reason"], meta["generated_len"], flush=True)

    params = sampling_params(config, config["max_new_tokens"])
    results = iter_completed_requests(llm, config, chain((first,), records), params)
    writes = deque()
    with ThreadPoolExecutor(max_workers=1) as writer:
        for result in results:
            while writes and writes[0].done():
                writes.popleft().result()
            if len(writes) >= config["max_num_seqs"]:
                writes.popleft().result()
            writes.append(writer.submit(publish, *result))
        for write in writes:
            write.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "run", "preflight", "prepare", "check"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--max-num-seqs", "--batch-size", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "collect.json",
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        validate_inputs(config)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"collect: {args.config}: {exc}\n")
    if args.command == "check":
        print(config["output"])
        return
    if args.tp_size is not None:
        config["tensor_parallel_size"] = args.tp_size
    if args.max_num_seqs is not None:
        if args.max_num_seqs < 1:
            parser.error("--max-num-seqs must be positive")
        config["max_num_seqs"] = args.max_num_seqs
    if args.command == "prepare":
        if args.plan is None:
            parser.error("prepare requires --plan")
        prepare_plan(config, args.plan)
        return
    if args.command == "smoke":
        run_smoke(config)
        return
    if args.command == "preflight":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
        total_bytes = 0
        for bucket in config["buckets"]:
            if bucket == ">128k":
                eligible, total = count_eligible(config, tokenizer, bucket)
                print(f">128k eligible {eligible} / {total}")
            records = select_records(config, tokenizer, bucket)
            nbytes = estimate_prefill_bytes(records, config["capture_layers"])
            total_bytes += nbytes
            print(f"{bucket}: {len(records)} prompts, prefill {nbytes} bytes")
        print(f"prefill total {total_bytes} bytes")
        return
    run_collection(config, args.plan, args.shard_index, args.num_shards)


if __name__ == "__main__":
    main()
