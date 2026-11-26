"""Validate a few full prompts per bucket with production collection settings."""

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_generate.collect import (
    collection_settings, encode_prompt, iter_bucket_files,
    load_config, prompt_fits, run_collection,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=2)
    args = parser.parse_args()
    if args.per_bucket < 1:
        parser.error("--per-bucket must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "config/collect.json")
    config["output"] = str(args.output)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    count = args.per_bucket
    plan = args.output / "selection.jsonl"
    report = {"status": "running", "buckets": {}, "config": config}
    report_path = args.output / "validation.json"

    def save():
        tmp = report_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(report_path)

    save()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
        with plan.open("w", encoding="utf-8") as handle:
            for bucket in config["buckets"]:
                selected, seen = [], set()
                for path in iter_bucket_files(Path(config["source"]), bucket):
                    with path.open(encoding="utf-8") as source:
                        for line in source:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            if record["id"] in seen:
                                continue
                            tokens = encode_prompt(tokenizer, record["prompt"])
                            if not prompt_fits(len(tokens), config):
                                continue
                            seen.add(record["id"])
                            selected.append({"id": record["id"], "bucket": bucket,
                                             "token_ids": tokens, "prompt_len": len(tokens),
                                             "meta": record.get("meta", {}),
                                             "collection_settings": collection_settings(config)})
                            if len(selected) == count:
                                break
                    if len(selected) == count:
                        break
                if len(selected) != count:
                    raise RuntimeError(f"{bucket}: cannot find {count} eligible full prompts")
                report["buckets"][bucket] = {
                    "prompt_lengths": [r["prompt_len"] for r in selected],
                }
                print(f"selected {bucket}: {report['buckets'][bucket]}", flush=True)
                for record in selected:
                    handle.write(json.dumps(record) + "\n")
                save()
        run_collection(config, plan)
        for bucket in config["buckets"]:
            metas = [json.loads(p.read_text()) for p in (args.output / bucket).glob("*/meta.json")]
            assert len(metas) == count, (bucket, len(metas))
            report["buckets"][bucket]["results"] = metas
        report["status"] = "passed"
        save()
        print("bucket validation passed", flush=True)
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
