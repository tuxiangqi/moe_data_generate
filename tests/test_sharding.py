"""CPU checks for deterministic allocation and resuming across shard counts."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from data_generate.collect import (
    check_context_budget, collection_settings, completed_record, generate_batch,
    iter_plan, prepare_plan, record_domain, select_records, validate_inputs,
)


class ShardingTests(unittest.TestCase):
    def test_input_check_rejects_missing_shard_and_source_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "tokenizer.json").write_text("{}")
            shard = model / "model-00001-of-00001.safetensors"
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"layer.weight": shard.name},
            }))
            source = root / "source"
            output = root / "new" / "output"
            config = dict(model=str(model), source=str(source), output=str(output),
                          buckets=["filled", "empty"])
            with self.assertRaisesRegex(FileNotFoundError, shard.name):
                validate_inputs(config)
            shard.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(FileNotFoundError, "dataset source"):
                validate_inputs(config)
            (source / "filled").mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "no natural.json files"):
                validate_inputs(config)
            (source / "filled" / "natural.json").touch()
            validate_inputs(config)
            self.assertFalse(output.parent.exists())
            output.parent.mkdir()
            output.touch()
            with self.assertRaisesRegex(NotADirectoryError, "output is not a directory"):
                validate_inputs(config)

    def test_kv_budget_reserves_decode_tokens_and_block_rounding(self):
        cache = SimpleNamespace(kv_cache_size_tokens=1000, effective_attention_block_size=16)
        llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(cache_config=cache)))
        batch = [{"bucket": "test", "prompt_len": 100}] * 2
        # Prefill alone fits; reserving generation makes this batch unsafe.
        with self.assertRaises(RuntimeError):
            check_context_budget(llm, {"max_new_tokens": 400, "max_context": 600}, batch)
        check_context_budget(llm, {"max_new_tokens": 200, "max_context": 600}, batch)

    def test_batch_preserves_request_order_and_limits(self):
        prompts = [[1, 2], [3, 4, 5]]
        outputs = [SimpleNamespace(prompt_token_ids=p) for p in prompts]
        llm = Mock()
        llm.generate.return_value = outputs
        with patch("data_generate.collect.finish_request", side_effect=["first", "second"]) as finish:
            self.assertEqual(generate_batch(llm, {}, Path("unused"), prompts, "params", [2, 3]),
                             ["first", "second"])
            self.assertEqual([call.args[-1] for call in finish.call_args_list], [2, 3])
        llm.generate.return_value = list(reversed(outputs))
        with patch("data_generate.collect.finish_request") as finish:
            with self.assertRaises(RuntimeError):
                generate_batch(llm, {}, Path("unused"), prompts, "params", [2, 3])
            finish.assert_not_called()

    def test_every_request_assigned_once_for_different_node_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.jsonl"
            plan.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(2500)))
            for shards in (1, 2, 4, 8, 12, 20):
                groups = [list(iter_plan(plan, rank, shards)) for rank in range(shards)]
                assigned = [item["id"] for group in groups for item in group]
                self.assertEqual(sorted(assigned), list(range(2500)))
                self.assertLessEqual(max(map(len, groups)) - min(map(len, groups)), 1)

    def test_resume_uses_final_directory_and_rejects_truncated_results(self):
        config = dict(model="model", max_new_tokens=16, max_context=4096, seed=0,
                      capture_layers=[39, 4], eog_token_ids=[9])
        record = {"id": "sample"}
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "sample"
            self.assertFalse(completed_record(dest, record, config))
            dest.mkdir()
            meta = dict(corpus_id="sample", collection_settings=collection_settings(config),
                        capture_layers=config["capture_layers"], prompt_len=1, generated_len=1)
            (dest / "meta.json").write_text(json.dumps(meta))
            layers = dest / "layers.json"
            layers.write_text(json.dumps({"capture_layers": [39, 4]}))
            for name, width in {"expert_ids.bin": 40 * 8 * 4,
                                "expert_scores.bin": 40 * 256 * 4,
                                "router_weights_topk.bin": 40 * 8 * 4,
                                "hidden_states.bin": 2 * 2048 * 2}.items():
                (dest / name).write_bytes(bytes(2 * width))
            for name in ("token_ids.npy", "phase.npy", "position_ids.npy"):
                np.save(dest / name, np.zeros(2, dtype=np.int32))
            # No manifest entry is needed: rename commits the complete request.
            self.assertTrue(completed_record(dest, record, config))
            with self.assertRaises(RuntimeError):
                completed_record(dest, record, dict(config, max_new_tokens=17))
            layers.write_text(json.dumps({"capture_layers": [4, 39]}))
            with self.assertRaisesRegex(RuntimeError, "incompatible capture layers"):
                completed_record(dest, record, config)
            layers.write_text(json.dumps({"capture_layers": [39, 4]}))
            (dest / "hidden_states.bin").write_bytes(b"")
            with self.assertRaises(RuntimeError):
                completed_record(dest, record, config)

    def test_domain_quotas_redistribute_shortages_and_preserve_short_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bucket" / "natural.json"
            source.parent.mkdir()
            rows = [
                {"id": f"{domain}-{index}", "prompt": "valid", "meta": {"domain": domain}}
                for domain, count in (("small", 1), ("medium", 7), ("large", 9))
                for index in range(count)
            ]
            rows += [rows[0], {"id": "too-long", "prompt": "x" * 64, "meta": {"domain": "small"}}]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = dict(source=str(root), per_bucket=9, buckets=["bucket", "empty"], seed=42,
                          model="model", max_new_tokens=4, max_context=16,
                          capture_layers=[4], eog_token_ids=[9])
            tokenizer = SimpleNamespace(encode=lambda text, **_: list(range(len(text))))
            records = select_records(config, tokenizer, "bucket")
            self.assertEqual(Counter(map(record_domain, records)),
                             {"small": 1, "medium": 4, "large": 4})
            self.assertEqual(records, select_records(config, tokenizer, "bucket"))
            plan = root / "selection.jsonl"
            config["per_bucket"] = 20
            with patch.dict("sys.modules", {"transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            )}):
                prepare_plan(config, plan)
            self.assertEqual(len(list(iter_plan(plan, 0, 1))), 17)
            prepare_plan(config, plan)
            plan.write_text("")
            with self.assertRaisesRegex(RuntimeError, "incomplete plan"):
                prepare_plan(config, plan)


if __name__ == "__main__":
    unittest.main()
