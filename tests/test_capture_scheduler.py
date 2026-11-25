"""Check stop metadata and the final-forward contract with upstream requests."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm import SamplingParams
from vllm.v1.request import Request, RequestStatus

from data_generate.vllm_scheduler import CaptureScheduler


class CaptureSchedulerTest(unittest.TestCase):
    def test_retains_last_token_before_the_final_forward(self):
        scheduler = CaptureScheduler.__new__(CaptureScheduler)
        scheduler.max_model_len = 8
        scheduler.num_sampled_tokens_per_step = 1
        for max_tokens, prompt_length in ((1, 1), (4, 6)):
            with self.subTest(max_tokens=max_tokens, prompt_length=prompt_length):
                request = Request(
                    "request",
                    [1] * prompt_length,
                    SamplingParams(max_tokens=max_tokens),
                    None,
                )
                with tempfile.TemporaryDirectory() as directory, patch.dict(
                    os.environ, {"MOE_COLLECT_DIR": directory}
                ):
                    stop_file = Path(directory) / request.request_id / "stop.json"
                    self.assertEqual(
                        scheduler._update_request_with_output(request, [2]), ([2], False)
                    )
                    self.assertFalse(stop_file.exists())
                    self.assertEqual(
                        scheduler._update_request_with_output(request, [3]), ([], True)
                    )
                    self.assertEqual(list(request.output_token_ids), [2])
                    self.assertEqual(request.status, RequestStatus.FINISHED_LENGTH_CAPPED)
                    self.assertEqual(
                        json.loads(stop_file.read_text())["stop_reason"], "max_new_tokens"
                    )

    def test_replayed_eog_is_excluded_without_an_extra_forward(self):
        scheduler = CaptureScheduler.__new__(CaptureScheduler)
        request = Request(
            "request", [1], SamplingParams(max_tokens=4, extra_args={"moe_eog_ids": [2]}), None
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MOE_COLLECT_DIR": directory}
        ):
            self.assertEqual(scheduler._update_request_with_output(request, [2]), ([], True))
            self.assertFalse(request.output_token_ids)
            self.assertEqual(request.status, RequestStatus.FINISHED_STOPPED)
            stop_file = Path(directory) / request.request_id / "stop.json"
            self.assertEqual(json.loads(stop_file.read_text())["stop_reason"], "eog")


if __name__ == "__main__":
    unittest.main()
