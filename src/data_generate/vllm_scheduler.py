# SPDX-License-Identifier: Apache-2.0
"""vLLM scheduler that forwards the last retained token before stopping."""

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from .repeat_guard import guard_for_request, write_stop


class CaptureScheduler(Scheduler):
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        if is_stale or request.sampling_params is None:
            return super()._update_request_with_output(request, new_token_ids, is_stale)
        return self._update_moe_capture(request, new_token_ids)

    def _update_moe_capture(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """Keep every sampled decode token except EOG, then forward the last one."""
        if request.num_output_placeholders > 0:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        if getattr(request, "moe_discard_next", False):
            request.moe_discard_next = False
            request.status = request.moe_pending_status
            request.stop_reason = request.moe_pending_stop
            write_stop(request)
            return [], True

        if not new_token_ids:
            return [], False

        sampling_params = request.sampling_params
        assert sampling_params is not None
        eog_ids = set(sampling_params.all_stop_token_ids)
        if sampling_params.eos_token_id is not None:
            eog_ids.add(sampling_params.eos_token_id)
        # Trace replay clears stop ids so an EOG token can be replayed.
        # Collection still treats these ids as end-of-generation.
        for token_id in (sampling_params.extra_args or {}).get("moe_eog_ids") or []:
            eog_ids.add(int(token_id))
        guard = guard_for_request(request)
        kept: list[int] = []
        for token in new_token_ids:
            if token in eog_ids:
                request.status = RequestStatus.FINISHED_STOPPED
                request.stop_reason = "eog"
                request.moe_repeat_kind = None
                write_stop(request)
                return kept, True
            request.append_output_token_ids(token)
            kept.append(token)
            verdict = guard.observe(request.all_token_ids, request.num_output_tokens)
            hit_limit = (
                request.num_tokens >= self.max_model_len - self.num_sampled_tokens_per_step
                or request.num_output_tokens >= request.max_tokens
            )
            if verdict == 0 and not hit_limit:
                continue
            request.moe_discard_next = True
            if verdict > 0:
                request.moe_pending_status = RequestStatus.FINISHED_REPETITION
                request.moe_pending_stop = "repetition"
                request.moe_repeat_kind = "period"
                request.moe_repeat_period = verdict
            elif verdict < 0:
                request.moe_pending_status = RequestStatus.FINISHED_REPETITION
                request.moe_pending_stop = "repetition"
                request.moe_repeat_kind = "diversity"
                request.moe_repeat_unique = guard.last_unique
            else:
                request.moe_pending_status = RequestStatus.FINISHED_LENGTH_CAPPED
                request.moe_pending_stop = "max_new_tokens"
                request.moe_repeat_kind = None
            return kept, False
        return kept, False
