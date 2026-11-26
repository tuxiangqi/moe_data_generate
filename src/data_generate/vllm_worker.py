"""vLLM v0.32 GPU worker integration for MoE collection."""

from functools import partial

import torch
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

from .vllm_capture import ExpertPredictionCapturer, capture_layers_from_env


def _capture_routing(
    select_experts, capture, hidden_states, router_logits, *args, **kwargs
):
    weights, ids = select_experts(hidden_states, router_logits, *args, **kwargs)
    capture(hidden_states, router_logits, ids, weights)
    return weights, ids


def _capture_monolithic(forward, select_experts, x, router_logits, input_ids=None):
    select_experts(x, router_logits, topk_indices_dtype=torch.int32, input_ids=input_ids)
    return forward(x=x, router_logits=router_logits, input_ids=input_ids)


class CaptureModelRunner(GPUModelRunner):
    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        config = self.vllm_config.model_config
        num_layers = int(config.get_total_num_hidden_layers())
        runners = [
            module
            for module in self.model.modules()
            if isinstance(module, MoERunner) and isinstance(module.router, FusedMoERouter)
        ]
        layer_ids = sorted(int(module.layer_id) for module in runners)
        if layer_ids != list(range(num_layers)):
            raise RuntimeError(
                f"Expected MoE layers 0..{num_layers - 1}, found {layer_ids}"
            )
        self._capturer = ExpertPredictionCapturer(
            max_num_batched_tokens=self.vllm_config.scheduler_config.max_num_batched_tokens,
            num_layers=num_layers,
            num_experts=int(config.get_num_experts()),
            top_k=int(config.get_num_experts_per_tok()),
            hidden_size=int(config.get_hidden_size()),
            capture_layers=capture_layers_from_env(num_layers),
        )
        for module in runners:
            router = module.router
            router.select_experts = partial(
                _capture_routing,
                router.select_experts,
                partial(self._capturer.capture, int(module.layer_id)),
            )
            experts = module.routed_experts
            if experts.quant_method.is_monolithic:
                experts.forward_monolithic = partial(
                    _capture_monolithic, experts.forward_monolithic, router.select_experts
                )

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kwargs
    ):
        self._capturer.rows_by_layer.clear()
        output = super().execute_model(
            scheduler_output, intermediate_tensors, dummy_run=dummy_run, **kwargs
        )
        if not dummy_run:
            self._capturer.finish_requests(scheduler_output.finished_req_ids)
            if scheduler_output.total_num_scheduled_tokens:
                self._capturer.flush(self.execute_model_state.input_batch)
        return output

    def shutdown(self) -> None:
        try:
            if capturer := getattr(self, "_capturer", None):
                capturer.close()
        finally:
            super().shutdown()


class CaptureWorker(Worker):
    def _make_model_runner(self):
        config = self.vllm_config
        parallel = config.parallel_config
        if (
            not self.use_v2_model_runner
            or not config.model_config.enforce_eager
            or config.is_mm_encoder_only
            or config.speculative_config
            or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.enable_expert_parallel
            or parallel.enable_eplb
        ):
            raise RuntimeError("MoE collection requires eager mode, GPU runner V2, and TP only")
        if hasattr(FusedMoERouter, "set_prediction_fn"):
            raise RuntimeError("Use the upstream vLLM wheel with the data_generate plugin")
        return CaptureModelRunner(config, self.device)
