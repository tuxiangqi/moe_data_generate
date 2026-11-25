# SPDX-License-Identifier: Apache-2.0
"""Store MoE router tensors by request after each model forward."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from vllm.distributed.parallel_state import get_tp_group

logger = logging.getLogger(__name__)

class ExpertPredictionCapturer:
    def __init__(
        self,
        max_num_batched_tokens: int,
        num_layers: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        capture_layers: tuple[int, ...],
    ) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.capture_layers = capture_layers
        self._layer_to_slot = {layer: slot for slot, layer in enumerate(capture_layers)}
        self.expert_ids = torch.empty(
            (max_num_batched_tokens, num_layers, top_k),
            dtype=torch.int32,
            device=device,
        )
        self.expert_scores = torch.empty(
            (max_num_batched_tokens, num_layers, num_experts),
            dtype=torch.float32,
            device=device,
        )
        self.router_weights = torch.empty(
            (max_num_batched_tokens, num_layers, top_k),
            dtype=torch.float32,
            device=device,
        )
        self.hidden = torch.empty(
            (max_num_batched_tokens, len(capture_layers), hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self._copy_stream = torch.cuda.Stream()
        self._cursor: dict[str, int] = {}
        self.rows_by_layer: dict[int, int] = {}
        self._root = Path(os.environ["MOE_COLLECT_DIR"])
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info(
            "MoE prediction capture: layers=%d experts=%d topk=%d hidden=%d "
            "capture_layers=%s max_tokens=%d dir=%s",
            num_layers,
            num_experts,
            top_k,
            hidden_size,
            list(capture_layers),
            max_num_batched_tokens,
            self._root,
        )

    def capture(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        if layer_id < 0 or layer_id >= self.num_layers:
            return
        n = hidden_states.shape[0]
        self.rows_by_layer[layer_id] = n
        logits = router_logits[:, : self.num_experts]
        self.expert_ids[:n, layer_id].copy_(topk_ids[:n])
        scores = self.expert_scores[:n, layer_id]
        scores.copy_(logits[:n])
        self.router_weights[:n, layer_id].copy_(topk_weights[:n])
        slot = self._layer_to_slot.get(layer_id)
        if slot is not None:
            self.hidden[:n, slot].copy_(hidden_states[:n])

    def forget(self, request_id: str) -> None:
        self._cursor.pop(request_id, None)

    def flush(self, input_batch) -> None:
        if not get_tp_group().is_first_rank:
            return
        query = input_batch.query_start_loc_np
        starts = input_batch.num_computed_tokens_np
        n = int(query[-1])
        if n <= 0:
            return
        missing = [
            layer for layer in range(self.num_layers)
            if self.rows_by_layer.get(layer, 0) < n
        ]
        if missing:
            raise RuntimeError(f"MoE capture missing {n} rows at layers {missing}")
        compute = torch.cuda.current_stream()
        ids = self.expert_ids[:n]
        scores = self.expert_scores[:n]
        weights = self.router_weights[:n]
        hidden = self.hidden[:n]
        event = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_stream(compute)
            ids_cpu = ids.to("cpu", non_blocking=True)
            scores_cpu = scores.to("cpu", non_blocking=True)
            weights_cpu = weights.to("cpu", non_blocking=True)
            hidden_cpu = hidden.to("cpu", non_blocking=True)
            event.record(self._copy_stream)
        event.synchronize()
        ids_np = ids_cpu.numpy()
        scores_np = scores_cpu.numpy()
        weights_np = weights_cpu.numpy()
        hidden_np = hidden_cpu.view(torch.uint16).numpy()
        for i, req_id in enumerate(input_batch.req_ids):
            begin = int(query[i])
            end = int(query[i + 1])
            rows = end - begin
            if rows <= 0:
                continue
            token_start = int(starts[i])
            seen = self._cursor.get(req_id, 0)
            if token_start != seen:
                raise RuntimeError(
                    f"MoE capture gap for {req_id}: file has {seen} tokens, "
                    f"this step starts at {token_start}"
                )
            folder = self._root / req_id
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "layers.json").exists():
                (folder / "layers.json").write_text(
                    json.dumps(
                        {
                            "capture_layers": list(self.capture_layers),
                            "num_layers": self.num_layers,
                            "num_experts": self.num_experts,
                            "top_k": self.top_k,
                            "hidden_size": self.hidden_size,
                        }
                    ),
                    encoding="utf-8",
                )
            _append(folder / "expert_ids.bin", ids_np[begin:end])
            _append(folder / "expert_scores.bin", scores_np[begin:end])
            _append(folder / "router_weights_topk.bin", weights_np[begin:end])
            _append(folder / "hidden_states.bin", hidden_np[begin:end])
            self._cursor[req_id] = seen + rows


def _append(path: Path, array: np.ndarray) -> None:
    with path.open("ab") as handle:
        array.tofile(handle)


def capture_layers_from_env(num_layers: int) -> tuple[int, ...]:
    raw = os.environ.get("MOE_CAPTURE_LAYERS")
    if raw is None:
        raise ValueError("MOE_CAPTURE_LAYERS is required for MoE collection")
    try:
        layers = json.loads(raw)
    except ValueError as exc:
        raise ValueError("MOE_CAPTURE_LAYERS must be a JSON array of layer IDs") from exc
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(layer) is not int or not 0 <= layer < num_layers for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError(
            f"MOE_CAPTURE_LAYERS must contain unique layer IDs in 0..{num_layers - 1}"
        )
    return tuple(layers)
