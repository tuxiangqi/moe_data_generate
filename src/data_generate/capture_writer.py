"""Buffer CPU snapshots and publish complete captures from one writer thread."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep

import numpy as np


FILES = ("expert_ids.bin", "expert_scores.bin", "router_weights_topk.bin", "hidden_states.bin")
READY = ".capture-ready"
ERROR = ".capture-error"
MAX_BUFFERED_STEPS = 32
MAX_BUFFERED_BYTES = 32 * 2**20
MAX_PENDING_WRITES = 2


class BufferedCaptureWriter:
    def __init__(self, root: Path, layers_json: str) -> None:
        self._root = root
        self._layers_json = layers_json
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moe-capture")
        self._jobs = deque()
        self._pending = {}
        self._buffered_bytes = 0
        self._buffered_steps = 0
        self._opened = set()
        self._failed = False

    def _submit(self, operation, arg) -> None:
        while self._jobs and self._jobs[0].done():
            self._jobs.popleft().result()
        # Bound outstanding CPU snapshots, including when shared storage stalls.
        if len(self._jobs) >= MAX_PENDING_WRITES:
            self._jobs.popleft().result()
        self._jobs.append(self._executor.submit(self._run, operation, arg))

    def _run(self, operation, arg) -> None:
        if self._failed:
            raise RuntimeError("MoE capture writer has failed")
        try:
            operation(arg)
        except Exception as exc:
            self._failed = True
            try:
                (self._root / ERROR).write_text(repr(exc), encoding="utf-8")
            except OSError:
                pass
            raise

    def submit(self, chunks: dict[str, tuple[np.ndarray, ...]]) -> None:
        self._submit(self._buffer, chunks)

    def finish(self, request_ids: list[str]) -> None:
        if request_ids:
            self._submit(self._finish, request_ids)

    def _buffer(self, chunks: dict[str, tuple[np.ndarray, ...]]) -> None:
        for request_id, arrays in chunks.items():
            self._pending.setdefault(request_id, []).append(arrays)
            self._buffered_bytes += sum(array.nbytes for array in arrays)
        self._buffered_steps += 1
        if (self._buffered_steps >= MAX_BUFFERED_STEPS
                or self._buffered_bytes >= MAX_BUFFERED_BYTES):
            self._drain(None)

    def _write_pending(self, request_id: str) -> None:
        chunks = self._pending.pop(request_id, None)
        if chunks is None:
            return
        folder = self._root / request_id
        if request_id not in self._opened:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "layers.json").write_text(self._layers_json, encoding="utf-8")
            self._opened.add(request_id)
        for filename, arrays in zip(FILES, zip(*chunks), strict=True):
            data = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
            with (folder / filename).open("ab") as handle:
                data.tofile(handle)

    def _drain(self, _) -> None:
        for request_id in list(self._pending):
            self._write_pending(request_id)
        self._buffered_bytes = self._buffered_steps = 0

    def _finish(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self._write_pending(request_id)
            (self._root / request_id / READY).touch()
            self._opened.remove(request_id)

    def close(self) -> None:
        try:
            self._submit(self._drain, None)
        finally:
            self._executor.shutdown(wait=True)
        for job in self._jobs:
            job.result()


def wait_for_capture(folder: Path, timeout: float = 120.0) -> None:
    deadline = monotonic() + timeout
    while not (folder / READY).is_file():
        failure = folder.parent / ERROR
        if failure.is_file():
            raise RuntimeError(f"MoE capture writer failed: {failure.read_text(encoding='utf-8')}")
        if monotonic() >= deadline:
            raise TimeoutError(f"MoE capture writer did not publish {folder} within {timeout:g}s")
        sleep(0.02)
