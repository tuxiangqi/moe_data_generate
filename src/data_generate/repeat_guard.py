# SPDX-License-Identifier: Apache-2.0
"""Greedy repetition stop used by MoE expert-prediction collection.

Port of ExpertsCall native/moe-collector/repeat_guard.hpp.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class RepeatGuard:
    def __init__(
        self,
        period_max: int,
        min_tokens: int,
        diversity_window: int,
        diversity_max: int,
    ) -> None:
        self.period_max = max(0, period_max)
        self.min_tokens = max(0, min_tokens)
        self.diversity_window = max(0, diversity_window)
        self.diversity_max = max(0, diversity_max)
        self.matches = [0] * (self.period_max + 1)
        self.window: list[int] = []
        self.have: list[bool] = []
        self.next = 0
        self.counts: dict[int, int] = {}
        self.last_unique = 0

    def enabled(self) -> bool:
        return (self.period_max > 0 and self.min_tokens > 1) or (
            self.diversity_window > 1 and self.diversity_max > 0
        )

    def observe(self, tokens, generated: int) -> int:
        """Return 0 to continue, a positive period, or -1 for low diversity."""
        if not self.enabled():
            return 0
        n = len(tokens)
        period = self._periodic(tokens, n, generated)
        if period > 0:
            return period
        if self.diversity_window <= 1 or self.diversity_max <= 0 or n == 0:
            return 0
        token = int(tokens[n - 1])
        if not self.window:
            self.window = [0] * self.diversity_window
            self.have = [False] * self.diversity_window
        slot = self.window[self.next]
        if self.have[self.next]:
            left = self.counts[slot] - 1
            if left == 0:
                del self.counts[slot]
            else:
                self.counts[slot] = left
        else:
            self.have[self.next] = True
        self.window[self.next] = token
        self.counts[token] = self.counts.get(token, 0) + 1
        self.next = (self.next + 1) % len(self.window)
        self.last_unique = len(self.counts)
        if generated >= self.diversity_window and self.last_unique <= self.diversity_max:
            return -1
        return 0

    def _periodic(self, tokens, n: int, generated: int) -> int:
        if self.period_max <= 0 or self.min_tokens <= 1:
            return 0
        check = generated >= self.min_tokens
        last = int(tokens[n - 1])
        for p in range(1, min(self.period_max, n - 1) + 1):
            if last == int(tokens[n - 1 - p]):
                self.matches[p] += 1
            else:
                self.matches[p] = 0
            if check:
                need = max(self.min_tokens, 2 * p)
                if self.matches[p] + 1 >= need:
                    return p
        return 0


def guard_for_request(request) -> RepeatGuard:
    existing = getattr(request, "moe_repeat", None)
    if existing is not None:
        return existing
    params = request.sampling_params
    extra = {}
    if params is not None and params.extra_args:
        extra = params.extra_args.get("moe_repeat") or {}

    def value(key: str, env: str, default: int) -> int:
        if key in extra:
            return int(extra[key])
        return int(os.environ.get(env, default))

    guard = RepeatGuard(
        period_max=value("period_max", "MOE_REPEAT_PERIOD_MAX", 512),
        min_tokens=value("min_tokens", "MOE_REPEAT_MIN_TOKENS", 96),
        diversity_window=value("diversity_window", "MOE_REPEAT_DIVERSITY_WINDOW", 2048),
        diversity_max=value("diversity_max", "MOE_REPEAT_DIVERSITY_MAX", 64),
    )
    request.moe_repeat = guard
    return guard


def write_stop(request) -> None:
    root = os.environ.get("MOE_COLLECT_DIR")
    if not root:
        return
    folder = Path(root) / request.request_id
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "stop_reason": request.stop_reason,
        "repeat_kind": getattr(request, "moe_repeat_kind", None),
        "repeat_period": getattr(request, "moe_repeat_period", 0),
        "repeat_unique": getattr(request, "moe_repeat_unique", 0),
    }
    (folder / "stop.json").write_text(json.dumps(payload), encoding="utf-8")
