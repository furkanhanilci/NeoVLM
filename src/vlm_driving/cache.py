"""Time-aware cache for asynchronous slow VLM features."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CachedTokens:
    tokens: torch.Tensor
    timestamp_s: float


class AsyncTokenCache:
    """Stores latest compact VLM tokens and reports token age on read."""

    def __init__(self, max_age_s: float) -> None:
        self.max_age_s = max_age_s
        self._latest: CachedTokens | None = None

    def update(self, tokens: torch.Tensor, timestamp_s: float) -> None:
        self._latest = CachedTokens(tokens=tokens.detach(), timestamp_s=timestamp_s)

    def read(self, now_s: float) -> tuple[torch.Tensor | None, float, bool]:
        if self._latest is None:
            return None, float("inf"), False

        age_s = max(0.0, now_s - self._latest.timestamp_s)
        return self._latest.tokens, age_s, age_s <= self.max_age_s
