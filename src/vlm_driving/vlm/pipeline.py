"""Synchronous slow-token path wiring provider, resampler, and cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vlm_driving.cache import AsyncTokenCache
from vlm_driving.models import QueryResampler
from vlm_driving.vlm.provider import TokenProvider


@dataclass(frozen=True)
class TokenizerStepResult:
    hidden_states: torch.Tensor
    compact_tokens: torch.Tensor
    timestamp_s: float


class SlowTokenizer:
    """Runs the slow VLM-like token path synchronously for smoke testing."""

    def __init__(
        self,
        provider: TokenProvider,
        resampler: QueryResampler,
        cache: AsyncTokenCache,
        batch_size: int,
        seq_len: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.provider = provider
        self.resampler = resampler
        self.cache = cache
        self.batch_size = batch_size
        self.seq_len = seq_len

    def step(self, now_s: float) -> TokenizerStepResult:
        with torch.no_grad():
            hidden_states = self.provider.encode(batch_size=self.batch_size, seq_len=self.seq_len)
            compact_tokens = self.resampler(hidden_states)
        self.cache.update(compact_tokens, timestamp_s=now_s)
        return TokenizerStepResult(
            hidden_states=hidden_states,
            compact_tokens=compact_tokens,
            timestamp_s=now_s,
        )
