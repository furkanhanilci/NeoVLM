"""Token provider contracts for VLM hidden-state producers."""

from __future__ import annotations

import time
from typing import Protocol

import torch


class TokenProvider(Protocol):
    """Produces VLM-like hidden states for a batch of observations."""

    hidden_size: int

    def encode(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Return hidden states shaped [batch, seq_len, hidden_size]."""


class DummyTokenProvider:
    """Deterministic synthetic provider used before real VLM weights are available."""

    def __init__(
        self,
        hidden_size: int,
        seed: int = 0,
        simulated_latency_s: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if simulated_latency_s < 0:
            raise ValueError("simulated_latency_s must be non-negative")
        self.hidden_size = hidden_size
        self.simulated_latency_s = simulated_latency_s
        self.device = torch.device(device)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def encode(self, batch_size: int, seq_len: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.simulated_latency_s:
            time.sleep(self.simulated_latency_s)
        tokens = torch.randn(
            batch_size,
            seq_len,
            self.hidden_size,
            generator=self._generator,
            dtype=torch.float32,
        )
        return tokens.to(self.device)
