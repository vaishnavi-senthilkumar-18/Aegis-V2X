
"""Deterministic sensing degradation for Aegis-V2X tests."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SensingFaultResult:
    """Outcome of applying degradation to a sensor measurement."""

    value: float
    fault: str


class SensingFaultInjector:
    """Simulate bounded noise or a fixed sensor bias."""

    def __init__(
        self,
        noise_std: float = 0.0,
        bias: float = 0.0,
        seed: int = 42,
    ) -> None:
        if noise_std < 0.0:
            raise ValueError("noise_std cannot be negative.")

        self.noise_std = noise_std
        self.bias = bias
        self._random = random.Random(seed)

    def apply(self, measurement: float) -> SensingFaultResult:
        """Return a measurement affected by configured sensor faults."""
        noise = (
            self._random.gauss(0.0, self.noise_std)
            if self.noise_std > 0.0
            else 0.0
        )
        value = measurement + self.bias + noise

        if self.noise_std > 0.0 and self.bias != 0.0:
            fault = "noise_and_bias"
        elif self.noise_std > 0.0:
            fault = "noise"
        elif self.bias != 0.0:
            fault = "bias"
        else:
            fault = "none"

        return SensingFaultResult(value=value, fault=fault)
