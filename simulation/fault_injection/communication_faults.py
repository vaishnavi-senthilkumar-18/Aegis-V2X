
"""Deterministic communication fault injection for Aegis-V2X tests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FaultResult:
    """Outcome of processing one communication message."""

    delivered: bool
    message: Any
    delay_ms: float
    fault: str


class CommunicationFaultInjector:
    """Apply configurable packet loss and simulated message delay."""

    def __init__(
        self,
        packet_loss_rate: float = 0.0,
        delay_ms: float = 0.0,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= packet_loss_rate <= 1.0:
            raise ValueError("packet_loss_rate must be between 0 and 1.")
        if delay_ms < 0.0:
            raise ValueError("delay_ms cannot be negative.")

        self.packet_loss_rate = packet_loss_rate
        self.delay_ms = delay_ms
        self._random = random.Random(seed)

    def apply(self, message: Any) -> FaultResult:
        """Simulate communication loss or delay for one message."""
        if self._random.random() < self.packet_loss_rate:
            return FaultResult(
                delivered=False,
                message=None,
                delay_ms=0.0,
                fault="packet_loss",
            )

        fault = "delay" if self.delay_ms > 0.0 else "none"
        return FaultResult(
            delivered=True,
            message=message,
            delay_ms=self.delay_ms,
            fault=fault,
        )
