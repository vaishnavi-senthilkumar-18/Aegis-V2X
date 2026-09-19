"""
Abstract contract for Trust-Adaptive Horizon Selection (TAHS).

Mathematical basis (03_Mathematical_Formulation.docx, Section 7):
    H_t = H_min + (H_max - H_min) * sigmoid(beta*T_t - gamma*C_t)
    H in {1, 2, 3, 5, 8, 10}   (discretized, see configs/model.yaml)

Monotonicity invariant (Section 11): for fixed criticality,
    T1 > T2  =>  H1 >= H2
i.e. higher trust never reduces the prediction horizon. Phase 5
implementations MUST satisfy this and it should be covered by a
dedicated property-based unit test.

Implemented in: Phase 5 (ai/twintrust_ap/tahs.py — this file, extended)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ai.utils.config_loader import load_config


class BaseTAHS(ABC):
    """Contract every concrete TAHS implementation must satisfy."""

    @abstractmethod
    def select_horizon(self, trust: float, criticality: float) -> int:
        """
        Select the prediction horizon given current trust and criticality.

        Parameters
        ----------
        trust : float
            Calibrated trust probability T_t, in [0, 1].
        criticality : float
            Criticality score C_t, in [0, 1].

        Returns
        -------
        int
            Selected horizon, one of the discretized values in
            configs/model.yaml -> tahs.horizon_discretization.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class TAHSParams:
    """`H_t` formula parameters. Defaults are loaded directly from
    `configs/model.yaml`'s `tahs:` section (unlike, e.g., the Trust
    Estimator's weights, these ARE the project's real, concrete config
    values -- not placeholders), not re-derived or hand-picked here."""

    horizon_min: int
    horizon_max: int
    beta: float
    gamma: float
    horizon_discretization: tuple[int, ...]

    @classmethod
    def from_model_config(cls) -> "TAHSParams":
        t = load_config("model")["tahs"]
        return cls(
            horizon_min=t["horizon_min"],
            horizon_max=t["horizon_max"],
            beta=t["beta"],
            gamma=t["gamma"],
            horizon_discretization=tuple(t["horizon_discretization"]),
        )


class TAHS(BaseTAHS):
    """Concrete Trust-Adaptive Horizon Selection (Phase 5).

    Implements exactly the formula this module's docstring specifies:

        H_t = H_min + (H_max - H_min) * sigmoid(beta*T_t - gamma*C_t)

    then snaps the continuous result to the nearest value in
    `configs/model.yaml`'s `tahs.horizon_discretization` set (nearest-value
    snapping, not floor/ceil -- the same discretization rule the Phase 2
    `BootstrapTAHS` reference implementation uses in
    `simulation/annotation/decision_labeler.py`, so both remain consistent
    with each other for the same inputs).

    Monotonicity (docs/interfaces.md invariant #2 -- for fixed criticality,
    T1 > T2 => H1 >= H2) holds because sigmoid is strictly increasing in its
    argument and beta > 0: increasing trust strictly increases the
    continuous H_t, and nearest-value snapping onto a fixed sorted
    discretization set preserves a non-decreasing (>=) relationship.
    """

    def __init__(self, params: TAHSParams | None = None):
        self._p = params or TAHSParams.from_model_config()

    def select_horizon(self, trust: float, criticality: float) -> int:
        if not (0.0 <= trust <= 1.0):
            raise ValueError(f"trust must be in [0, 1], got {trust!r}")
        if not (0.0 <= criticality <= 1.0):
            raise ValueError(f"criticality must be in [0, 1], got {criticality!r}")

        sigmoid_arg = self._p.beta * trust - self._p.gamma * criticality
        sigmoid_val = 1.0 / (1.0 + math.exp(-sigmoid_arg))
        continuous_h = self._p.horizon_min + (self._p.horizon_max - self._p.horizon_min) * sigmoid_val

        return min(self._p.horizon_discretization, key=lambda h: abs(h - continuous_h))
