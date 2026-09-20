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

============================================================================
PHASE 5 THREE-INPUT EXTENSION (Trust + Criticality + GDCA Authority)
============================================================================
`select_horizon` gained a third, OPTIONAL parameter, `authority: float =
0.0` (`ai/twintrust_ap/gdca.py`'s Authority score A_t, in [0, 1]),
entering the same sigmoid argument as one more linear term:

    sigmoid_arg = beta*trust - gamma*criticality + delta*authority

`delta` is a PROVISIONAL, UNCALIBRATED architectural coefficient
(default 1.0) -- exactly the same status `beta`/`gamma` already have
(see docs/interfaces.md and configs/model.yaml's own comments): it is
NOT optimized, learned, validated, or empirically calibrated against any
outcome. It was chosen only to match the existing pattern (each input
enters linearly, with its own coefficient) established by
beta/gamma -- see the 2026-09-20 TAHS three-input design audit
(`claude/project_status.md`) for the full justification, including why
no authoritative 3-input formula exists anywhere else in this
repository and why this is therefore a new, explicitly-flagged design
choice rather than a lookup from an existing specification.

BACKWARD COMPATIBILITY: `authority` defaults to `0.0`, the additive
identity for a linear term (`delta * 0.0 == 0.0` for any `delta`), so
`select_horizon(trust, criticality)` and
`select_horizon(trust, criticality, 0.0)` are mathematically identical
and both reproduce the original two-input formula's output exactly.
This is NOT a claim that `authority=0.0` is semantically "neutral" for
GDCA (GDCA's own docstring documents `authority=0.0` as meaning "zero
consistency/authority", not "no information") -- it only guarantees
that OMITTING the argument changes nothing.

Monotonicity invariant, extended (docs/interfaces.md): for fixed trust
and criticality, `A1 > A2 => H1 >= H2` (holds automatically whenever
`delta >= 0`, by the same sigmoid-monotonicity argument already used
for trust).

GDCA's own mathematics (`ai/twintrust_ap/gdca.py`) are NOT modified by
this extension; this module only accepts Authority as an already-
computed `float` argument, exactly as it already accepts `trust` and
`criticality`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ai.utils.config_loader import load_config


class BaseTAHS(ABC):
    """Contract every concrete TAHS implementation must satisfy."""

    @abstractmethod
    def select_horizon(self, trust: float, criticality: float, authority: float = 0.0) -> int:
        """
        Select the prediction horizon given current trust, criticality,
        and (optionally) GDCA Authority.

        Parameters
        ----------
        trust : float
            Calibrated trust probability T_t, in [0, 1].
        criticality : float
            Criticality score C_t, in [0, 1].
        authority : float, optional
            GDCA Authority score A_t, in [0, 1]. Defaults to 0.0, the
            additive identity for the `delta*authority` term -- omitting
            it reproduces the original two-input formula exactly (see
            module docstring's "PHASE 5 THREE-INPUT EXTENSION").

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
    values -- not placeholders), not re-derived or hand-picked here.

    `delta` (authority sensitivity) is a PROVISIONAL, UNCALIBRATED
    architectural coefficient -- same status as `beta`/`gamma` -- added
    for the Phase 5 three-input extension (see module docstring). It is
    loaded from `configs/model.yaml` exactly like `beta`/`gamma`, not
    independently derived or optimized here.
    """

    horizon_min: int
    horizon_max: int
    beta: float
    gamma: float
    delta: float
    horizon_discretization: tuple[int, ...]

    @classmethod
    def from_model_config(cls) -> "TAHSParams":
        t = load_config("model")["tahs"]
        return cls(
            horizon_min=t["horizon_min"],
            horizon_max=t["horizon_max"],
            beta=t["beta"],
            gamma=t["gamma"],
            delta=t["delta"],
            horizon_discretization=tuple(t["horizon_discretization"]),
        )


class TAHS(BaseTAHS):
    """Concrete Trust-Adaptive Horizon Selection (Phase 5).

    Implements exactly the formula this module's docstring specifies:

        H_t = H_min + (H_max - H_min) * sigmoid(beta*T_t - gamma*C_t + delta*A_t)

    (the `delta*A_t` term is the Phase 5 three-input extension -- see
    module docstring's "PHASE 5 THREE-INPUT EXTENSION"; `A_t` defaults to
    0.0, reproducing the original two-input formula exactly when omitted)
    then snaps the continuous result to the nearest value in
    `configs/model.yaml`'s `tahs.horizon_discretization` set (nearest-value
    snapping, not floor/ceil -- the same discretization rule the Phase 2
    `BootstrapTAHS` reference implementation uses in
    `simulation/annotation/decision_labeler.py`, so both remain consistent
    with each other for the same inputs).

    Monotonicity (docs/interfaces.md invariant #2 -- for fixed criticality
    and authority, T1 > T2 => H1 >= H2) holds because sigmoid is strictly
    increasing in its argument and beta > 0: increasing trust strictly
    increases the continuous H_t, and nearest-value snapping onto a fixed
    sorted discretization set preserves a non-decreasing (>=) relationship.
    The same argument extends to authority (fixed trust/criticality,
    A1 > A2 => H1 >= H2) since delta >= 0.
    """

    def __init__(self, params: TAHSParams | None = None):
        self._p = params or TAHSParams.from_model_config()

    def select_horizon(self, trust: float, criticality: float, authority: float = 0.0) -> int:
        if not (0.0 <= trust <= 1.0):
            raise ValueError(f"trust must be in [0, 1], got {trust!r}")
        if not (0.0 <= criticality <= 1.0):
            raise ValueError(f"criticality must be in [0, 1], got {criticality!r}")
        if not (0.0 <= authority <= 1.0):
            raise ValueError(f"authority must be in [0, 1], got {authority!r}")

        sigmoid_arg = self._p.beta * trust - self._p.gamma * criticality + self._p.delta * authority
        sigmoid_val = 1.0 / (1.0 + math.exp(-sigmoid_arg))
        continuous_h = self._p.horizon_min + (self._p.horizon_max - self._p.horizon_min) * sigmoid_val

        return min(self._p.horizon_discretization, key=lambda h: abs(h - continuous_h))
