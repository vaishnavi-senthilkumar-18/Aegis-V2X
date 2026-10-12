"""
Severity scale definitions for Aegis-V2X fault injection.

All fault types operate on a normalized severity scalar in [0, 1]. This module defines
the shared severity bands and a helper to label a numeric severity value, so that logs,
scenario catalogs, and plots use consistent terminology across every fault type.
"""

from __future__ import annotations

from enum import Enum


class SeverityBand(str, Enum):
    """Qualitative severity bands, aligned with the Trust probability bands defined in
    03_Mathematical_Formulation.docx Sec. 5, for conceptual consistency across the project.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# (lower_bound_inclusive, upper_bound_inclusive, band)
_SEVERITY_BANDS: list[tuple[float, float, SeverityBand]] = [
    (0.0, 0.0, SeverityBand.NONE),
    (0.0001, 0.25, SeverityBand.LOW),
    (0.2501, 0.50, SeverityBand.MEDIUM),
    (0.5001, 0.75, SeverityBand.HIGH),
    (0.7501, 1.0, SeverityBand.CRITICAL),
]


def validate_severity(severity: float) -> float:
    """Validate that severity is a finite float in [0, 1]. Raises ValueError otherwise.

    Parameters
    ----------
    severity : float
        Candidate severity value.

    Returns
    -------
    float
        The same value, if valid.

    Raises
    ------
    ValueError
        If severity is not in [0, 1] or is not a real number.
    """
    try:
        value = float(severity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Severity must be numeric, got: {severity!r}") from exc

    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Severity must be in [0.0, 1.0], got: {value}")

    return value


def severity_band(severity: float) -> SeverityBand:
    """Map a numeric severity value to its qualitative band.

    Parameters
    ----------
    severity : float
        Severity value in [0, 1].

    Returns
    -------
    SeverityBand
        The qualitative band this severity falls into.

    Example
    -------
    >>> severity_band(0.5)
    <SeverityBand.MEDIUM: 'medium'>
    """
    value = validate_severity(severity)
    for lower, upper, band in _SEVERITY_BANDS:
        if lower <= value <= upper:
            return band
    # Should be unreachable given validate_severity's bounds check.
    raise ValueError(f"Severity {value} did not match any defined band.")
