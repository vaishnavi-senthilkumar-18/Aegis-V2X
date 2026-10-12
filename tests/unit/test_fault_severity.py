import pytest

from simulation.fault_injection.severity import SeverityBand, severity_band, validate_severity


@pytest.mark.parametrize(
    "value,expected_band",
    [
        (0.0, SeverityBand.NONE),
        (0.1, SeverityBand.LOW),
        (0.25, SeverityBand.LOW),
        (0.3, SeverityBand.MEDIUM),
        (0.5, SeverityBand.MEDIUM),
        (0.6, SeverityBand.HIGH),
        (0.75, SeverityBand.HIGH),
        (0.9, SeverityBand.CRITICAL),
        (1.0, SeverityBand.CRITICAL),
    ],
)
def test_severity_band_mapping(value, expected_band):
    assert severity_band(value) == expected_band


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_severity_raises(invalid):
    with pytest.raises(ValueError):
        validate_severity(invalid)


def test_non_numeric_severity_raises():
    with pytest.raises(ValueError):
        validate_severity("high")
