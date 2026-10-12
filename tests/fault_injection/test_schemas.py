import pytest
from pydantic import ValidationError

from simulation.fault_injection.schemas import (
    FaultCategory,
    FaultConfig,
    FaultType,
    ScenarioFaultPlan,
)


def test_valid_fault_config():
    config = FaultConfig(
        type=FaultType.PACKET_LOSS,
        category=FaultCategory.COMMUNICATION,
        severity=0.5,
        start_frame=0,
        end_frame=100,
    )
    assert config.severity == 0.5


def test_severity_out_of_range_rejected():
    with pytest.raises(ValidationError):
        FaultConfig(
            type=FaultType.PACKET_LOSS,
            category=FaultCategory.COMMUNICATION,
            severity=1.5,
        )


def test_mismatched_category_rejected():
    with pytest.raises(ValidationError):
        FaultConfig(
            type=FaultType.PACKET_LOSS,  # belongs to COMMUNICATION
            category=FaultCategory.PERCEPTION,  # wrong category
            severity=0.5,
        )


def test_end_frame_before_start_frame_rejected():
    with pytest.raises(ValidationError):
        FaultConfig(
            type=FaultType.PACKET_LOSS,
            category=FaultCategory.COMMUNICATION,
            severity=0.5,
            start_frame=100,
            end_frame=50,
        )


def test_end_frame_none_means_always_active():
    config = FaultConfig(
        type=FaultType.STALE_STATE,
        category=FaultCategory.DIGITAL_TWIN,
        severity=0.3,
        start_frame=0,
        end_frame=None,
    )
    assert config.is_active_at(0)
    assert config.is_active_at(9999)


def test_is_active_at_respects_frame_range():
    config = FaultConfig(
        type=FaultType.LATENCY,
        category=FaultCategory.COMMUNICATION,
        severity=0.5,
        start_frame=10,
        end_frame=20,
    )
    assert not config.is_active_at(5)
    assert config.is_active_at(10)
    assert config.is_active_at(15)
    assert config.is_active_at(20)
    assert not config.is_active_at(21)


def test_scenario_fault_plan_faults_active_at():
    plan = ScenarioFaultPlan(
        scenario_id="TestScene",
        seed=1,
        faults=[
            FaultConfig(
                type=FaultType.PACKET_LOSS,
                category=FaultCategory.COMMUNICATION,
                severity=0.5,
                start_frame=0,
                end_frame=10,
            ),
            FaultConfig(
                type=FaultType.STALE_STATE,
                category=FaultCategory.DIGITAL_TWIN,
                severity=0.3,
                start_frame=5,
                end_frame=None,
            ),
        ],
    )
    active_at_3 = plan.faults_active_at(3)
    active_at_7 = plan.faults_active_at(7)
    active_at_15 = plan.faults_active_at(15)

    assert len(active_at_3) == 1
    assert len(active_at_7) == 2
    assert len(active_at_15) == 1
    assert active_at_15[0].type == FaultType.STALE_STATE
