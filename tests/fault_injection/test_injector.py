from simulation.fault_injection.injector import FaultInjector
from simulation.fault_injection.schemas import (
    FaultCategory,
    FaultConfig,
    FaultType,
    ScenarioFaultPlan,
)


def _make_plan(scenario_id="TestScenario", seed=42, faults=None):
    return ScenarioFaultPlan(scenario_id=scenario_id, seed=seed, faults=faults or [])


def test_no_faults_passes_frames_through_unchanged(synthetic_scenario):
    plan = _make_plan(faults=[])
    injector = FaultInjector(plan)
    result = injector.inject(synthetic_scenario)
    assert len(result.frames) == len(synthetic_scenario)
    assert len(result.dropped_frame_indices) == 0


def test_single_fault_applied_to_all_frames(synthetic_scenario):
    plan = _make_plan(
        faults=[
            FaultConfig(
                type=FaultType.STALE_STATE,
                category=FaultCategory.DIGITAL_TWIN,
                severity=0.5,
                start_frame=0,
                end_frame=None,
            )
        ]
    )
    injector = FaultInjector(plan)
    result = injector.inject(synthetic_scenario)
    assert len(result.frames) == len(synthetic_scenario)
    for frame in result.frames:
        assert frame["twin_age_ms"] > 0


def test_fault_respects_frame_range(synthetic_scenario):
    plan = _make_plan(
        faults=[
            FaultConfig(
                type=FaultType.STALE_STATE,
                category=FaultCategory.DIGITAL_TWIN,
                severity=0.5,
                start_frame=10,
                end_frame=15,
            )
        ]
    )
    injector = FaultInjector(plan)
    result = injector.inject(synthetic_scenario)
    # Frames outside [10, 15] should be unaffected (twin_age_ms stays at baseline 0.0)
    assert result.frames[0]["twin_age_ms"] == 0.0
    assert result.frames[12]["twin_age_ms"] > 0.0
    assert result.frames[18]["twin_age_ms"] == 0.0


def test_full_packet_loss_drops_every_frame(synthetic_scenario):
    plan = _make_plan(
        faults=[
            FaultConfig(
                type=FaultType.PACKET_LOSS,
                category=FaultCategory.COMMUNICATION,
                severity=1.0,
                start_frame=0,
                end_frame=None,
            )
        ]
    )
    injector = FaultInjector(plan)
    result = injector.inject(synthetic_scenario)
    assert len(result.frames) == 0
    assert len(result.dropped_frame_indices) == len(synthetic_scenario)


def test_composable_faults_both_apply(synthetic_scenario):
    plan = _make_plan(
        faults=[
            FaultConfig(
                type=FaultType.STALE_STATE,
                category=FaultCategory.DIGITAL_TWIN,
                severity=0.3,
                start_frame=0,
                end_frame=None,
            ),
            FaultConfig(
                type=FaultType.LATENCY,
                category=FaultCategory.COMMUNICATION,
                severity=0.4,
                start_frame=0,
                end_frame=None,
            ),
        ]
    )
    injector = FaultInjector(plan)
    result = injector.inject(synthetic_scenario)
    frame0 = result.frames[0]
    assert frame0["twin_age_ms"] > 0
    assert frame0["wireless_timestamp"] > synthetic_scenario[0]["wireless_timestamp"]


def test_reproducibility_same_plan_same_output(synthetic_scenario):
    plan_a = _make_plan(
        seed=99,
        faults=[
            FaultConfig(
                type=FaultType.LIDAR_NOISE,
                category=FaultCategory.PERCEPTION,
                severity=0.5,
                start_frame=0,
                end_frame=None,
            )
        ],
    )
    plan_b = _make_plan(
        seed=99,
        faults=[
            FaultConfig(
                type=FaultType.LIDAR_NOISE,
                category=FaultCategory.PERCEPTION,
                severity=0.5,
                start_frame=0,
                end_frame=None,
            )
        ],
    )
    result_a = FaultInjector(plan_a).inject(synthetic_scenario)
    result_b = FaultInjector(plan_b).inject(synthetic_scenario)

    for frame_a, frame_b in zip(result_a.frames, result_b.frames):
        import numpy as np

        np.testing.assert_array_equal(frame_a["lidar"], frame_b["lidar"])


def test_different_seeds_produce_different_output(synthetic_scenario):
    plan_a = _make_plan(
        seed=1,
        faults=[
            FaultConfig(
                type=FaultType.LIDAR_NOISE,
                category=FaultCategory.PERCEPTION,
                severity=0.5,
                start_frame=0,
                end_frame=None,
            )
        ],
    )
    plan_b = _make_plan(
        seed=2,
        faults=[
            FaultConfig(
                type=FaultType.LIDAR_NOISE,
                category=FaultCategory.PERCEPTION,
                severity=0.5,
                start_frame=0,
                end_frame=None,
            )
        ],
    )
    result_a = FaultInjector(plan_a).inject(synthetic_scenario)
    result_b = FaultInjector(plan_b).inject(synthetic_scenario)

    assert not (result_a.frames[0]["lidar"] == result_b.frames[0]["lidar"]).all()
