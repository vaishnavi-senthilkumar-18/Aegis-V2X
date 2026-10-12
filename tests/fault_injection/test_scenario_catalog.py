from simulation.fault_injection.scenario_catalog import ScenarioCatalog
from simulation.fault_injection.schemas import (
    FaultCategory,
    FaultConfig,
    FaultType,
    ScenarioFaultPlan,
)


def _make_plan(scenario_id="TestScenario", seed=1):
    return ScenarioFaultPlan(
        scenario_id=scenario_id,
        seed=seed,
        faults=[
            FaultConfig(
                type=FaultType.PACKET_LOSS,
                category=FaultCategory.COMMUNICATION,
                severity=0.5,
                start_frame=0,
                end_frame=None,
            )
        ],
    )


def test_catalog_creates_file_if_missing(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    assert not catalog_path.exists()
    ScenarioCatalog(catalog_path)
    assert catalog_path.exists()


def test_add_entry_and_retrieve(tmp_path):
    catalog = ScenarioCatalog(tmp_path / "catalog.json")
    plan = _make_plan()
    catalog.add_entry(plan, num_frames_in=20, num_frames_out=15, num_frames_dropped=5)

    entries = catalog.all_entries()
    assert len(entries) == 1
    assert entries[0]["scenario_id"] == "TestScenario"
    assert entries[0]["num_frames_dropped"] == 5


def test_find_by_scenario(tmp_path):
    catalog = ScenarioCatalog(tmp_path / "catalog.json")
    catalog.add_entry(_make_plan(scenario_id="SceneA"), 10, 10, 0)
    catalog.add_entry(_make_plan(scenario_id="SceneB"), 10, 10, 0)

    results = catalog.find_by_scenario("SceneA")
    assert len(results) == 1
    assert results[0]["scenario_id"] == "SceneA"


def test_find_by_fault_type(tmp_path):
    catalog = ScenarioCatalog(tmp_path / "catalog.json")
    catalog.add_entry(_make_plan(scenario_id="SceneA"), 10, 10, 0)

    results = catalog.find_by_fault_type("packet_loss")
    assert len(results) == 1

    no_results = catalog.find_by_fault_type("gps_drift")
    assert len(no_results) == 0


def test_catalog_persists_across_instances(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog1 = ScenarioCatalog(catalog_path)
    catalog1.add_entry(_make_plan(), 10, 10, 0)

    catalog2 = ScenarioCatalog(catalog_path)
    assert len(catalog2.all_entries()) == 1
