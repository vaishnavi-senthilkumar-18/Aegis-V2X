import json

from simulation.fault_injection.fault_logger import FaultLogger


def test_logger_writes_event(tmp_path):
    log_path = tmp_path / "faults.jsonl"
    logger = FaultLogger(log_path)

    logger.log_event(
        component="communication",
        fault="packet_loss",
        parameters={"packet_loss_rate": 1.0},
        outcome={"delivered": False},
        scenario_id="test_scenario_01",
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    saved_event = json.loads(lines[0])
    assert saved_event["fault"] == "packet_loss"
    assert saved_event["scenario_id"] == "test_scenario_01"
    assert saved_event["outcome"]["delivered"] is False
    assert saved_event["component"] == "communication"


def test_logger_appends_multiple_events(tmp_path):
    log_path = tmp_path / "faults.jsonl"
    logger = FaultLogger(log_path)

    logger.log_event(
        component="communication",
        fault="packet_loss",
        parameters={"packet_loss_rate": 0.5},
        outcome={"delivered": False},
    )
    logger.log_event(
        component="sensing",
        fault="noise",
        parameters={"noise_std": 0.2},
        outcome={"value": 10.1},
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["fault"] == "packet_loss"
    assert json.loads(lines[1])["fault"] == "noise"
