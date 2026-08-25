"""Unit tests for simulation.synchronization — the <=10ms tolerance from
configs/simulation.yaml `synchronization.tolerance_ms` (frozen, Phase 1)."""

import pytest

from simulation.synchronization.buffer import TimestampedBuffer
from simulation.synchronization.synchronizer import StreamSynchronizer, SyncResult, SyncViolation


def test_buffer_nearest_finds_closest_timestamp():
    buf = TimestampedBuffer()
    buf.extend([(0.0, "a"), (0.1, "b"), (0.2, "c")])
    match = buf.nearest(0.095)
    assert match.data == "b"
    assert match.delta_seconds == pytest.approx(0.005, abs=1e-9)


def test_buffer_nearest_empty_returns_none():
    assert TimestampedBuffer().nearest(1.0) is None


def test_synchronizer_within_tolerance_produces_sync_result():
    sync = StreamSynchronizer(tolerance_ms=10.0)
    csi_buf = TimestampedBuffer()
    csi_buf.extend([(0.000, {"snr_db": 20.0}), (0.100, {"snr_db": 21.0})])
    gps_buf = TimestampedBuffer()
    gps_buf.extend(
        [(0.003, {"timestamp": 0.003, "lat": 1.0}), (0.101, {"timestamp": 0.101, "lat": 1.0})]
    )

    sync.register_stream("csi", csi_buf, is_anchor=True)
    sync.register_stream("gps", gps_buf)

    results, violations = StreamSynchronizer.split_results(
        sync.synchronize(scene_id="SceneTest", vehicle_id=1)
    )

    assert len(violations) == 0
    assert len(results) == 2
    assert all(isinstance(r, SyncResult) for r in results)
    assert results[0].max_offset_ms == pytest.approx(3.0, abs=1e-6)
    assert results[0].wireless_timestamp == 0.0
    assert results[0].simulation_timestamp == pytest.approx(0.003, abs=1e-9)


def test_synchronizer_out_of_tolerance_is_flagged_not_silently_dropped():
    sync = StreamSynchronizer(tolerance_ms=10.0)
    csi_buf = TimestampedBuffer()
    csi_buf.extend([(0.000, {"snr_db": 20.0})])
    gps_buf = TimestampedBuffer()
    gps_buf.extend([(0.050, {"timestamp": 0.050, "lat": 1.0})])

    sync.register_stream("csi", csi_buf, is_anchor=True)
    sync.register_stream("gps", gps_buf)

    results, violations = StreamSynchronizer.split_results(
        sync.synchronize(scene_id="SceneTest", vehicle_id=1)
    )

    assert len(results) == 0
    assert len(violations) == 1
    violation = violations[0]
    assert isinstance(violation, SyncViolation)
    assert violation.offending_stream == "gps"
    assert violation.offset_ms == pytest.approx(50.0, abs=1e-6)
    assert violation.tolerance_ms == 10.0


def test_synchronizer_requires_anchor_stream():
    sync = StreamSynchronizer(tolerance_ms=10.0)
    sync.register_stream("gps", TimestampedBuffer())
    with pytest.raises(RuntimeError):
        sync.synchronize("Scene", 1)


def test_synchronizer_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError):
        StreamSynchronizer(tolerance_ms=0.0)
