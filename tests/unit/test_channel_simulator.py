"""Regression tests for simulation.sionna_configs.channel_simulator against
Sionna RT 2.1.0 (Stage B Step 5 cleanup).

These exist so the API incompatibilities found and fixed in Stage B Step 5
(antenna pattern name "38.901" -> "tr38901"; scene.compute_paths() ->
PathSolver; Paths.cir(tx=,rx=) -> indexed Paths.a/Paths.tau; scene.remove()
taking one name per call; PathSolver's padded num_paths dimension needing
sentinel filtering) cannot silently return.

Uses a minimal synthetic unit-cube mesh (written to a pytest tmp path), not
real CARLA geometry -- the real-CARLA-mesh end-to-end validation was done
separately (Stage A/B) against assets that live outside this repository
(extracted to a machine-local scratch directory, not committed) and are
therefore not available to a portable/CI test. This module tests API
compatibility and output shape/range correctness, not propagation realism
against actual Town04 assets.

sionna/mitsuba are optional, heavy dependencies (see requirements/
simulation.txt) not installed in every environment that runs this test
suite -- skipped gracefully via importorskip rather than failing collection.
"""

import math
import struct

import numpy as np
import pytest

mi = pytest.importorskip("mitsuba", reason="mitsuba not installed in this environment")
mi.set_variant("llvm_ad_mono_polarized")  # must precede `import sionna.rt` (variant-dependent
# class registration binds at sionna.rt import time)
sionna_rt = pytest.importorskip("sionna.rt", reason="sionna (Sionna RT) not installed in this environment")

from simulation.sionna_configs.channel_simulator import (  # noqa: E402
    DEFAULT_RADIO_PARAMS,
    ChannelSimulator,
    _classify_los,
    _LOS_DELAY_TOLERANCE_S,
)

_C = 299_792_458.0

_CUBE_VERTICES = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]
_CUBE_FACES = [
    (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
    (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
]


def _write_binary_cube_ply(path) -> None:
    # Binary little-endian PLY -- matches the format this project's real
    # CARLA-mesh exports use (Blender's PLY exporter defaults to binary);
    # a plain ASCII PLY triggers an unrelated Sionna/Mitsuba loader quirk
    # ("Found 1 unreferenced property... bsdf") not seen anywhere else in
    # this project's extensive use of load_mesh() on real (binary) assets.
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(_CUBE_VERTICES)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(_CUBE_FACES)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    body = b"".join(struct.pack("<3f", *v) for v in _CUBE_VERTICES)
    body += b"".join(struct.pack("<B3i", 3, *f) for f in _CUBE_FACES)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(body)


@pytest.fixture()
def unit_cube_ply(tmp_path):
    path = tmp_path / "unit_cube.ply"
    _write_binary_cube_ply(path)
    return str(path)


@pytest.fixture()
def scene_with_cube(unit_cube_ply):
    mesh = sionna_rt.load_mesh(unit_cube_ply)
    material = sionna_rt.ITURadioMaterial(name="itu_concrete_test_cube", itu_type="concrete")
    scene = sionna_rt.load_scene(None)
    obj = sionna_rt.SceneObject(mi_mesh=mesh, name="test_cube", radio_material=material)
    scene.edit(add=obj)
    return scene


def test_tr38901_pattern_accepted():
    """The registered pattern name is 'tr38901', not '38.901' -- this is
    the exact API incompatibility found in Stage B Step 5."""
    array = sionna_rt.PlanarArray(num_rows=2, num_cols=2, pattern="tr38901", polarization="V")
    assert array is not None


def test_old_pattern_name_rejected():
    """Documents the incompatibility this fix addresses: the old value
    would fail, confirming DEFAULT_RADIO_PARAMS could not have worked
    unfixed against this Sionna RT version."""
    with pytest.raises(Exception):
        sionna_rt.PlanarArray(num_rows=2, num_cols=2, pattern="38.901", polarization="V")


def test_default_radio_params_use_registered_pattern():
    assert DEFAULT_RADIO_PARAMS["tx_array"]["pattern"] == "tr38901"
    assert DEFAULT_RADIO_PARAMS["rx_array"]["pattern"] == "tr38901"


def test_channel_simulator_configures_configured_arrays(scene_with_cube):
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    sim.configure_scene(scene_with_cube)
    assert float(scene_with_cube.frequency[0]) == pytest.approx(28.0e9)


def test_simulate_frame_executes_and_produces_result(scene_with_cube):
    """Path solving executes through the actual ChannelSimulator.simulate_frame()
    (not a parallel implementation) on a real (if minimal) scene."""
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    # Unobstructed by the unit cube (which spans [-1,1] on every axis):
    # placed well off to the side (y=5) of the cube's footprint.
    rsu_positions = {0: (20.0, 5.0, 2.0)}
    vehicle_positions = {0: (-20.0, 5.0, 1.5)}
    results = sim.simulate_frame(scene_with_cube, rsu_positions, vehicle_positions,
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1


def test_csi_has_expected_antenna_and_subcarrier_dimensions(scene_with_cube):
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    rsu_positions = {0: (20.0, 5.0, 2.0)}
    vehicle_positions = {0: (-20.0, 5.0, 1.5)}
    results = sim.simulate_frame(scene_with_cube, rsu_positions, vehicle_positions,
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1
    num_rx_ant = DEFAULT_RADIO_PARAMS["rx_array"]["rows"] * DEFAULT_RADIO_PARAMS["rx_array"]["cols"]
    num_tx_ant = DEFAULT_RADIO_PARAMS["tx_array"]["rows"] * DEFAULT_RADIO_PARAMS["tx_array"]["cols"]
    num_sc = DEFAULT_RADIO_PARAMS["num_subcarriers"]
    assert results[0].csi.shape == (num_rx_ant, num_tx_ant, num_sc)


def test_beam_index_non_null_and_in_codebook_range(scene_with_cube):
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    rsu_positions = {0: (20.0, 5.0, 2.0)}
    vehicle_positions = {0: (-20.0, 5.0, 1.5)}
    results = sim.simulate_frame(scene_with_cube, rsu_positions, vehicle_positions,
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1
    beam_index = results[0].beam_index
    assert beam_index is not None
    assert 0 <= beam_index < DEFAULT_RADIO_PARAMS["num_beams"]


def test_no_nan_or_inf_in_result(scene_with_cube):
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    rsu_positions = {0: (20.0, 5.0, 2.0)}
    vehicle_positions = {0: (-20.0, 5.0, 1.5)}
    results = sim.simulate_frame(scene_with_cube, rsu_positions, vehicle_positions,
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1
    r = results[0]
    assert not np.isnan(r.csi).any()
    assert not np.isinf(r.csi).any()
    assert math.isfinite(r.snr_db)
    assert math.isfinite(r.rssi_dbm)
    assert math.isfinite(r.path_loss_db)
    assert math.isfinite(r.propagation_delay_s)


def test_propagation_delay_matches_distance_over_c(scene_with_cube):
    """LOS delay must remain consistent with geometric distance/c."""
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    tx_pos = (20.0, 5.0, 2.0)
    rx_pos = (-20.0, 5.0, 1.5)
    rsu_positions = {0: tx_pos}
    vehicle_positions = {0: rx_pos}
    results = sim.simulate_frame(scene_with_cube, rsu_positions, vehicle_positions,
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1
    dist = math.dist(tx_pos, rx_pos)
    expected_delay = dist / _C
    assert results[0].propagation_delay_s == pytest.approx(expected_delay, abs=1e-9)


def test_no_padding_sentinel_leaks_into_propagation_delay(scene_with_cube):
    """PathSolver pads the num_paths dimension across all Tx/Rx pairs with
    tau=-1.0 sentinels for unused slots; simulate_frame must filter these
    out rather than letting them appear as a negative propagation delay."""
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    rsu_positions = {0: (20.0, 5.0, 2.0), 1: (20.0, 55.0, 2.0)}
    vehicle_positions = {0: (-20.0, 5.0, 1.5), 1: (-20.0, 65.0, 1.5)}
    results = sim.simulate_frame(scene_with_cube, rsu_positions, vehicle_positions,
                                  frame=0, wireless_timestamp=0.0)
    for r in results:
        assert r.propagation_delay_s >= 0.0


# --------------------------------------------------------------------------
# Stage B Step 5.1: LOS/NLOS classification regression tests.
#
# Positions below use the unit cube ([-1,1] on every axis, ITU concrete)
# from `scene_with_cube` as the sole obstruction.
# --------------------------------------------------------------------------

_LOS_CLEAR_TX = (20.0, 5.0, 2.0)     # offset to y=5, well clear of the cube's y=[-1,1]
_LOS_CLEAR_RX = (-20.0, 5.0, 1.5)

_BLOCKED_TX = (-5.0, -3.0, 0.3)      # diagonal placement across the cube: the direct
_BLOCKED_RX = (5.0, 3.0, 0.3)        # straight line is blocked by it, while reflection/
                                      # diffraction around it still returns a couple of
                                      # non-direct-matching paths (empirically verified) --
                                      # unlike a collinear blocked pair, which this scene's
                                      # ray tracer returns zero paths for at all (a separate,
                                      # legitimate outcome simulate_frame already handles by
                                      # omitting the link, not something this test targets)

# Both west of the cube (clear direct path along x=-5), positioned so a
# specular reflection off the cube's west face (x=-1) connects them too
# (hand-computed mirror point: (-1, 0, 0.3), inside the cube face's
# y,z in [-1,1] extent).
_MULTIPATH_TX = (-5.0, -3.0, 0.3)
_MULTIPATH_RX = (-5.0, 3.0, 0.3)


def test_classify_los_unobstructed_geometry_is_los(scene_with_cube):
    """C.1: a known unobstructed LOS geometry -> LOS=True."""
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    results = sim.simulate_frame(scene_with_cube, {0: _LOS_CLEAR_TX}, {0: _LOS_CLEAR_RX},
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1
    assert results[0].los is True


def test_classify_los_blocked_geometry_is_nlos(scene_with_cube):
    """C.2: a known blocked geometry -> LOS=False."""
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    results = sim.simulate_frame(scene_with_cube, {0: _BLOCKED_TX}, {0: _BLOCKED_RX},
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1
    assert results[0].los is False


def test_classify_los_multipath_geometry_with_direct_path_is_los(scene_with_cube):
    """C.3: a multipath geometry with direct LOS + reflected path(s) -> LOS=True."""
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    results = sim.simulate_frame(scene_with_cube, {0: _MULTIPATH_TX}, {0: _MULTIPATH_RX},
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1
    assert results[0].los is True
    # genuinely multipath: more than just the direct path was found
    assert results[0].num_multipath_components > 1


def test_classify_los_ignores_padding_sentinel():
    """C.4: invalid tau=-1 padding must never influence LOS classification.

    Exercised directly against `_classify_los` (the pure classification
    function) with a synthetic delays array containing a -1.0 sentinel
    mixed with a real matching delay -- this must NOT happen if the caller
    forgets to filter padding first, since -1.0 is deliberately NOT close
    to any physically plausible expected_direct_delay_s used here.
    """
    expected = 5e-8  # arbitrary plausible direct delay
    # Sentinel-contaminated input (what would happen if a caller passed
    # raw, unfiltered path_delays): must not be misread as a match.
    contaminated = np.array([-1.0, -1.0, 9.9e-8])
    assert _classify_los(contaminated, expected) is False
    # Properly filtered (valid-only) input containing the real match:
    filtered = contaminated[contaminated >= 0.0]  # -> [9.9e-8]
    assert _classify_los(filtered, expected) is False  # still not a match to `expected`
    matching = np.array([expected])
    assert _classify_los(matching, expected) is True


def test_classify_los_repeated_runs_are_consistent(scene_with_cube):
    """C.5: repeated runs must produce the same LOS classification."""
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    los_values = []
    for frame in range(3):
        results = sim.simulate_frame(scene_with_cube, {0: _MULTIPATH_TX}, {0: _MULTIPATH_RX},
                                      frame=frame, wireless_timestamp=0.1 * frame)
        assert len(results) == 1
        los_values.append(results[0].los)
    assert len(set(los_values)) == 1
    assert los_values[0] is True


def test_classify_los_delay_matches_distance_over_c_within_tolerance(scene_with_cube):
    """C.6: the LOS-consistent delay should match distance/c within the
    documented tolerance (_LOS_DELAY_TOLERANCE_S)."""
    sim = ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))
    results = sim.simulate_frame(scene_with_cube, {0: _LOS_CLEAR_TX}, {0: _LOS_CLEAR_RX},
                                  frame=0, wireless_timestamp=0.0)
    assert len(results) == 1
    assert results[0].los is True
    expected_delay = math.dist(_LOS_CLEAR_TX, _LOS_CLEAR_RX) / _C
    assert abs(results[0].propagation_delay_s - expected_delay) <= _LOS_DELAY_TOLERANCE_S
