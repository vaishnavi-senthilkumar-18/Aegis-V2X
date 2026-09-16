"""Regression tests for simulation.sionna_configs.wireless_dataset_generator
(Stage B Step 6).

Uses the same minimal synthetic unit-cube mesh pattern as
test_channel_simulator.py -- portable/CI-safe, not real CARLA geometry (the
real-CARLA-mesh pilot run was done separately, against assets that live
outside this repository).
"""

import json
import math
import struct

import numpy as np
import pytest

mi = pytest.importorskip("mitsuba", reason="mitsuba not installed in this environment")
mi.set_variant("llvm_ad_mono_polarized")  # must precede `import sionna.rt`
sionna_rt = pytest.importorskip("sionna.rt", reason="sionna (Sionna RT) not installed in this environment")

from simulation.sionna_configs.channel_simulator import DEFAULT_RADIO_PARAMS, ChannelSimulator  # noqa: E402
from simulation.sionna_configs.wireless_dataset_generator import (  # noqa: E402
    CheckpointManifest,
    SampleOutcome,
    SampleSpec,
    WirelessDatasetGenerator,
)

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
def scene_with_cube(tmp_path):
    ply_path = tmp_path / "unit_cube.ply"
    _write_binary_cube_ply(ply_path)
    mesh = sionna_rt.load_mesh(str(ply_path))
    material = sionna_rt.ITURadioMaterial(name="itu_concrete_test_cube", itu_type="concrete")
    scene = sionna_rt.load_scene(None)
    obj = sionna_rt.SceneObject(mi_mesh=mesh, name="test_cube", radio_material=material)
    scene.edit(add=obj)
    return scene


@pytest.fixture()
def simulator():
    return ChannelSimulator(dict(DEFAULT_RADIO_PARAMS))


def _clear_spec(sample_id="s0", frame=0):
    return SampleSpec(
        sample_id=sample_id, frame=frame, wireless_timestamp=0.0,
        rsu_positions={0: (20.0, 5.0, 2.0)}, vehicle_positions={0: (-20.0, 5.0, 1.5)},
        condition_label="los_test",
    )


# --------------------------------------------------------------------------
# CheckpointManifest
# --------------------------------------------------------------------------

def test_checkpoint_manifest_starts_empty(tmp_path):
    manifest = CheckpointManifest(tmp_path / "checkpoint.json")
    assert not manifest.is_done("anything")
    assert manifest.summary()["total_recorded"] == 0


def test_checkpoint_manifest_persists_across_instances(tmp_path):
    path = tmp_path / "checkpoint.json"
    m1 = CheckpointManifest(path)
    m1.mark_done(SampleOutcome(sample_id="s0", status="completed", num_links=3))
    assert path.exists()

    m2 = CheckpointManifest(path)  # fresh instance, same file
    assert m2.is_done("s0")
    assert m2.summary()["by_status"]["completed"] == 1


def test_checkpoint_manifest_write_is_valid_json_every_time(tmp_path):
    path = tmp_path / "checkpoint.json"
    manifest = CheckpointManifest(path)
    for i in range(5):
        manifest.mark_done(SampleOutcome(sample_id=f"s{i}", status="completed"))
        with open(path) as fh:
            json.load(fh)  # must not raise -- atomic write must never leave a torn file


# --------------------------------------------------------------------------
# WirelessDatasetGenerator -- end-to-end against the actual class (not a
# parallel reimplementation)
# --------------------------------------------------------------------------

def test_generate_produces_expected_output_files(scene_with_cube, simulator, tmp_path):
    gen = WirelessDatasetGenerator(scene_with_cube, simulator, output_dir=tmp_path / "out")
    outcomes = gen.generate([_clear_spec("s0")])
    assert len(outcomes) == 1
    assert outcomes[0].status == "completed"
    assert outcomes[0].num_links == 1

    sample_dir = tmp_path / "out" / "s0"
    npz_files = list(sample_dir.glob("link_*.npz"))
    json_files = list(sample_dir.glob("link_*.json"))
    assert len(npz_files) == 1
    assert len(json_files) == 1

    with np.load(npz_files[0]) as data:
        csi = data["csi"]
        num_rx_ant = DEFAULT_RADIO_PARAMS["rx_array"]["rows"] * DEFAULT_RADIO_PARAMS["rx_array"]["cols"]
        num_tx_ant = DEFAULT_RADIO_PARAMS["tx_array"]["rows"] * DEFAULT_RADIO_PARAMS["tx_array"]["cols"]
        num_sc = DEFAULT_RADIO_PARAMS["num_subcarriers"]
        assert csi.shape == (num_rx_ant, num_tx_ant, num_sc)

    with open(json_files[0]) as fh:
        metadata = json.load(fh)
    assert metadata["los"] is True
    dist = math.dist((20.0, 5.0, 2.0), (-20.0, 5.0, 1.5))
    assert abs(metadata["propagation_delay_s"] - dist / 299_792_458.0) < 1e-9


def test_generate_is_resumable_and_skips_completed_samples(scene_with_cube, simulator, tmp_path):
    out_dir = tmp_path / "out"
    gen1 = WirelessDatasetGenerator(scene_with_cube, simulator, output_dir=out_dir)
    outcomes1 = gen1.generate([_clear_spec("s0"), _clear_spec("s1", frame=1)])
    assert len(outcomes1) == 2
    assert gen1.checkpoint.summary()["total_recorded"] == 2

    # New generator instance (simulates a fresh process after interruption),
    # same output_dir/checkpoint file, same sample IDs plus one new one.
    gen2 = WirelessDatasetGenerator(scene_with_cube, simulator, output_dir=out_dir)
    outcomes2 = gen2.generate(
        [_clear_spec("s0"), _clear_spec("s1", frame=1), _clear_spec("s2", frame=2)], resume=True
    )
    # only the genuinely new sample should have been (re-)generated
    assert len(outcomes2) == 1
    assert outcomes2[0].sample_id == "s2"
    assert gen2.checkpoint.summary()["total_recorded"] == 3


def test_generate_without_resume_reprocesses_everything(scene_with_cube, simulator, tmp_path):
    out_dir = tmp_path / "out"
    gen = WirelessDatasetGenerator(scene_with_cube, simulator, output_dir=out_dir)
    gen.generate([_clear_spec("s0")])
    outcomes = gen.generate([_clear_spec("s0")], resume=False)
    assert len(outcomes) == 1  # reprocessed, not skipped


def test_generate_records_failure_without_crashing_the_run(scene_with_cube, tmp_path):
    class _AlwaysFailsSimulator:
        def simulate_frame(self, *args, **kwargs):
            raise RuntimeError("synthetic induced failure")

    gen = WirelessDatasetGenerator(scene_with_cube, _AlwaysFailsSimulator(), output_dir=tmp_path / "out",
                                    max_retries=1, retry_backoff_s=0.0)
    outcomes = gen.generate([_clear_spec("s0"), _clear_spec("s1", frame=1)])
    assert len(outcomes) == 2
    assert all(o.status == "failed" for o in outcomes)
    assert all(o.attempts == 2 for o in outcomes)  # 1 initial + 1 retry
    assert all("synthetic induced failure" in o.error for o in outcomes)
    # failures are still checkpointed (not re-attempted forever on resume)
    assert gen.checkpoint.is_done("s0")


def test_generate_no_links_is_recorded_not_silently_dropped(scene_with_cube, simulator, tmp_path):
    spec = SampleSpec(sample_id="empty", frame=0, wireless_timestamp=0.0,
                       rsu_positions={}, vehicle_positions={}, condition_label="degenerate")
    gen = WirelessDatasetGenerator(scene_with_cube, simulator, output_dir=tmp_path / "out")
    outcomes = gen.generate([spec])
    assert len(outcomes) == 1
    assert outcomes[0].status == "skipped_no_links"
    assert gen.checkpoint.is_done("empty")


def test_generate_condition_label_is_preserved_in_outcome(scene_with_cube, simulator, tmp_path):
    spec = _clear_spec("s0")
    gen = WirelessDatasetGenerator(scene_with_cube, simulator, output_dir=tmp_path / "out")
    outcomes = gen.generate([spec])
    assert outcomes[0].condition_label == "los_test"
