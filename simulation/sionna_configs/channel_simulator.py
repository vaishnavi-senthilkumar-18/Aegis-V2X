"""Ray-traced wireless channel simulation between roadside units (Tx) and
vehicles (Rx) using Sionna RT.

Which quantities to compute is driven by configs/simulation.yaml's
`sionna_rt.outputs` list (frozen, Phase 1): csi, snr, rssi, path_loss,
delay_spread, beam_index, los_nlos, multipath_components, reflection_paths,
propagation_delay. Radio/antenna parameters (carrier frequency, array size,
codebook size, ray-tracing depth) are not specified in that file and are
documented internal defaults below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import mitsuba as mi
    import drjit as dr
    from sionna.rt import PathSolver, PlanarArray, Receiver, Scene, Transmitter
except ImportError as exc:  # pragma: no cover
    raise ImportError("sionna_configs.channel_simulator requires the 'sionna' package (Sionna RT).") from exc

logger = logging.getLogger("aegis_v2x.simulation.sionna_configs.channel_simulator")

_BOLTZMANN_DBM_HZ_K = -228.6
_REFERENCE_TEMP_K = 290.0
_SPEED_OF_LIGHT_M_S = 299_792_458.0

# Stage B Step 5.1: tolerance for matching a ray-traced path's delay against
# the geometric direct-path delay (distance / c) when classifying LOS.
#
# 1 nanosecond was chosen, not the previous ad hoc "min <= mean * 0.5"
# heuristic (removed -- see _classify_los), because:
#   - Every direct-path delay match observed across this project's real
#     52-mesh scene validation (Stage A/B, Stage B Step 5) agreed with
#     distance/c to within 1e-13..1e-15 s -- float64 numpy precision on top
#     of the ray tracer's own geometry, several orders of magnitude tighter
#     than 1ns. 1ns is a comfortable, non-brittle margin above that noise
#     floor, not a value tuned to make any particular case pass.
#   - At this project's configured bandwidth (100 MHz, DEFAULT_RADIO_PARAMS),
#     the channel's own delay/sample resolution is 1/100e6 = 10 ns -- so
#     resolving LOS timing to 1 ns is already an order of magnitude finer
#     than the system can distinguish, and there is no physical basis to
#     demand tighter agreement than that.
_LOS_DELAY_TOLERANCE_S = 1e-9

# Documented internal defaults (configs/simulation.yaml only specifies
# ray_tracing: true and the outputs list, not these physical-layer params).
#
# NOTE (Stage B Step 5 validation, Sionna RT 2.1.0): the antenna pattern name
# was corrected from "38.901" to "tr38901" -- Sionna RT 2.1.0's
# antenna_pattern_registry only recognizes ['iso', 'dipole', 'hw_dipole',
# 'tr38901'] (confirmed via sionna.rt.antenna_pattern.antenna_pattern_registry
# .list()); "38.901" is not a registered name and would fail at PlanarArray
# construction. This is the only change made to this dict.
DEFAULT_RADIO_PARAMS = {
    "carrier_frequency_hz": 28.0e9, "bandwidth_hz": 100.0e6, "num_subcarriers": 128,
    "tx_array": {"rows": 8, "cols": 8, "pattern": "tr38901", "polarization": "V"},
    "rx_array": {"rows": 4, "cols": 4, "pattern": "tr38901", "polarization": "V"},
    "num_beams": 64, "tx_power_dbm": 23.0, "noise_figure_db": 7.0,
    "ray_tracing": {"max_depth": 5, "method": "fibonacci", "num_samples": 1_000_000,
                    "los": True, "reflection": True, "diffraction": True,
                    "scattering": True, "scat_keep_prob": 0.001, "edge_diffraction": True},
}


def _classify_los(valid_delays: np.ndarray, expected_direct_delay_s: float,
                   tolerance_s: float = _LOS_DELAY_TOLERANCE_S) -> bool:
    """Physically grounded LOS/NLOS classification (Stage B Step 5.1).

    Replaces the old `min(delay) <= mean(delay) * 0.5` heuristic, which
    misclassified geometrically clear LOS links as NLOS once diffuse
    reflection/diffraction produced many additional paths that pulled the
    mean delay up.

    A link is LOS iff at least one *valid* ray-traced path's delay agrees,
    within ``tolerance_s``, with the geometric direct/free-space delay
    (Euclidean TX-RX distance / c) -- i.e. iff the ray tracer actually found
    the direct path, not merely because the shortest path happens to be
    much shorter than the average of everything else found.

    ``valid_delays`` must already have padding-sentinel entries (Sionna RT
    2.1.0's tau=-1.0 for unused slots in a solve batching multiple Tx/Rx
    pairs) removed by the caller -- this function does not know which
    entries are sentinels vs. genuine (if unlikely) zero/negative delays,
    so it deliberately does not re-filter here.
    """
    if valid_delays.size == 0:
        return False
    return bool(np.any(np.abs(valid_delays - expected_direct_delay_s) <= tolerance_s))


@dataclass
class ChannelSimulationResult:
    link_id: str
    vehicle_id: int
    frame: int
    wireless_timestamp: float
    csi: np.ndarray
    snr_db: float
    rssi_dbm: float
    path_loss_db: float
    delay_spread_s: float
    propagation_delay_s: float
    beam_index: int
    los: bool
    num_multipath_components: int


class ChannelSimulator:
    """Runs Sionna RT ray tracing for a single frame's RSU-vehicle links."""

    def __init__(self, radio_params: dict = None):
        self._p = radio_params or DEFAULT_RADIO_PARAMS
        self._noise_power_dbm = self._compute_noise_power_dbm()
        self._beam_codebook = np.deg2rad(np.linspace(-60.0, 60.0, self._p["num_beams"]))

    def _compute_noise_power_dbm(self) -> float:
        bw_hz = float(self._p["bandwidth_hz"])
        nf_db = float(self._p["noise_figure_db"])
        thermal_noise_dbm = _BOLTZMANN_DBM_HZ_K + 10.0 * np.log10(_REFERENCE_TEMP_K) + 10.0 * np.log10(bw_hz)
        return thermal_noise_dbm + nf_db

    def configure_scene(self, scene: Scene) -> None:
        scene.tx_array = PlanarArray(num_rows=self._p["tx_array"]["rows"], num_cols=self._p["tx_array"]["cols"],
                                      pattern=self._p["tx_array"]["pattern"], polarization=self._p["tx_array"]["polarization"])
        scene.rx_array = PlanarArray(num_rows=self._p["rx_array"]["rows"], num_cols=self._p["rx_array"]["cols"],
                                      pattern=self._p["rx_array"]["pattern"], polarization=self._p["rx_array"]["polarization"])
        scene.frequency = float(self._p["carrier_frequency_hz"])

    def simulate_frame(self, scene: Scene, rsu_positions: Dict[int, Tuple[float, float, float]],
                        vehicle_positions: Dict[int, Tuple[float, float, float]],
                        frame: int, wireless_timestamp: float) -> List[ChannelSimulationResult]:
        # Sionna RT 2.1.0 compatibility (Stage B Step 5 cleanup):
        # scene.remove() takes one object/name per call, not a list.
        self.configure_scene(scene)
        for name in list(scene.transmitters.keys()) + list(scene.receivers.keys()):
            scene.remove(name)

        rsu_ids = list(rsu_positions.keys())
        vehicle_ids = list(vehicle_positions.keys())
        for rsu_id in rsu_ids:
            pos = rsu_positions[rsu_id]
            scene.add(Transmitter(name=f"tx_{rsu_id}", position=mi.Point3f(*[float(v) for v in pos]),
                                   power_dbm=float(self._p["tx_power_dbm"])))
        for vehicle_id in vehicle_ids:
            pos = vehicle_positions[vehicle_id]
            scene.add(Receiver(name=f"rx_{vehicle_id}", position=mi.Point3f(*[float(v) for v in pos])))

        if not rsu_ids or not vehicle_ids:
            return []

        # Sionna RT 2.1.0 compatibility: scene.compute_paths(...) no longer
        # exists; path solving is done via the PathSolver class. Parameter
        # mapping from the old ray_tracing dict: reflection->
        # specular_reflection, scattering->diffuse_reflection. "method",
        # "num_samples", and "scat_keep_prob" have no equivalent in the
        # current PathSolver signature and are not used. synthetic_array=True
        # matches every other validated scene in this project (Stage A/B).
        rt_params = self._p["ray_tracing"]
        solver = PathSolver()
        paths = solver(
            scene,
            max_depth=rt_params["max_depth"],
            los=rt_params.get("los", True),
            specular_reflection=rt_params.get("reflection", True),
            diffuse_reflection=rt_params.get("scattering", True),
            refraction=False,
            diffraction=rt_params.get("diffraction", True),
            edge_diffraction=rt_params.get("edge_diffraction", True),
            synthetic_array=True,
        )
        dr.eval(paths.a)
        dr.sync_thread()

        # Sionna RT 2.1.0 compatibility: Paths.cir(tx=, rx=) no longer
        # accepts tx/rx keyword arguments -- a single Paths object already
        # covers every configured Tx/Rx pair simultaneously, indexed by
        # insertion order into scene.transmitters/scene.receivers (both are
        # ordinary (ordered) dicts). a shape: [num_rx, num_rx_ant, num_tx,
        # num_tx_ant, num_paths]; tau shape: [num_rx, num_tx, num_paths]
        # (no antenna dimension -- confirmed empirically for this Sionna
        # RT version).
        a_re, a_im = paths.a
        a_re = np.asarray(a_re)
        a_im = np.asarray(a_im)
        tau = np.asarray(paths.tau)

        num_sc = int(self._p["num_subcarriers"])
        bw_hz = float(self._p["bandwidth_hz"])
        frequencies = np.linspace(-bw_hz / 2, bw_hz / 2, num_sc)

        results: List[ChannelSimulationResult] = []
        for tx_idx, rsu_id in enumerate(rsu_ids):
            for rx_idx, vehicle_id in enumerate(vehicle_ids):
                link_id = f"{rsu_id}_{vehicle_id}"
                a_pair = a_re[rx_idx, :, tx_idx, :, :] + 1j * a_im[rx_idx, :, tx_idx, :, :]
                tau_pair = tau[rx_idx, tx_idx, :]
                if a_pair.size == 0 or tau_pair.size == 0:
                    logger.debug("No propagation path for link %s at frame %d (blocked)", link_id, frame)
                    continue
                # Geometric direct-path delay for LOS classification
                # (Stage B Step 5.1) -- link-level (one TX reference
                # position, one RX reference position), independent of the
                # antenna/path array dimensions carried by a_pair/tau_pair.
                tx_xyz = np.asarray(rsu_positions[rsu_id], dtype=np.float64)
                rx_xyz = np.asarray(vehicle_positions[vehicle_id], dtype=np.float64)
                expected_delay_s = float(np.linalg.norm(rx_xyz - tx_xyz)) / _SPEED_OF_LIGHT_M_S
                results.append(self._to_result(link_id, vehicle_id, frame, wireless_timestamp,
                                                 a_pair, tau_pair, frequencies, expected_delay_s))
        return results

    def _to_result(self, link_id: str, vehicle_id: int, frame: int, wireless_timestamp: float,
                    path_gains: np.ndarray, path_delays: np.ndarray, frequencies: np.ndarray,
                    expected_direct_delay_s: float) -> ChannelSimulationResult:
        """path_gains: complex ndarray [num_rx_ant, num_tx_ant, num_paths].
        path_delays: real ndarray [num_paths]. (Previously 1D/scalar-link
        shapes, generalized here -- Stage B Step 5 cleanup -- to carry the
        real antenna dimensions Sionna RT 2.1.0 returns for a multi-element
        array; the CIR->CFR summation formula itself is unchanged.)
        """
        # Sionna RT 2.1.0 compatibility: PathSolver pads the num_paths
        # dimension to a common size across every Tx/Rx pair solved in the
        # same call; unused slots are marked with tau=-1.0 and zero
        # amplitude (confirmed empirically). Excluded from delay statistics
        # below so a padding sentinel can't be picked up by np.min()/
        # np.mean() as if it were a real (and impossibly negative) delay.
        valid = path_delays >= 0.0
        num_multipath_components = int(np.count_nonzero(valid))

        phase = np.exp(-2j * np.pi * frequencies[None, None, None, :] * path_delays[None, None, :, None])
        csi = np.sum(path_gains[:, :, :, None] * phase, axis=2)  # -> [num_rx_ant, num_tx_ant, num_sc]
        # (padded slots contribute exactly 0 to this sum since their
        # amplitude is 0, so no additional masking is needed here)

        received_power_lin = np.mean(np.abs(csi) ** 2)
        received_power_dbm = 10.0 * np.log10(max(received_power_lin, 1e-15)) + float(self._p["tx_power_dbm"])
        snr_db = received_power_dbm - self._noise_power_dbm
        path_loss_db = float(self._p["tx_power_dbm"]) - received_power_dbm

        valid_delays = path_delays[valid]
        path_power = np.mean(np.abs(path_gains) ** 2, axis=(0, 1))  # per-path power, avg over antennas -> [num_paths]
        weights = path_power[valid]
        mean_delay = float(np.average(valid_delays, weights=weights)) if weights.sum() > 0 else 0.0
        delay_spread = float(np.sqrt(np.average((valid_delays - mean_delay) ** 2, weights=weights))) if weights.sum() > 0 else 0.0

        # _select_best_beam() is unchanged (Stage B Step 5 reuse requirement).
        # It expects a vector indexed by antenna element (length must match
        # the num_beams codebook, which by construction equals num_tx_ant
        # for the project's default 8x8=64-element/64-beam configuration).
        # Built here as the per-TX-antenna channel, summed over paths and
        # averaged over RX antennas.
        tx_ant_vector = path_gains.sum(axis=2).mean(axis=0)  # -> [num_tx_ant]
        beam_index = self._select_best_beam(tx_ant_vector)
        los = _classify_los(valid_delays, expected_direct_delay_s)

        return ChannelSimulationResult(
            link_id=link_id, vehicle_id=vehicle_id, frame=frame, wireless_timestamp=wireless_timestamp,
            csi=csi.astype(np.complex64), snr_db=float(snr_db), rssi_dbm=float(received_power_dbm),
            path_loss_db=float(path_loss_db), delay_spread_s=delay_spread,
            propagation_delay_s=float(np.min(valid_delays)) if valid_delays.size else 0.0,
            beam_index=beam_index, los=los, num_multipath_components=num_multipath_components,
        )

    def _select_best_beam(self, csi: np.ndarray) -> int:
        num_sc = len(csi)
        best_idx, best_gain = 0, -np.inf
        for idx, angle_rad in enumerate(self._beam_codebook):
            steering = np.exp(1j * np.pi * np.sin(angle_rad) * np.arange(num_sc))
            gain = float(np.abs(np.vdot(steering, csi)) ** 2)
            if gain > best_gain:
                best_gain, best_idx = gain, idx
        return best_idx
