"""Shared pytest fixtures for Aegis-V2X tests."""

import pytest

from digital_twin.state import ChannelState, DigitalTwinState, EnvironmentalContext, MobilityState


@pytest.fixture
def sample_dt_state() -> DigitalTwinState:
    """A representative DigitalTwinState for use across unit tests."""
    return DigitalTwinState(
        timestamp=10.0,
        channel=ChannelState(csi=None, snr=20.0, beam_index=3, path_loss=90.0),
        mobility=MobilityState(
            position=(0.0, 0.0, 0.0),
            velocity=(15.0, 0.0, 0.0),
            heading=0.0,
            relative_speed=5.0,
        ),
        environment=EnvironmentalContext(
            weather="clear_day", traffic_density="sparse", blockage_probability=0.05
        ),
        prediction_uncertainty=0.1,
        sync_age_seconds=0.02,
        metadata={"scene_id": "scene_01", "vehicle_id": "veh_01", "frame_id": 42},
    )
