"""Frame ORM model — one row per synchronized multimodal observation.

This is the core table of the schema. It maps to two specification
sections simultaneously (see `docs/backend_api_documentation.md` §3 for
the full field-by-field cross-reference):

* Dataset Design & Annotation Guide, Ch. 10 (Dataset Schema) — the raw
  sensor/comm fields (LiDAR path, GPS, IMU, CSI, SNR, RSSI, beam index...).
* Mathematical Formulation, Sec. 4 (Digital Twin state) — `M_t` (mobility
  state), `E_t` (environmental context), `U_t` (prediction uncertainty),
  `Age_t` (twin synchronization age).

`position_x/y/z` and `lane_id` were added in migration `6c25aca03f6f`
(Phase 3 dashboard rebuild) as CARLA-native local intersection
coordinates — distinct from `gps_lat/lon`, which are too coarse for the
intersection-scale SVG radar map on the Digital Twin dashboard page.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Frame(Base):
    """A single synchronized multimodal observation for one vehicle.

    Attributes:
        id: Primary key (UUID).
        scene_id: Owning scene.
        vehicle_id: Vehicle this observation belongs to.
        frame_index: Monotonic per-scene sample index (Dataset Schema "Sample/Frame_ID").
        simulation_timestamp: CARLA simulation clock time (seconds), maps to `t`.
        wireless_timestamp: Sionna RT wireless-side clock time (seconds).
        sync_timestamp: Reconciled timestamp used for cross-modal alignment.
        sync_offset_ms: |simulation_timestamp - wireless_timestamp| * 1000.
        is_sync_valid: Whether `sync_offset_ms` is within the Ch. 9 tolerance (<=10ms).
        lidar_path: Filesystem/object-store reference to the LiDAR point cloud (not inline).
        gps_lat/gps_lon/gps_alt: GPS fix, part of `M_t`.
        position_x/y/z: CARLA-native local intersection coordinates (meters).
        lane_id: CARLA lane identifier the vehicle occupies at this frame.
        imu_data: IMU reading (JSON), part of `M_t`.
        speed_mps: Vehicle speed, part of `M_t`.
        csi: Channel State Information (JSON), `CSI_t`.
        snr_db: Signal-to-noise ratio, `SNR_t`.
        rssi_dbm: Received signal strength indicator.
        path_loss_db: Path loss.
        beam_index: Selected beam index, `B_t`.
        traffic_density: Local traffic density, part of `E_t`.
        weather: Weather condition label, part of `E_t`.
        mobility_state: Full mobility-state vector (JSON), `M_t`.
        environmental_context: Full environmental-context vector (JSON), `E_t`.
        prediction_uncertainty: Scalar uncertainty estimate, `U_t`.
        twin_age_ms: Digital Twin synchronization age in milliseconds, `Age_t`.
        gt_future_csi: Ground-truth future CSI (JSON), for supervised channel prediction.
        gt_future_beam: Ground-truth future beam index, for supervised beam prediction.
        source: "synthetic" (Phase 3 stand-in generator) or "carla_sionna" (real Phase 2 pipeline).
        created_at: Row creation timestamp.
    """

    __tablename__ = "frames"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vehicle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vehicles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)

    simulation_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    wireless_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    sync_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    sync_offset_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_sync_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    lidar_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    gps_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_alt: Mapped[float | None] = mapped_column(Float, nullable=True)

    # CARLA-native local intersection coordinates (migration 6c25aca03f6f).
    position_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    lane_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    imu_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    speed_mps: Mapped[float | None] = mapped_column(Float, nullable=True)

    csi: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    snr_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    rssi_dbm: Mapped[float | None] = mapped_column(Float, nullable=True)
    path_loss_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    beam_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    traffic_density: Mapped[float | None] = mapped_column(Float, nullable=True)
    weather: Mapped[str | None] = mapped_column(String(64), nullable=True)

    mobility_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    environmental_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    prediction_uncertainty: Mapped[float | None] = mapped_column(Float, nullable=True)
    twin_age_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    gt_future_csi: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    gt_future_beam: Mapped[int | None] = mapped_column(Integer, nullable=True)

    source: Mapped[str] = mapped_column(String(32), nullable=False, default="synthetic")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    scene: Mapped[Scene] = relationship(back_populates="frames")  # noqa: F821
    vehicle: Mapped[Vehicle] = relationship(back_populates="frames")  # noqa: F821
    trust_record: Mapped[TrustRecord | None] = relationship(  # noqa: F821
        back_populates="frame", uselist=False, cascade="all, delete-orphan"
    )
    criticality_record: Mapped[CriticalityRecord | None] = relationship(  # noqa: F821
        back_populates="frame", uselist=False, cascade="all, delete-orphan"
    )
    decision: Mapped[Decision | None] = relationship(  # noqa: F821
        back_populates="frame", uselist=False, cascade="all, delete-orphan"
    )
