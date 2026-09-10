"""CRUD operations for Scene and Vehicle."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.scene import Scene, Vehicle
from app.schemas.scene import SceneCreate, VehicleCreate


def create_scene(db: Session, payload: SceneCreate) -> Scene:
    """Insert a new scene row and return it."""
    scene = Scene(**payload.model_dump())
    db.add(scene)
    db.commit()
    db.refresh(scene)
    return scene


def get_scene(db: Session, scene_id: uuid.UUID) -> Scene | None:
    """Fetch a single scene by id, or None if it doesn't exist."""
    return db.get(Scene, scene_id)


def list_scenes(db: Session, skip: int = 0, limit: int = 100) -> list[Scene]:
    """List scenes, most recently created first."""
    stmt = select(Scene).order_by(Scene.created_at.desc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def add_vehicle(db: Session, scene_id: uuid.UUID, payload: VehicleCreate) -> Vehicle | None:
    """Register a vehicle within a scene. Returns None if the scene doesn't exist."""
    scene = get_scene(db, scene_id)
    if scene is None:
        return None
    vehicle = Vehicle(scene_id=scene_id, **payload.model_dump())
    db.add(vehicle)
    db.commit()
    db.refresh(vehicle)
    return vehicle


def list_vehicles(db: Session, scene_id: uuid.UUID) -> list[Vehicle]:
    """List all vehicles registered within a scene."""
    stmt = select(Vehicle).where(Vehicle.scene_id == scene_id).order_by(Vehicle.vehicle_code)
    return list(db.execute(stmt).scalars().all())
