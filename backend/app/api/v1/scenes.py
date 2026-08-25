"""Scene and Vehicle endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import require_api_key
from app.crud import scene as scene_crud
from app.schemas.scene import SceneCreate, SceneRead, VehicleCreate, VehicleRead

router = APIRouter(prefix="/scenes", tags=["scenes"])


@router.post(
    "", response_model=SceneRead, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def create_scene(payload: SceneCreate, db: Session = Depends(get_db)) -> SceneRead:
    """Create a new scene (simulation episode)."""
    scene = scene_crud.create_scene(db, payload)
    return SceneRead.model_validate(scene)


@router.get("", response_model=list[SceneRead])
def list_scenes(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
) -> list[SceneRead]:
    """List scenes, most recently created first."""
    scenes = scene_crud.list_scenes(db, skip=skip, limit=limit)
    return [SceneRead.model_validate(s) for s in scenes]


@router.get("/{scene_id}", response_model=SceneRead)
def get_scene(scene_id: uuid.UUID, db: Session = Depends(get_db)) -> SceneRead:
    """Fetch a single scene by id."""
    scene = scene_crud.get_scene(db, scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return SceneRead.model_validate(scene)


@router.post(
    "/{scene_id}/vehicles", response_model=VehicleRead, status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def add_vehicle(
    scene_id: uuid.UUID, payload: VehicleCreate, db: Session = Depends(get_db)
) -> VehicleRead:
    """Register a vehicle within an existing scene."""
    vehicle = scene_crud.add_vehicle(db, scene_id, payload)
    if vehicle is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return VehicleRead.model_validate(vehicle)


@router.get("/{scene_id}/vehicles", response_model=list[VehicleRead])
def list_vehicles(scene_id: uuid.UUID, db: Session = Depends(get_db)) -> list[VehicleRead]:
    """List all vehicles registered within a scene."""
    scene = scene_crud.get_scene(db, scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    vehicles = scene_crud.list_vehicles(db, scene_id)
    return [VehicleRead.model_validate(v) for v in vehicles]
