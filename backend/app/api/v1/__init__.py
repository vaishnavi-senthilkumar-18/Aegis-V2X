"""Aggregates every v1 router into a single APIRouter for `app.main` to include."""

from fastapi import APIRouter

from app.api.v1 import criticality, decisions, experiments, frames, health, scenes, synthetic, trust

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(scenes.router)
api_v1_router.include_router(frames.router)
api_v1_router.include_router(trust.router)
api_v1_router.include_router(criticality.router)
api_v1_router.include_router(decisions.router)
api_v1_router.include_router(experiments.router)
api_v1_router.include_router(synthetic.router)
