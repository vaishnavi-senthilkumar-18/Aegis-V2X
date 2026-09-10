"""Aegis-V2X backend application entrypoint.

Wires together the FastAPI app, CORS, Prometheus instrumentation, the
versioned API router, and (once built) the React dashboard SPA static
mount at `/dashboard/`.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import FileResponse
from starlette.types import Scope

from app.api.v1 import api_v1_router
from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging()

app = FastAPI(
    title=settings.project_name,
    description=(
        "Aegis-V2X Phase 3 backend: serves the Digital Twin data model "
        "(scenes, vehicles, frames, trust/criticality records, TwinTrust-AP "
        "decisions, experiments) over a versioned REST API, and hosts the "
        "React research console dashboard."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


class SPAStaticFiles(StaticFiles):
    """Static file server that falls back to `index.html` for client-side routes.

    `react-router-dom` routes such as `/dashboard/analytics` only exist in
    the browser's history API — there is no `analytics` file on disk.
    Starlette's `StaticFiles.get_response` doesn't return a 404 response
    for a missing path, it *raises* `starlette.exceptions.HTTPException(404)`
    — the first version of this class checked `response.status_code == 404`
    on the return value, which never triggered, and every direct navigation
    or hard refresh to a client-side route 404'd. The second version caught
    `fastapi.HTTPException`, which still didn't work: `fastapi.HTTPException`
    is a *subclass* of `starlette.exceptions.HTTPException`, and Starlette
    raises the base class directly, so `except fastapi.HTTPException` never
    matched. Catching `starlette.exceptions.HTTPException` (for any
    extension-less path, so real missing assets still 404 normally) and
    falling back to `index.html` was the actual fix, found during the
    Phase 3 dashboard rebuild.
    """

    async def get_response(self, path: str, scope: Scope) -> FileResponse:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and "." not in Path(path).name:
                return await super().get_response("index.html", scope)
            raise


_dashboard_dist = Path(__file__).resolve().parent.parent.parent / "dashboard" / "dist"
if os.environ.get("AEGIS_SKIP_DASHBOARD_MOUNT") != "1" and _dashboard_dist.exists():
    app.mount(
        "/dashboard",
        SPAStaticFiles(directory=str(_dashboard_dist), html=True),
        name="dashboard",
    )
