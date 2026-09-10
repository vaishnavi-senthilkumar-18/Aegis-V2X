"""Minimal API-key gate for write endpoints.

Scope (deliberately minimal for a research prototype — see
`docs/backend_api_documentation.md` and `claude/project_status.md`'s
"Open architecture questions" item 3): the backend was built with no
authentication at all, which was fine for purely local development but
became a real gap once the local backend was exposed publicly through a
VS Code Dev Tunnel so Lovable's cloud-hosted dashboard preview could
reach it — anyone with that tunnel URL could write data with no login
required.

This adds a single shared API key checked via the `X-API-Key` header on
write endpoints only (POST/PATCH). Read endpoints (GET) stay open, since
the dashboard's read-only polling isn't the risk this closes and forcing
auth on every read would break the existing unauthenticated dashboard
without a larger session/token design that's out of scope here.

When `settings.api_key` is unset (the default — no `API_KEY` in the
environment or `.env`), `require_api_key` is a no-op, so today's
unauthenticated local-dev workflow keeps working unchanged. Setting
`API_KEY` turns the gate on for every write endpoint at once.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.core.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """FastAPI dependency: reject write requests missing/mismatching `X-API-Key`.

    No-op when `settings.api_key` is unset. Raises 401 when it is set and
    the request's `X-API-Key` header doesn't match.
    """
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )
