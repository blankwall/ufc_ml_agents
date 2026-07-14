"""
Ingest Router
=============
Endpoints backing the /ingest UI:
  - POST /api/ingest/preview   dry-run scrape a Sherdog fighter (no writes)
  - POST /api/ingest/save      commit fighter to DB + optionally write an alias
  - GET  /api/aliases          list all fighter aliases
  - POST /api/aliases          add/update a single alias
  - DELETE /api/aliases/{alias} remove an alias
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fastapi_app.services import fighter_alias_service
from fastapi_app.services.ingest_service import preview_fighter, save_fighter

router = APIRouter()


class IngestPreviewRequest(BaseModel):
    sherdog_url: str
    requested_name: Optional[str] = None


class IngestSaveRequest(BaseModel):
    sherdog_url: str
    requested_name: Optional[str] = None
    write_alias: bool = False
    alias_from: Optional[str] = None
    alias_to: Optional[str] = None
    bust_cache: bool = False


class AliasUpsertRequest(BaseModel):
    alias: str
    canonical: str


@router.post("/ingest/preview")
async def api_ingest_preview(body: IngestPreviewRequest):
    """Dry-run scrape a Sherdog fighter page. Writes nothing."""
    try:
        return preview_fighter(body.sherdog_url, body.requested_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {e}")


@router.post("/ingest/save")
async def api_ingest_save(body: IngestSaveRequest):
    """Commit the fighter to the DB and optionally persist a name alias."""
    try:
        return save_fighter(
            sherdog_url=body.sherdog_url,
            requested_name=body.requested_name,
            write_alias=body.write_alias,
            alias_from=body.alias_from,
            alias_to=body.alias_to,
            bust_cache=body.bust_cache,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {e}")


@router.get("/aliases")
async def api_list_aliases():
    """Return all fighter aliases (alias → canonical DB name)."""
    return fighter_alias_service.get_aliases()


@router.post("/aliases")
async def api_upsert_alias(body: AliasUpsertRequest):
    """Add or update a single alias and persist it."""
    try:
        return fighter_alias_service.upsert_alias(body.alias, body.canonical)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/aliases/{alias}")
async def api_delete_alias(alias: str):
    """Remove an alias. 404 if it does not exist."""
    if not fighter_alias_service.remove_alias(alias):
        raise HTTPException(status_code=404, detail=f"No alias '{alias}'")
    return {"status": "deleted", "alias": alias}
