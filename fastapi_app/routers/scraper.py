from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.scraper_service import scrape_and_save, list_user_events, delete_user_event, update_results, _slug, USER_EVENTS_DIR
from services.predict_service import analyze_event
from services.sherdog_recovery_service import recover_fighter_from_url

router = APIRouter()


class AddEventRequest(BaseModel):
    bfo_url: str
    ufc_stats_url: Optional[str] = None


class UpdateResultsRequest(BaseModel):
    ufc_stats_url: str


class AnalyzeRequest(BaseModel):
    bfo_url: str
    ufc_stats_url: Optional[str] = None
    force_rescrape: bool = False  # set True to re-fetch from BFO even if cached


class RecoverFighterRequest(BaseModel):
    sherdog_url: str
    requested_name: Optional[str] = None
    dry_run: bool = False
    bust_cache: bool = False


# ── Scrape / manage ───────────────────────────────────────────────────────────

@router.post("/add-event")
async def api_add_event(body: AddEventRequest):
    """
    Scrape a BFO event URL (and optionally a UFC Stats URL for results),
    save to data/user_events/, and return a summary.
    """
    try:
        result = scrape_and_save(body.bfo_url, body.ufc_stats_url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scrape failed: {e}")
    return result


@router.post("/recover-fighter")
async def api_recover_fighter(body: RecoverFighterRequest):
    """Import a single fighter from a Sherdog profile URL into the DB."""
    try:
        result = recover_fighter_from_url(
            fighter_url=body.sherdog_url,
            requested_name=body.requested_name,
            dry_run=body.dry_run,
            bust_cache=body.bust_cache,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recovery failed: {e}")
    return result


@router.get("/user-events")
async def api_list_user_events():
    """List all saved user-added events (metadata only, no fight details)."""
    return list_user_events()


@router.delete("/user-events/{slug}")
async def api_delete_user_event(slug: str):
    """Delete a saved user event by slug."""
    deleted = delete_user_event(slug)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No user event with slug '{slug}'")
    return {"status": "deleted", "slug": slug}


@router.post("/user-events/{slug}/results")
async def api_update_results(slug: str, body: UpdateResultsRequest):
    """
    Update results for an existing event without re-scraping odds.
    Provide the event slug and a UFC Stats URL.
    """
    try:
        result = update_results(slug, body.ufc_stats_url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")
    return result


# ── One-shot analyze ──────────────────────────────────────────────────────────

@router.post("/analyze")
async def api_analyze(body: AnalyzeRequest):
    """
    One-shot: give a BFO URL (+ optional UFC Stats URL) and get back a full
    model analysis for every fight on the card.

    - If the event has already been scraped it uses the cached data (fast).
    - Set force_rescrape=true to re-fetch from BFO (e.g. after odds move).
    - Model predictions are cached per fight pair so repeat calls are cheap.

    Returns:
        {
          event_name, event_date, event_url, source_type,
          summary: { n_fights, n_results, wins, accuracy, pnl, roi },
          fights: [
            { fighter1, fighter2, f1_odds, f2_odds,
              market_prob_f1, model_prob_f1, model_pick, model_source,
              edge, winner, method, round, correct, pnl, error }
          ]
        }
    """
    if body.force_rescrape:
        slug = _slug(body.bfo_url)
        ev_path = USER_EVENTS_DIR / f"{slug}.json"
        if ev_path.exists():
            ev_path.unlink()

    try:
        result = analyze_event(body.bfo_url, body.ufc_stats_url)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    return result


@router.get("/analyze/{slug}")
async def api_analyze_by_slug(slug: str):
    """
    Re-run (or return cached) analysis for a previously added event by slug.
    Get the slug from GET /api/user-events.
    """
    ev_path = USER_EVENTS_DIR / f"{slug}.json"
    if not ev_path.exists():
        raise HTTPException(status_code=404, detail=f"No saved event with slug '{slug}'")

    import json
    payload = json.loads(ev_path.read_text())
    bfo_url = payload.get("bfo_url", "")
    stats_url = payload.get("ufc_stats_url", "")

    try:
        result = analyze_event(bfo_url, stats_url or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    return result
