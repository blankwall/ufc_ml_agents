from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import or_, text

from database.schema import Fight, Fighter

# Fighter name aliases (real-name → DB name or nickname stored in DB).
# Stored in config/fighter_aliases.json and loaded/persisted via
# fighter_alias_service. FIGHTER_ALIASES is the same dict object the service
# mutates in place, so runtime upserts (e.g. from the ingest UI) are visible to
# every module that imports this name.
from fastapi_app.services.fighter_alias_service import ALIASES as FIGHTER_ALIASES


def _best_fighter_match(session, fighters: list[Fighter]) -> Optional[Fighter]:
    if not fighters:
        return None
    if len(fighters) == 1:
        return fighters[0]

    scored = []
    for fighter in fighters:
        count = session.query(Fight).filter(
            or_(Fight.fighter_1_id == fighter.id, Fight.fighter_2_id == fighter.id)
        ).count()
        scored.append((count, fighter.id, fighter))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def resolve_fighter(session, name: str) -> Optional[Fighter]:
    """Best-effort fighter name → DB Fighter.

    Normalises apostrophes, hyphens, and dots on both sides so name variants like
    'Loneer Kavanagh' and 'Waldo Cortes-Acosta' still resolve to the canonical
    main-DB fighter row.
    """

    def _normalized_sql(value: str) -> str:
        return re.sub(r"['\.]", "", value.replace("-", " ")).strip().lower()

    lookup_name = FIGHTER_ALIASES.get(name, name)

    rows = session.query(Fighter).filter(Fighter.name.ilike(f"%{lookup_name}%")).all()
    best = _best_fighter_match(session, rows)
    if best is not None:
        return best

    normalized = _normalized_sql(lookup_name)
    if normalized:
        sql = text(
            "SELECT id FROM fighters "
            "WHERE LOWER(REPLACE(REPLACE(name, '''', ''), '.', '')) LIKE :q"
        )
        ids = [row[0] for row in session.execute(sql, {"q": f"%{normalized}%"})]
        if ids:
            rows = session.query(Fighter).filter(Fighter.id.in_(ids)).all()
            best = _best_fighter_match(session, rows)
            if best is not None:
                return best

    parts = lookup_name.split()
    if len(parts) >= 2:
        first = re.sub(r"['\.\-]", "", parts[0]).lower()
        raw_last = re.sub(r"['\.\-]", "", " ".join(parts[1:])).lower()
        for prefix_len in range(min(len(raw_last), 8), 3, -1):
            prefix = raw_last[:prefix_len]
            sql = text(
                "SELECT id FROM fighters "
                "WHERE LOWER(REPLACE(REPLACE(name, '''', ''), '-', ' ')) LIKE :q"
            )
            ids = [row[0] for row in session.execute(sql, {"q": f"%{prefix}%"})]
            if not ids:
                continue
            rows = session.query(Fighter).filter(Fighter.id.in_(ids)).all()
            rows = [
                fighter
                for fighter in rows
                if first[:3] in fighter.name.lower().replace("'", "").replace("-", " ")
            ]
            best = _best_fighter_match(session, rows)
            if best is not None:
                return best

    return None
