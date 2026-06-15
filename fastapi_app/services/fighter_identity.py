from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import or_, text

from database.schema import Fight, Fighter

# Add entries here when a CSV or sidecar uses a name variant the DB doesn't recognise.
FIGHTER_ALIASES: dict[str, str] = {
    "Bobby Green": "King Green",
    "Sean Omalley": "Sean O'Malley",
    "Charles Johson": "Charles Johnson",
    "Michal Oleksiejczluk": "Michal Oleksiejczuk",
    "Waldo Cortes-Acosta": "Waldo Cortes Acosta",
    "Loneer Kavanagh": "Lone'er Kavanagh",
    "Carlos Leal Miranda": "Carlos Leal",
    "Long Xiao": "Xiao Long",
    "Lupita Godinez": "Loopy Godinez",
    "Benoit St. Denis": "Benoit Saint Denis",
    "Benoit St Denis": "Benoit Saint Denis",
    "Kim Sang Wook": "Sangwook Kim",
    "Jose Medina": "Jose Daniel Medina",
    "Montserrat Rendon": "Montse Rendon",
    "Azamt Bekoev": "Azamat Bekoev",
    "Casey Oneill": "Casey O'Neill",
    "Soo Young Yoo": "SuYoung You",
    # Outcome name mismatches (odds source vs UFC stats canonical)
    "Michael Aswell": "Michael Aswell Jr.",
    "Cameron Rowston": "Cam Rowston",
    "Don Mar Fan": "Dom Mar Fan",
    "Juan Martinetti": "Adrian Luna Martinetti",
    # Sergey sidecar / alternate-source name for the same fighter.
    "Konklak Suphisara": "Loma Lookboonmee",
    "Sergey Pavlovich": "Sergei Pavlovich",
    "Mingyang Zhang": "Zhang Mingyang",
    "Su Mudaerji": "Sumudaerji",
    "José Henrique": "Jose Souza",
    "Meng Ding": "Ding Meng",
    "Aori Qileng": "Aoriqileng",
    "Jingnan Xiong": "Xiong Jingnan",
    "Kangjie Zhu": "Zhu Kangjie",
    "Luis Dias de Assis": "Luis Felipe Dias",
}


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
