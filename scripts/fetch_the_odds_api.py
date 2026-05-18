#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi_app.services.the_odds_api_service import sync_new_the_odds_api_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch new MMA events from The Odds API")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and report without writing the normalized CSV")
    args = parser.parse_args()

    result = sync_new_the_odds_api_events(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
