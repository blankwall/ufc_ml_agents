#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi_app.services.ufcstats_sync_service import sync_completed_ufcstats_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync newly completed UFCStats events into the database")
    parser.add_argument("--dry-run", action="store_true", help="Scrape and validate without writing to the database")
    args = parser.parse_args()

    result = sync_completed_ufcstats_events(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
