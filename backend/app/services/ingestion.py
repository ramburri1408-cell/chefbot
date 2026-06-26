"""
app/services/ingestion.py

Menu ingestion pipeline: JSON seed data → PostgreSQL + vector store.

Run once to bootstrap, then re-run whenever the menu changes.
Idempotent: upserts on name so re-running doesn't duplicate records.

Usage:
    python -m app.services.ingestion
    python -m app.services.ingestion --menu data/menu.json
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from app.services.rag import get_vector_store

log = logging.getLogger(__name__)
DEFAULT_MENU = Path(__file__).parent.parent.parent / "data" / "menu.json"


async def ingest(menu_path: Path = DEFAULT_MENU) -> None:
    log.info("Loading menu from %s", menu_path)
    dishes = json.loads(menu_path.read_text())
    log.info("Loaded %d dishes", len(dishes))

    store = get_vector_store()
    count = await store.upsert(dishes)
    log.info("Indexed %d dishes into vector store", count)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest menu into ChefBot")
    parser.add_argument("--menu", default=str(DEFAULT_MENU), help="Path to menu JSON")
    args = parser.parse_args()

    asyncio.run(ingest(Path(args.menu)))
    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
