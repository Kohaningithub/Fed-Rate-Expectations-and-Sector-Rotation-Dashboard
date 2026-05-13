from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    DEFAULT_START_DATE,
    SNAPSHOT_DIR,
    SNAPSHOT_MACRO_FILE,
    SNAPSHOT_METADATA_FILE,
    SNAPSHOT_PRICES_FILE,
    build_live_dashboard_data,
)


def main() -> None:
    data = build_live_dashboard_data(start_date=DEFAULT_START_DATE)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    data.macro.reset_index(names="Date").to_csv(SNAPSHOT_MACRO_FILE, index=False)
    data.prices.reset_index(names="Date").to_csv(SNAPSHOT_PRICES_FILE, index=False)

    metadata = {
        "snapshot_created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "macro_through": str(data.macro.index.max().date()),
        "prices_through": str(data.prices.index.max().date()),
        "source": data.metadata.get("source", ""),
        "price_sources": data.metadata.get("price_sources", ""),
    }
    SNAPSHOT_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()
