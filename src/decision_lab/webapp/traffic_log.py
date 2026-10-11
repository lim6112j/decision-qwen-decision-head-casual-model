"""Opt-in capture of ``/api/decide-dynamic`` calls for the real-data pipeline.

Two gates must both pass before anything is written:

1. ``web.log_traffic`` is enabled in config, **and**
2. the request carries the header ``X-Decision-Lab-Log: 1``.

Logged text is real user content, so capture is never silent. Rows land in
``data/traffic/YYYY-MM-DD.jsonl`` (gitignored) — local only.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

LOG_HEADER = "x-decision-lab-log"
LOG_HEADER_VALUE = "1"


class TrafficLogger:
    """Append-only JSONL logger, one file per UTC day, thread-safe."""

    def __init__(self, traffic_dir: Path, enabled: bool = False) -> None:
        self.dir = Path(traffic_dir)
        self.enabled = enabled
        self._lock = threading.Lock()

    def should_log(self, headers) -> bool:
        """True only when capture is enabled and the caller opted in."""
        if not self.enabled:
            return False
        return (headers.get(LOG_HEADER) or "").strip() == LOG_HEADER_VALUE

    def record(self, row: dict) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")