"""Append-only CSV log of every fired action.

Each row:
    timestamp_iso, actor, gesture, action, status

Created lazily on the first write. Safe to delete - the next action will
recreate the header.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from threading import Lock
from typing import List, Optional


CSV_HEADER: List[str] = ["timestamp_iso", "actor", "gesture", "action", "status"]


class SessionLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._lock = Lock()

    def log(self, actor: str, gesture: str, action: str, status: str) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        row = [timestamp, actor or "?", gesture, action, status]
        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            new_file = not os.path.exists(self.path)
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(CSV_HEADER)
                writer.writerow(row)

    def recent_rows(self, n: int = 5) -> List[List[str]]:
        """Return the last `n` rows (without header). For HUD preview."""
        if not self.enabled or not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except OSError:
            return []
        if len(rows) <= 1:
            return []
        # Skip the header row.
        return rows[1:][-n:]
