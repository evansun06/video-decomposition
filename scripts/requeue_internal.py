from __future__ import annotations

from datetime import datetime, timezone
import os
import sqlite3
from pathlib import Path


db_path = os.environ.get("VIDEO_DB")
if not db_path:
    raise SystemExit("VIDEO_DB is not set.")
if not Path(db_path).exists():
    raise SystemExit(f"SQLite DB not found: {db_path}")

now = datetime.now(timezone.utc).isoformat(timespec="seconds")
con = sqlite3.connect(db_path)
cursor = con.execute(
    """
    UPDATE videos
    SET
        transcription_status = 'queued',
        transcription_batch_id = NULL,
        gcs_uri = NULL,
        transcription_started_at = NULL,
        transcription_finished_at = NULL,
        transcription_error = NULL,
        updated_at = ?
    WHERE transcription_status = 'failed'
      AND transcription_error LIKE '%internal error%'
    """,
    (now,),
)
con.commit()

print(f"Requeued internal-error transcription rows: {cursor.rowcount}")
