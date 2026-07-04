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
con.row_factory = sqlite3.Row

rows = con.execute(
    """
    SELECT video_id, transcript_path
    FROM videos
    WHERE transcription_status = 'done'
      AND transcript_path IS NOT NULL
    """
).fetchall()

video_ids: list[str] = []
for row in rows:
    transcript_path = Path(str(row["transcript_path"]))
    if transcript_path.exists() and transcript_path.read_text(encoding="utf-8").strip():
        continue
    video_ids.append(str(row["video_id"]))

if video_ids:
    placeholders = ", ".join("?" for _video_id in video_ids)
    con.execute(
        f"""
        UPDATE videos
        SET
            transcription_status = 'queued',
            transcription_batch_id = NULL,
            gcs_uri = NULL,
            transcript_path = NULL,
            text_panel_path = NULL,
            sentence_panel_path = NULL,
            transcription_started_at = NULL,
            transcription_finished_at = NULL,
            transcription_error = NULL,
            updated_at = ?
        WHERE video_id IN ({placeholders})
        """,
        (now, *video_ids),
    )
    con.commit()

print(f"Requeued empty completed transcript rows: {len(video_ids)}")
