from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def print_rows(title: str, rows: list[sqlite3.Row]) -> None:
    print()
    print(title)
    print("-" * len(title))
    if not rows:
        print("(none)")
        return
    for row in rows:
        print(" | ".join("" if value is None else str(value) for value in row))


db_path = os.environ.get("VIDEO_DB")
if not db_path:
    raise SystemExit("VIDEO_DB is not set.")

if not Path(db_path).exists():
    raise SystemExit(f"SQLite DB not found: {db_path}")

con = sqlite3.connect(db_path)
con.row_factory = sqlite3.Row

print(f"DB: {db_path}")
print(f"NASOUTPUTPATH: {os.environ.get('NASOUTPUTPATH', '')}")

print_rows(
    "Recent batches",
    con.execute(
        """
        SELECT batch_id, status, submitted_at, finished_at, error
        FROM transcription_batches
        ORDER BY submitted_at DESC
        LIMIT 10
        """
    ).fetchall(),
)

print_rows(
    "Videos in done batches",
    con.execute(
        """
        SELECT
            v.video_id,
            b.batch_id,
            v.transcription_status,
            v.transcript_path,
            v.transcription_error
        FROM videos v
        JOIN transcription_batches b ON v.transcription_batch_id = b.batch_id
        WHERE b.status = 'done'
        ORDER BY b.finished_at DESC, v.video_id
        LIMIT 50
        """
    ).fetchall(),
)

print()
print("Done transcript file existence")
print("------------------------------")
rows = con.execute(
    """
    SELECT v.video_id, v.transcript_path
    FROM videos v
    JOIN transcription_batches b ON v.transcription_batch_id = b.batch_id
    WHERE b.status = 'done'
    ORDER BY b.finished_at DESC, v.video_id
    LIMIT 50
    """
).fetchall()
if not rows:
    print("(none)")
else:
    for row in rows:
        path = row["transcript_path"]
        exists = Path(path).exists() if path else False
        print(f"{row['video_id']} | {path or ''} | exists={exists}")

print_rows(
    "Recent failed videos",
    con.execute(
        """
        SELECT
            video_id,
            transcription_batch_id,
            gcs_uri,
            transcription_error,
            transcription_finished_at
        FROM videos
        WHERE transcription_status = 'failed'
        ORDER BY transcription_finished_at DESC
        LIMIT 20
        """
    ).fetchall(),
)
