from __future__ import annotations

import os
import sqlite3
from pathlib import Path


db_path = os.environ.get("VIDEO_DB")
if not db_path:
    raise SystemExit("VIDEO_DB is not set.")
if not Path(db_path).exists():
    raise SystemExit(f"SQLite DB not found: {db_path}")

con = sqlite3.connect(db_path)

rows = con.execute(
    """
    SELECT
        CASE
            WHEN transcription_error LIKE '%too long%' THEN 'too_long'
            WHEN transcription_error LIKE '%internal error%' THEN 'internal_error'
            WHEN transcription_error LIKE '%word_level_confidence%' THEN 'old_bad_config_word_confidence'
            WHEN transcription_error LIKE '%does not exist in the location named "global"%' THEN 'old_bad_config_global'
            WHEN transcription_error IS NULL OR transcription_error = '' THEN 'blank_error'
            ELSE 'other'
        END AS reason,
        COUNT(*)
    FROM videos
    WHERE transcription_status = 'failed'
    GROUP BY reason
    ORDER BY COUNT(*) DESC, reason
    """
).fetchall()

print("Failed transcription reason counts")
print("----------------------------------")
for reason, count in rows:
    print(f"{reason}: {count}")
