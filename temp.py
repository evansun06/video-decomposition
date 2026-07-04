from pathlib import Path
import os
import sqlite3


db_path = os.environ["VIDEO_DB"]
con = sqlite3.connect(db_path)

print(f"DB: {db_path}")
print()

print("Recent batches")
for row in con.execute(
    """
    SELECT batch_id, status, submitted_at, finished_at, error
    FROM transcription_batches
    ORDER BY submitted_at DESC
    LIMIT 10
    """
):
    print(" | ".join("" if x is None else str(x) for x in row))

print()
print("Recent failed videos")
for row in con.execute(
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
):
    print(" | ".join("" if x is None else str(x) for x in row))

print()
print("Recent done videos")
for row in con.execute(
    """
    SELECT
        video_id,
        transcript_path,
        text_panel_path,
        sentence_panel_path,
        transcription_finished_at
    FROM videos
    WHERE transcription_status = 'done'
    ORDER BY transcription_finished_at DESC
    LIMIT 20
    """
):
    video_id, transcript_path, text_panel_path, sentence_panel_path, finished_at = row
    transcript_exists = Path(transcript_path).exists() if transcript_path else False
    print(
        " | ".join(
            [
                str(video_id),
                "" if transcript_path is None else str(transcript_path),
                f"exists={transcript_exists}",
                "" if text_panel_path is None else str(text_panel_path),
                "" if sentence_panel_path is None else str(sentence_panel_path),
                "" if finished_at is None else str(finished_at),
            ]
        )
    )
