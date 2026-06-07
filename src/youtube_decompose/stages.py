from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_OUTPUT_ROOT, GoogleSpeechConfig
from .folders import setup_work_folder
from .google_speech import GoogleTranscriptionResult, transcribe_audio_with_google
from .job_state.state import DEFAULT_DB_PATH, StageStatus
from .media import convert_video_to_audio, convert_video_to_images


@dataclass(frozen=True)
class VideoStateRow:
    video_id: str
    nas_path: str
    audio_status: str
    audio_output_path: str | None
    image_status: str
    image_output_dir: str | None
    frame_count: int | None
    transcription_status: str
    transcript_path: str | None
    text_panel_path: str | None
    sentence_panel_path: str | None
    gcs_uri: str | None


def _utc_now_sql() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _fetch_video(connection: sqlite3.Connection, video_id: str) -> VideoStateRow:
    row = connection.execute(
        """
        SELECT
            video_id,
            nas_path,
            audio_status,
            audio_output_path,
            image_status,
            image_output_dir,
            frame_count,
            transcription_status,
            transcript_path,
            text_panel_path,
            sentence_panel_path,
            gcs_uri
        FROM videos
        WHERE video_id = ?
        """,
        (video_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown video_id: {video_id}")
    return VideoStateRow(**dict(row))


def _mark_audio_running(connection: sqlite3.Connection, video_id: str) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            audio_status = ?,
            audio_attempts = audio_attempts + 1,
            audio_started_at = ?,
            audio_finished_at = NULL,
            audio_error = NULL,
            updated_at = ?
        WHERE video_id = ?
        """,
        (StageStatus.RUNNING.value, now, now, video_id),
    )


def _mark_audio_done(
    connection: sqlite3.Connection,
    video_id: str,
    audio_path: Path,
) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            audio_status = ?,
            audio_output_path = ?,
            audio_finished_at = ?,
            audio_error = NULL,
            transcription_status = CASE
                WHEN transcription_status = ? THEN ?
                ELSE transcription_status
            END,
            updated_at = ?
        WHERE video_id = ?
        """,
        (
            StageStatus.DONE.value,
            str(audio_path),
            now,
            StageStatus.WAITING_AUDIO.value,
            StageStatus.QUEUED.value,
            now,
            video_id,
        ),
    )


def _mark_audio_failed(
    connection: sqlite3.Connection,
    video_id: str,
    error: str,
) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            audio_status = ?,
            audio_finished_at = ?,
            audio_error = ?,
            updated_at = ?
        WHERE video_id = ?
        """,
        (StageStatus.FAILED.value, now, error, now, video_id),
    )


def _mark_image_running(connection: sqlite3.Connection, video_id: str) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            image_status = ?,
            image_attempts = image_attempts + 1,
            image_started_at = ?,
            image_finished_at = NULL,
            image_error = NULL,
            updated_at = ?
        WHERE video_id = ?
        """,
        (StageStatus.RUNNING.value, now, now, video_id),
    )


def _mark_image_done(
    connection: sqlite3.Connection,
    video_id: str,
    image_dir: Path,
    frame_count: int,
) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            image_status = ?,
            image_output_dir = ?,
            frame_count = ?,
            image_finished_at = ?,
            image_error = NULL,
            updated_at = ?
        WHERE video_id = ?
        """,
        (StageStatus.DONE.value, str(image_dir), frame_count, now, now, video_id),
    )


def _mark_image_failed(
    connection: sqlite3.Connection,
    video_id: str,
    error: str,
) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            image_status = ?,
            image_finished_at = ?,
            image_error = ?,
            updated_at = ?
        WHERE video_id = ?
        """,
        (StageStatus.FAILED.value, now, error, now, video_id),
    )


def _mark_transcription_running(
    connection: sqlite3.Connection,
    video_id: str,
) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            transcription_status = ?,
            transcription_attempts = transcription_attempts + 1,
            transcription_started_at = ?,
            transcription_finished_at = NULL,
            transcription_error = NULL,
            updated_at = ?
        WHERE video_id = ?
        """,
        (StageStatus.RUNNING.value, now, now, video_id),
    )


def _mark_transcription_done(
    connection: sqlite3.Connection,
    video_id: str,
    result: GoogleTranscriptionResult,
) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            transcription_status = ?,
            transcript_path = ?,
            text_panel_path = ?,
            sentence_panel_path = ?,
            gcs_uri = ?,
            transcription_finished_at = ?,
            transcription_error = NULL,
            updated_at = ?
        WHERE video_id = ?
        """,
        (
            StageStatus.DONE.value,
            str(result.transcript_path),
            str(result.text_panel_path),
            str(result.sentence_panel_path),
            result.gcs_uri,
            now,
            now,
            video_id,
        ),
    )


def _mark_transcription_failed(
    connection: sqlite3.Connection,
    video_id: str,
    error: str,
) -> None:
    now = _utc_now_sql()
    connection.execute(
        """
        UPDATE videos
        SET
            transcription_status = ?,
            transcription_finished_at = ?,
            transcription_error = ?,
            updated_at = ?
        WHERE video_id = ?
        """,
        (StageStatus.FAILED.value, now, error, now, video_id),
    )


def _work_dir(output_root: str | Path, video_id: str) -> Path:
    return Path(output_root) / video_id


def _audio_path_from_row(row: VideoStateRow, work_dir: Path) -> Path:
    if row.audio_output_path:
        return Path(row.audio_output_path)
    return work_dir / "audio_temp" / "audio_full.wav"


def _require_runnable(status: str, stage: str) -> None:
    runnable = {StageStatus.QUEUED.value, StageStatus.FAILED.value}
    if status not in runnable:
        raise ValueError(f"{stage} stage is not runnable from status {status!r}.")


def _existing_transcription_result(row: VideoStateRow) -> GoogleTranscriptionResult:
    if not row.transcript_path or not row.text_panel_path or not row.sentence_panel_path:
        raise ValueError("Transcription output paths are not recorded.")

    transcript_path = Path(row.transcript_path)
    return GoogleTranscriptionResult(
        transcript_text=transcript_path.read_text(encoding="utf-8")
        if transcript_path.exists()
        else "",
        transcript_path=transcript_path,
        text_panel_path=Path(row.text_panel_path),
        sentence_panel_path=Path(row.sentence_panel_path),
        gcs_uri=row.gcs_uri or "",
    )


def run_audio_stage_for_video(
    video_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    with _connect(db_path) as connection:
        row = _fetch_video(connection, video_id)
        work_dir = _work_dir(output_root, video_id)
        audio_path = _audio_path_from_row(row, work_dir)

        if row.audio_status == StageStatus.DONE.value and audio_path.exists():
            return audio_path
        if audio_path.exists() and row.audio_status in {
            StageStatus.QUEUED.value,
            StageStatus.FAILED.value,
        }:
            _mark_audio_done(connection, video_id, audio_path)
            connection.commit()
            return audio_path

        _require_runnable(row.audio_status, "audio")
        _mark_audio_running(connection, video_id)
        connection.commit()

    try:
        if not row.nas_path.strip():
            raise ValueError("Source NAS path is blank.")

        setup_work_folder(work_dir)
        audio_path = convert_video_to_audio(video_path=row.nas_path, work_dir=work_dir)
    except Exception as exc:
        with _connect(db_path) as connection:
            _mark_audio_failed(connection, video_id, str(exc))
            connection.commit()
        raise

    with _connect(db_path) as connection:
        _mark_audio_done(connection, video_id, audio_path)
        connection.commit()

    return audio_path


def run_transcription_stage_for_video(
    video_id: str,
    google_config: GoogleSpeechConfig,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> GoogleTranscriptionResult:
    with _connect(db_path) as connection:
        row = _fetch_video(connection, video_id)
        work_dir = _work_dir(output_root, video_id)
        audio_path = _audio_path_from_row(row, work_dir)

        if row.transcription_status == StageStatus.DONE.value:
            return _existing_transcription_result(row)

        _require_runnable(row.transcription_status, "transcription")
        if row.audio_status != StageStatus.DONE.value:
            raise ValueError(
                "Transcription requires audio_status='done'; "
                f"found {row.audio_status!r}."
            )

        _mark_transcription_running(connection, video_id)
        connection.commit()

    try:
        if not audio_path.exists():
            raise ValueError(f"Audio output does not exist: {audio_path}")

        paths = setup_work_folder(work_dir)
        result = transcribe_audio_with_google(
            audio_path=audio_path,
            result_dir=paths["result"],
            config=google_config,
        )
    except Exception as exc:
        with _connect(db_path) as connection:
            _mark_transcription_failed(connection, video_id, str(exc))
            connection.commit()
        raise

    with _connect(db_path) as connection:
        _mark_transcription_done(connection, video_id, result)
        connection.commit()

    return result


def run_image_stage_for_video(
    video_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    frame_rate: int = 10,
) -> int:
    with _connect(db_path) as connection:
        row = _fetch_video(connection, video_id)
        work_dir = _work_dir(output_root, video_id)
        image_dir = work_dir / "image_temp"

        if row.image_status == StageStatus.DONE.value and row.frame_count is not None:
            return row.frame_count
        if image_dir.exists() and any(image_dir.glob("image_split-*.png")):
            frame_count = len(list(image_dir.glob("image_split-*.png")))
            _mark_image_done(connection, video_id, image_dir, frame_count)
            connection.commit()
            return frame_count

        _require_runnable(row.image_status, "image")
        _mark_image_running(connection, video_id)
        connection.commit()

    try:
        if not row.nas_path.strip():
            raise ValueError("Source NAS path is blank.")

        paths = setup_work_folder(work_dir)
        frame_count = convert_video_to_images(
            video_path=row.nas_path,
            work_dir=work_dir,
            frame_rate=frame_rate,
        )
        image_dir = paths["image"]
    except Exception as exc:
        with _connect(db_path) as connection:
            _mark_image_failed(connection, video_id, str(exc))
            connection.commit()
        raise

    with _connect(db_path) as connection:
        _mark_image_done(connection, video_id, image_dir, frame_count)
        connection.commit()

    return frame_count


__all__ = [
    "VideoStateRow",
    "run_audio_stage_for_video",
    "run_image_stage_for_video",
    "run_transcription_stage_for_video",
]
