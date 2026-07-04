from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import DEFAULT_OUTPUT_ROOT, GoogleSpeechConfig
from .folders import setup_work_folder
from .google_speech import (
    build_speech_client,
    download_google_batch_results_from_gcs,
    write_google_transcript_outputs,
)
from .job_state.state import DEFAULT_DB_PATH, StageStatus, ensure_database_schema
from .submission import resolve_output_root


OperationGetter = Callable[[str], Any]


@dataclass(frozen=True)
class PollSummary:
    checked_batches: int = 0
    pending_batches: int = 0
    done_batches: int = 0
    failed_batches: int = 0
    done_videos: int = 0
    failed_videos: int = 0


def _utc_now_sql() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _operation_done(operation: Any) -> bool:
    done = getattr(operation, "done", False)
    return bool(done() if callable(done) else done)


def _error_message(error: Any) -> str | None:
    if error is None:
        return None

    code = getattr(error, "code", 0)
    message = getattr(error, "message", "")
    if code:
        return str(message or f"Google operation failed with code {code}.")

    return None


def _operation_error(operation: Any) -> str | None:
    return _error_message(getattr(operation, "error", None))


def _file_result_error(file_result: Any) -> str | None:
    return _error_message(getattr(file_result, "error", None))


def _operation_response(operation: Any) -> Any:
    response = getattr(operation, "response", None)
    if response is None:
        raise ValueError("Completed Google operation did not include a response.")
    if hasattr(response, "results"):
        return response

    from google.cloud.speech_v2.types import cloud_speech

    parsed = cloud_speech.BatchRecognizeResponse()
    response.Unpack(parsed._pb)
    return parsed


def _get_file_result(response: Any, gcs_uri: str) -> Any | None:
    results = getattr(response, "results", {})
    if hasattr(results, "get"):
        return results.get(gcs_uri)
    try:
        return results[gcs_uri]
    except (KeyError, TypeError):
        return None


def _transcript_from_file_result(
    file_result: Any,
    google_config: GoogleSpeechConfig | None,
) -> Any | None:
    transcript = getattr(file_result, "transcript", None)
    if transcript is not None:
        return transcript

    output_uri = getattr(file_result, "uri", None)
    if output_uri:
        config = google_config or GoogleSpeechConfig.from_env()
        return download_google_batch_results_from_gcs(str(output_uri), config)

    return None


def _default_operation_getter(config: GoogleSpeechConfig) -> OperationGetter:
    client = build_speech_client(config)

    def get_operation(operation_name: str) -> Any:
        try:
            return client.get_operation(name=operation_name)
        except TypeError:
            return client.get_operation(request={"name": operation_name})

    return get_operation


def _fetch_submitted_batches(
    connection: sqlite3.Connection,
    *,
    limit: int,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            batch_id,
            operation_name,
            gcs_uris_json
        FROM transcription_batches
        WHERE status = 'submitted'
        ORDER BY submitted_at, batch_id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _videos_for_batch_uri(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    gcs_uri: str,
) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT
            video_id,
            transcription_status
        FROM videos
        WHERE transcription_batch_id = ?
          AND gcs_uri = ?
        ORDER BY video_id
        """,
        (batch_id, gcs_uri),
    ).fetchall()


def _mark_video_done(
    connection: sqlite3.Connection,
    *,
    video_id: str,
    transcript_path: Path,
    text_panel_path: Path,
    sentence_panel_path: Path,
    now: str,
) -> None:
    connection.execute(
        """
        UPDATE videos
        SET
            transcription_status = ?,
            transcript_path = ?,
            text_panel_path = ?,
            sentence_panel_path = ?,
            transcription_finished_at = ?,
            transcription_error = NULL,
            updated_at = ?
        WHERE video_id = ?
        """,
        (
            StageStatus.DONE.value,
            str(transcript_path),
            str(text_panel_path),
            str(sentence_panel_path),
            now,
            now,
            video_id,
        ),
    )


def _mark_video_failed(
    connection: sqlite3.Connection,
    *,
    video_id: str,
    error: str,
    now: str,
) -> None:
    connection.execute(
        """
        UPDATE videos
        SET
            transcription_status = ?,
            transcription_finished_at = ?,
            transcription_error = ?,
            updated_at = ?
        WHERE video_id = ?
          AND transcription_status != ?
        """,
        (
            StageStatus.FAILED.value,
            now,
            error,
            now,
            video_id,
            StageStatus.DONE.value,
        ),
    )


def _mark_batch_done(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    now: str,
) -> None:
    connection.execute(
        """
        UPDATE transcription_batches
        SET
            status = 'done',
            finished_at = ?,
            error = NULL,
            updated_at = ?
        WHERE batch_id = ?
        """,
        (now, now, batch_id),
    )


def _mark_batch_failed(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    error: str,
    now: str,
) -> None:
    connection.execute(
        """
        UPDATE transcription_batches
        SET
            status = 'failed',
            finished_at = ?,
            error = ?,
            updated_at = ?
        WHERE batch_id = ?
        """,
        (now, error, now, batch_id),
    )


def _fail_batch_videos(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    error: str,
    now: str,
) -> int:
    cursor = connection.execute(
        """
        UPDATE videos
        SET
            transcription_status = ?,
            transcription_finished_at = ?,
            transcription_error = ?,
            updated_at = ?
        WHERE transcription_batch_id = ?
          AND transcription_status != ?
        """,
        (
            StageStatus.FAILED.value,
            now,
            error,
            now,
            batch_id,
            StageStatus.DONE.value,
        ),
    )
    return cursor.rowcount


def _materialize_batch(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    gcs_uris: list[str],
    response: Any,
    output_root: Path,
    google_config: GoogleSpeechConfig | None,
) -> tuple[int, int]:
    done_videos = 0
    failed_videos = 0
    now = _utc_now_sql()

    for gcs_uri in gcs_uris:
        videos = _videos_for_batch_uri(
            connection,
            batch_id=batch_id,
            gcs_uri=gcs_uri,
        )
        if len(videos) != 1:
            error = (
                f"Expected one video for batch {batch_id!r} and URI {gcs_uri!r}; "
                f"found {len(videos)}."
            )
            raise ValueError(error)

        video = videos[0]
        video_id = str(video["video_id"])
        if video["transcription_status"] == StageStatus.DONE.value:
            continue

        file_result = _get_file_result(response, gcs_uri)
        if file_result is None:
            _mark_video_failed(
                connection,
                video_id=video_id,
                error=f"Google response missing result for {gcs_uri}.",
                now=now,
            )
            failed_videos += 1
            continue

        file_error = _file_result_error(file_result)
        if file_error:
            _mark_video_failed(
                connection,
                video_id=video_id,
                error=file_error,
                now=now,
            )
            failed_videos += 1
            continue

        transcript = _transcript_from_file_result(file_result, google_config)
        if transcript is None:
            _mark_video_failed(
                connection,
                video_id=video_id,
                error=f"Google response missing transcript for {gcs_uri}.",
                now=now,
            )
            failed_videos += 1
            continue

        paths = setup_work_folder(output_root / video_id)
        (
            _transcript_text,
            transcript_path,
            text_panel_path,
            sentence_panel_path,
        ) = write_google_transcript_outputs(transcript, paths["result"])
        _mark_video_done(
            connection,
            video_id=video_id,
            transcript_path=transcript_path,
            text_panel_path=text_panel_path,
            sentence_panel_path=sentence_panel_path,
            now=now,
        )
        done_videos += 1

    _mark_batch_done(connection, batch_id=batch_id, now=now)
    return done_videos, failed_videos


def poll_transcription_batches(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    limit: int = 50,
    google_config: GoogleSpeechConfig | None = None,
    operation_getter: OperationGetter | None = None,
) -> PollSummary:
    ensure_database_schema(db_path)
    output_root = Path(output_root)

    checked_batches = 0
    pending_batches = 0
    done_batches = 0
    failed_batches = 0
    done_videos = 0
    failed_videos = 0

    with _connect(db_path) as connection:
        batches = _fetch_submitted_batches(connection, limit=limit)

    if not batches:
        return PollSummary()

    if operation_getter is None:
        operation_getter = _default_operation_getter(
            google_config or GoogleSpeechConfig.from_env()
        )

    for batch in batches:
        checked_batches += 1
        batch_id = str(batch["batch_id"])
        operation_name = str(batch["operation_name"])
        gcs_uris = json.loads(str(batch["gcs_uris_json"]))
        if not isinstance(gcs_uris, list) or not all(
            isinstance(uri, str) for uri in gcs_uris
        ):
            raise ValueError(f"Batch {batch_id!r} has invalid gcs_uris_json.")

        operation = operation_getter(operation_name)
        if not _operation_done(operation):
            pending_batches += 1
            continue

        now = _utc_now_sql()
        operation_error = _operation_error(operation)
        if operation_error:
            with _connect(db_path) as connection:
                _mark_batch_failed(
                    connection,
                    batch_id=batch_id,
                    error=operation_error,
                    now=now,
                )
                failed_videos += _fail_batch_videos(
                    connection,
                    batch_id=batch_id,
                    error=operation_error,
                    now=now,
                )
                connection.commit()
            failed_batches += 1
            continue

        response = _operation_response(operation)
        try:
            with _connect(db_path) as connection:
                batch_done, batch_failed = _materialize_batch(
                    connection,
                    batch_id=batch_id,
                    gcs_uris=gcs_uris,
                    response=response,
                    output_root=output_root,
                    google_config=google_config,
                )
                connection.commit()
            done_videos += batch_done
            failed_videos += batch_failed
            done_batches += 1
        except Exception as exc:
            with _connect(db_path) as connection:
                connection.execute(
                    """
                    UPDATE transcription_batches
                    SET
                        error = ?,
                        updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (str(exc), _utc_now_sql(), batch_id),
                )
                connection.commit()
            raise

    return PollSummary(
        checked_batches=checked_batches,
        pending_batches=pending_batches,
        done_batches=done_batches,
        failed_batches=failed_batches,
        done_videos=done_videos,
        failed_videos=failed_videos,
    )


def _print_summary(summary: PollSummary) -> None:
    print(f"Checked batches: {summary.checked_batches:,}")
    print(f"Still pending: {summary.pending_batches:,}")
    print(f"Done batches: {summary.done_batches:,}")
    print(f"Failed batches: {summary.failed_batches:,}")
    print(f"Done videos: {summary.done_videos:,}")
    print(f"Failed videos: {summary.failed_videos:,}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m youtube_decompose.poll_transcription_batches",
        description="Poll submitted Google Speech-to-Text batch operations once.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite DB path. Defaults to {DEFAULT_DB_PATH}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "NAS output root. Defaults to $NASOUTPUTPATH. Completed transcript "
            "files are written under this root by video_id."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum submitted batches to poll in this pass. Defaults to 50.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    try:
        output_root = resolve_output_root(args.output_root)
    except ValueError as exc:
        parser.error(str(exc))

    summary = poll_transcription_batches(
        db_path=args.db,
        output_root=output_root,
        limit=args.limit,
    )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
