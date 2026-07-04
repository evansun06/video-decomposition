from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import GoogleSpeechConfig
from .google_speech import (
    build_batch_recognize_request,
    build_speech_client,
    generate_unique_filename,
    upload_audio_to_gcs,
)
from .job_state.state import DEFAULT_DB_PATH, StageStatus, ensure_database_schema
from .submission import (
    WorkerSummary,
    add_common_arguments,
    chunked,
    configure_logging,
    connect_sqlite,
    positive_int,
    resolve_output_root,
    run_logger,
    run_video_workers,
)


LOGGER_NAME = "youtube_decompose.submit_gcp_stt_batches"
MAX_BATCH_SIZE = 5
DEFAULT_BATCH_SIZE = 5


UploadFunc = Callable[[str | Path, GoogleSpeechConfig, str | None], str]
BatchSubmitter = Callable[[tuple[str, ...], str], str]


@dataclass(frozen=True)
class SttCandidate:
    video_id: str
    audio_path: Path


@dataclass(frozen=True)
class UploadResult:
    video_id: str
    audio_path: Path
    gcs_uri: str


@dataclass(frozen=True)
class SttSubmissionSummary:
    selected: int
    uploaded: int
    upload_failed: int
    batches_submitted: int
    batch_submit_failed_videos: int

    @property
    def failed(self) -> int:
        return self.upload_failed + self.batch_submit_failed_videos


def batch_size(value: str) -> int:
    parsed = positive_int(value)
    if parsed > MAX_BATCH_SIZE:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_BATCH_SIZE}")
    return parsed


def _utc_now_sql() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _operation_name(operation: Any) -> str:
    for candidate in (
        getattr(operation, "name", None),
        getattr(getattr(operation, "operation", None), "name", None),
        getattr(getattr(operation, "_operation", None), "name", None),
    ):
        if candidate:
            return str(candidate)

    raise ValueError("Google batch_recognize operation did not expose a name.")


def _default_batch_submitter(config: GoogleSpeechConfig) -> BatchSubmitter:
    client = build_speech_client(config)

    def submit(gcs_uris: tuple[str, ...], gcs_output_uri: str) -> str:
        request = build_batch_recognize_request(
            gcs_uris=list(gcs_uris),
            config=config,
            gcs_output_uri=gcs_output_uri,
        )
        operation = client.batch_recognize(request=request)
        return _operation_name(operation)

    return submit


def _batch_output_uri(config: GoogleSpeechConfig, batch_id: str) -> str:
    return f"gs://{config.bucket_name}/stt_results/{batch_id}/"


def _fetch_stt_candidates(
    *,
    db_path: str | Path,
    output_root: Path,
    limit: int | None,
    retry_failed: bool,
) -> list[SttCandidate]:
    ensure_database_schema(db_path)
    statuses = [StageStatus.QUEUED.value]
    if retry_failed:
        statuses.append(StageStatus.FAILED.value)

    placeholders = ", ".join("?" for _status in statuses)
    query = f"""
        SELECT video_id, audio_output_path
        FROM videos
        WHERE audio_status = ?
          AND transcription_status IN ({placeholders})
        ORDER BY video_id
    """
    parameters: list[object] = [StageStatus.DONE.value, *statuses]
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)

    with connect_sqlite(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()

    candidates: list[SttCandidate] = []
    for row in rows:
        video_id = str(row["video_id"])
        audio_output_path = row["audio_output_path"]
        audio_path = (
            Path(str(audio_output_path))
            if audio_output_path
            else output_root / video_id / "audio_temp" / "audio_full.wav"
        )
        candidates.append(SttCandidate(video_id=video_id, audio_path=audio_path))

    return candidates


def _mark_transcription_failed_with_attempt(
    *,
    db_path: str | Path,
    video_id: str,
    error: str,
    gcs_uri: str | None = None,
) -> None:
    now = _utc_now_sql()
    with connect_sqlite(db_path) as connection:
        connection.execute(
            """
            UPDATE videos
            SET
                transcription_status = ?,
                gcs_uri = COALESCE(?, gcs_uri),
                transcription_batch_id = NULL,
                transcription_attempts = transcription_attempts + 1,
                transcription_started_at = ?,
                transcription_finished_at = ?,
                transcription_error = ?,
                updated_at = ?
            WHERE video_id = ?
            """,
            (
                StageStatus.FAILED.value,
                gcs_uri,
                now,
                now,
                error,
                now,
                video_id,
            ),
        )
        connection.commit()


def _record_submitted_batch(
    *,
    db_path: str | Path,
    batch_id: str,
    operation_name: str,
    uploads: tuple[UploadResult, ...],
) -> None:
    now = _utc_now_sql()
    gcs_uris = [upload.gcs_uri for upload in uploads]

    with connect_sqlite(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO transcription_batches (
                batch_id,
                operation_name,
                status,
                gcs_uris_json,
                submitted_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, 'submitted', ?, ?, ?, ?)
            """,
            (
                batch_id,
                operation_name,
                json.dumps(gcs_uris),
                now,
                now,
                now,
            ),
        )
        connection.executemany(
            """
            UPDATE videos
            SET
                transcription_status = ?,
                gcs_uri = ?,
                transcription_batch_id = ?,
                transcription_attempts = transcription_attempts + 1,
                transcription_started_at = ?,
                transcription_finished_at = NULL,
                transcription_error = NULL,
                updated_at = ?
            WHERE video_id = ?
            """,
            [
                (
                    StageStatus.RUNNING.value,
                    upload.gcs_uri,
                    batch_id,
                    now,
                    now,
                    upload.video_id,
                )
                for upload in uploads
            ],
        )
        connection.commit()


def _mark_batch_submit_failed(
    *,
    db_path: str | Path,
    uploads: tuple[UploadResult, ...],
    error: str,
) -> None:
    for upload in uploads:
        _mark_transcription_failed_with_attempt(
            db_path=db_path,
            video_id=upload.video_id,
            error=error,
            gcs_uri=upload.gcs_uri,
        )


def submit_gcp_stt_batches(
    *,
    output_root: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int | None = None,
    workers: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retry_failed: bool = False,
    google_config: GoogleSpeechConfig | None = None,
    upload_func: UploadFunc = upload_audio_to_gcs,
    batch_submitter: BatchSubmitter | None = None,
) -> SttSubmissionSummary:
    output_root = Path(output_root)
    config = google_config or GoogleSpeechConfig.from_env()
    submit_batch = batch_submitter or _default_batch_submitter(config)
    logger = run_logger(LOGGER_NAME)

    candidates = _fetch_stt_candidates(
        db_path=db_path,
        output_root=output_root,
        limit=limit,
        retry_failed=retry_failed,
    )

    def upload_candidate(
        candidate: SttCandidate,
        item_logger: logging.LoggerAdapter,
    ) -> UploadResult:
        item_logger.info("Reading extracted audio path=%s", candidate.audio_path)
        if not candidate.audio_path.exists():
            error = f"Audio output does not exist: {candidate.audio_path}"
            _mark_transcription_failed_with_attempt(
                db_path=db_path,
                video_id=candidate.video_id,
                error=error,
            )
            raise ValueError(error)

        object_name = generate_unique_filename(
            f"{candidate.video_id}_{candidate.audio_path.name}"
        )
        try:
            gcs_uri = upload_func(candidate.audio_path, config, object_name)
        except Exception as exc:
            _mark_transcription_failed_with_attempt(
                db_path=db_path,
                video_id=candidate.video_id,
                error=str(exc),
            )
            raise

        item_logger.info("Uploaded audio to GCS gcs_uri=%s", gcs_uri)
        return UploadResult(
            video_id=candidate.video_id,
            audio_path=candidate.audio_path,
            gcs_uri=gcs_uri,
        )

    upload_summary: WorkerSummary[UploadResult] = run_video_workers(
        logger_name=LOGGER_NAME,
        action_name="GCP STT audio upload",
        items=candidates,
        workers=workers,
        get_video_id=lambda candidate: candidate.video_id,
        process_item=upload_candidate,
    )

    submitted_batches = 0
    batch_submit_failed_videos = 0
    for upload_batch in chunked(upload_summary.results, batch_size):
        batch_id = str(uuid.uuid4())
        gcs_uris = tuple(upload.gcs_uri for upload in upload_batch)
        gcs_output_uri = _batch_output_uri(config, batch_id)
        logger.info(
            "Submitting GCP STT batch batch_id=%s size=%s gcs_output_uri=%s",
            batch_id,
            len(upload_batch),
            gcs_output_uri,
        )
        try:
            operation_name = submit_batch(gcs_uris, gcs_output_uri)
            _record_submitted_batch(
                db_path=db_path,
                batch_id=batch_id,
                operation_name=operation_name,
                uploads=upload_batch,
            )
        except Exception as exc:
            error = str(exc)
            _mark_batch_submit_failed(
                db_path=db_path,
                uploads=upload_batch,
                error=error,
            )
            batch_submit_failed_videos += len(upload_batch)
            logger.exception("GCP STT batch submission failed batch_id=%s", batch_id)
            continue

        submitted_batches += 1
        for upload in upload_batch:
            logging.LoggerAdapter(
                logging.getLogger(LOGGER_NAME),
                {"video_id": upload.video_id},
            ).info(
                "Submitted GCP STT batch batch_id=%s operation_name=%s gcs_uri=%s",
                batch_id,
                operation_name,
                upload.gcs_uri,
            )

    summary = SttSubmissionSummary(
        selected=len(candidates),
        uploaded=upload_summary.succeeded,
        upload_failed=upload_summary.failed,
        batches_submitted=submitted_batches,
        batch_submit_failed_videos=batch_submit_failed_videos,
    )
    logger.info(
        "Completed GCP STT submission selected=%s uploaded=%s "
        "upload_failed=%s batches_submitted=%s batch_submit_failed_videos=%s",
        summary.selected,
        summary.uploaded,
        summary.upload_failed,
        summary.batches_submitted,
        summary.batch_submit_failed_videos,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m youtube_decompose.submit_gcp_stt_batches",
        description="Submit extracted audio files to Google Speech-to-Text batches.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--batch-size",
        type=batch_size,
        default=DEFAULT_BATCH_SIZE,
        help=f"GCP files per BatchRecognize request. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Include transcription_status='failed' rows as eligible work.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output_root = resolve_output_root(args.output_root)
    except ValueError as exc:
        parser.error(str(exc))

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(
        logger_name="submit_gcp_stt_batches",
        output_root=output_root,
        level=args.log_level,
        log_file=args.log_file,
    )
    run_logger(LOGGER_NAME).info("Logging to %s", log_path)

    summary = submit_gcp_stt_batches(
        db_path=args.db,
        output_root=output_root,
        limit=args.limit,
        workers=args.workers,
        batch_size=args.batch_size,
        retry_failed=args.retry_failed,
    )
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
