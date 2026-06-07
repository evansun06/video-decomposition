from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from .job_state.state import DEFAULT_DB_PATH, StageStatus, ensure_database_schema


RUN_LOG_VIDEO_ID = "__run__"
NAS_OUTPUT_ENV_VAR = "NASOUTPUTPATH"
LOG_FORMAT = (
    "%(asctime)s %(levelname)s [video_id=%(video_id)s] %(name)s: %(message)s"
)
LOG_LEVELS = {
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
}
ALLOWED_STATUS_COLUMNS = {
    "audio_status",
    "image_status",
    "transcription_status",
}


@dataclass(frozen=True)
class VideoCandidate:
    video_id: str
    nas_path: str


@dataclass(frozen=True)
class WorkerFailure:
    video_id: str
    error: str


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class WorkerSummary:
    selected: int
    succeeded: int
    failed: int
    results: tuple[R, ...]
    failures: tuple[WorkerFailure, ...]


class _VideoIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "video_id"):
            record.video_id = RUN_LOG_VIDEO_ID
        return True


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def log_level(value: str) -> str:
    normalized = value.upper()
    if normalized not in LOG_LEVELS:
        raise argparse.ArgumentTypeError(f"unknown log level: {value}")
    return normalized


def resolve_output_root(output_root: str | Path | None) -> Path:
    root = output_root
    if root is None:
        root = os.environ.get(NAS_OUTPUT_ENV_VAR)

    if root is None or str(root).strip() == "":
        raise ValueError(
            f"Pass --output-root or set {NAS_OUTPUT_ENV_VAR} so outputs are "
            "written under NASOUTPUTPATH/video_id/."
        )

    return Path(root).expanduser()


def configure_logging(
    *,
    logger_name: str,
    output_root: Path,
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> Path:
    log_path = Path(log_file).expanduser() if log_file else (
        output_root / "_logs" / f"{logger_name}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)
    video_filter = _VideoIdFilter()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(video_filter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(video_filter)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(getattr(logging, level))
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    return log_path


def run_logger(logger_name: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(
        logging.getLogger(logger_name),
        {"video_id": RUN_LOG_VIDEO_ID},
    )


def video_logger(logger_name: str, video_id: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(
        logging.getLogger(logger_name),
        {"video_id": video_id},
    )


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_workers: int = 1,
) -> None:
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
            f"NAS output root. Defaults to ${NAS_OUTPUT_ENV_VAR}. Outputs are "
            "written under this root by video_id."
        ),
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum videos to submit. Omit to submit all eligible videos.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=default_workers,
        help=f"Parallel workers. Defaults to {default_workers}.",
    )
    parser.add_argument(
        "--log-level",
        type=log_level,
        default="INFO",
        help="Python logging level. Defaults to INFO.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file. Defaults to output-root/_logs/<script>.log.",
    )


def connect_sqlite(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def fetch_stage_candidates(
    *,
    db_path: str | Path,
    status_column: str,
    statuses: Iterable[StageStatus],
    limit: int | None,
) -> list[VideoCandidate]:
    if status_column not in ALLOWED_STATUS_COLUMNS:
        raise ValueError(f"Unsupported status column: {status_column}")

    ensure_database_schema(db_path)
    status_values = [status.value for status in statuses]
    if not status_values:
        return []

    placeholders = ", ".join("?" for _status in status_values)
    query = f"""
        SELECT video_id, nas_path
        FROM videos
        WHERE {status_column} IN ({placeholders})
        ORDER BY video_id
    """
    parameters: list[object] = list(status_values)
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)

    with connect_sqlite(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [
        VideoCandidate(video_id=str(row["video_id"]), nas_path=str(row["nas_path"]))
        for row in rows
    ]


def chunked(values: Sequence[T], size: int) -> Iterable[tuple[T, ...]]:
    for index in range(0, len(values), size):
        yield tuple(values[index : index + size])


def run_video_workers(
    *,
    logger_name: str,
    action_name: str,
    items: Sequence[T],
    workers: int,
    get_video_id: Callable[[T], str],
    process_item: Callable[[T, logging.LoggerAdapter], R],
) -> WorkerSummary[R]:
    logger = run_logger(logger_name)
    logger.info(
        "Selected %s videos for %s with workers=%s",
        len(items),
        action_name,
        workers,
    )

    results: list[R] = []
    failures: list[WorkerFailure] = []

    def run_one(item: T) -> R:
        video_id = get_video_id(item)
        item_logger = video_logger(logger_name, video_id)
        item_logger.info("Starting %s", action_name)
        result = process_item(item, item_logger)
        item_logger.info("Finished %s", action_name)
        return result

    if workers == 1:
        for item in items:
            video_id = get_video_id(item)
            try:
                results.append(run_one(item))
            except Exception as exc:
                failures.append(WorkerFailure(video_id=video_id, error=str(exc)))
                video_logger(logger_name, video_id).exception("%s failed", action_name)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {executor.submit(run_one, item): item for item in items}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                video_id = get_video_id(item)
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append(WorkerFailure(video_id=video_id, error=str(exc)))
                    video_logger(logger_name, video_id).exception("%s failed", action_name)

    summary: WorkerSummary[R] = WorkerSummary(
        selected=len(items),
        succeeded=len(results),
        failed=len(failures),
        results=tuple(results),
        failures=tuple(failures),
    )
    logger.info(
        "Completed %s selected=%s succeeded=%s failed=%s",
        action_name,
        summary.selected,
        summary.succeeded,
        summary.failed,
    )
    return summary
