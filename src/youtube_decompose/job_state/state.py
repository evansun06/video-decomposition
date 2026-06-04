from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Sequence

from .util import (
    DEFAULT_MAX_NON_LATIN_TITLE_RATIO,
    max_non_latin_title_ratio as calculate_max_non_latin_title_ratio,
    parse_bool,
    parse_duration_minutes,
    should_exclude_duration,
)


DEFAULT_CSV_PATH = Path("src/data/youtube_content_v11_20250612.csv")
DEFAULT_DB_PATH = Path("job_state/video_state.sqlite")
DEFAULT_MAX_DURATION_MINUTES = 30.0
SCHEMA_VERSION = 2

REQUIRED_COLUMNS = {
    "id",
    "path",
    "title",
    "is_downloaded",
    "has_heatmap",
    "contentDetails.duration",
}


class StageStatus(str, Enum):
    WAITING_SOURCE = "waiting_source"
    WAITING_AUDIO = "waiting_audio"
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class VideoSeed:
    video_id: str
    nas_path: str
    duration_minutes: float
    source_downloaded: bool
    source_has_heatmap: bool
    source_row_count: int
    audio_status: StageStatus
    image_status: StageStatus
    transcription_status: StageStatus


@dataclass(frozen=True)
class SeedSummary:
    csv_path: Path
    db_path: Path
    seeded_at: str
    total_csv_rows: int
    unique_source_video_count: int
    seeded_video_count: int
    excluded_gte_30_count: int
    excluded_non_latin_title_count: int
    excluded_non_numeric_duration_count: int
    excluded_duration_conflict_count: int
    downloaded_no_heatmap_deleted_count: int
    downloaded_no_heatmap_title_deleted_count: int
    existing_done_count: int
    queued_downloaded_count: int
    not_downloaded_count: int


@dataclass(frozen=True)
class _SeedBuildResult:
    seed: VideoSeed | None
    excluded_reason: str | None
    source_downloaded: bool
    source_has_heatmap: bool


__all__ = [
    "DEFAULT_CSV_PATH",
    "DEFAULT_DB_PATH",
    "DEFAULT_MAX_DURATION_MINUTES",
    "DEFAULT_MAX_NON_LATIN_TITLE_RATIO",
    "SCHEMA_VERSION",
    "REQUIRED_COLUMNS",
    "SeedSummary",
    "StageStatus",
    "VideoSeed",
    "init_database",
    "load_video_seeds",
    "main",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _status_values_sql() -> str:
    return ", ".join(f"'{status.value}'" for status in StageStatus)


def _require_columns(fieldnames: Sequence[str] | None, csv_path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{csv_path} is empty or missing a header row.")

    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(missing)}"
        )


def _single_value(
    rows: list[dict[str, str]],
    *,
    video_id: str,
    column: str,
) -> str:
    values = {row[column].strip() for row in rows}
    if len(values) != 1:
        raise ValueError(
            f"Conflicting {column!r} values for video_id {video_id!r}: "
            f"{sorted(values)!r}"
        )
    return next(iter(values))


def _build_seed_for_video(
    video_id: str,
    rows: list[dict[str, str]],
    *,
    max_duration_minutes: float,
    max_non_latin_title_ratio: float,
) -> _SeedBuildResult:
    nas_path = _single_value(rows, video_id=video_id, column="path")
    source_downloaded = parse_bool(
        _single_value(rows, video_id=video_id, column="is_downloaded"),
        field_name="is_downloaded",
    )
    source_has_heatmap = parse_bool(
        _single_value(rows, video_id=video_id, column="has_heatmap"),
        field_name="has_heatmap",
    )

    duration_values = [
        parse_duration_minutes(row["contentDetails.duration"]) for row in rows
    ]
    if any(duration is None for duration in duration_values):
        return _SeedBuildResult(
            seed=None,
            excluded_reason="non_numeric_duration",
            source_downloaded=source_downloaded,
            source_has_heatmap=source_has_heatmap,
        )

    unique_durations = {duration for duration in duration_values if duration is not None}
    if len(unique_durations) != 1:
        return _SeedBuildResult(
            seed=None,
            excluded_reason="duration_conflict",
            source_downloaded=source_downloaded,
            source_has_heatmap=source_has_heatmap,
        )

    duration_minutes = next(iter(unique_durations))
    if should_exclude_duration(
        duration_minutes,
        max_duration_minutes=max_duration_minutes,
    ):
        return _SeedBuildResult(
            seed=None,
            excluded_reason="gte_30",
            source_downloaded=source_downloaded,
            source_has_heatmap=source_has_heatmap,
        )

    title_ratio = calculate_max_non_latin_title_ratio(
        row["title"].strip() for row in rows
    )
    if title_ratio > max_non_latin_title_ratio:
        return _SeedBuildResult(
            seed=None,
            excluded_reason="non_latin_title",
            source_downloaded=source_downloaded,
            source_has_heatmap=source_has_heatmap,
        )

    if source_has_heatmap:
        audio_status = StageStatus.DONE
        image_status = StageStatus.DONE
        transcription_status = StageStatus.DONE
    elif source_downloaded:
        audio_status = StageStatus.QUEUED
        image_status = StageStatus.QUEUED
        transcription_status = StageStatus.WAITING_AUDIO
    else:
        audio_status = StageStatus.WAITING_SOURCE
        image_status = StageStatus.WAITING_SOURCE
        transcription_status = StageStatus.WAITING_SOURCE

    return _SeedBuildResult(
        seed=VideoSeed(
            video_id=video_id,
            nas_path=nas_path,
            duration_minutes=duration_minutes,
            source_downloaded=source_downloaded,
            source_has_heatmap=source_has_heatmap,
            source_row_count=len(rows),
            audio_status=audio_status,
            image_status=image_status,
            transcription_status=transcription_status,
        ),
        excluded_reason=None,
        source_downloaded=source_downloaded,
        source_has_heatmap=source_has_heatmap,
    )


def _is_downloaded_no_heatmap(result: _SeedBuildResult) -> bool:
    return result.source_downloaded and not result.source_has_heatmap


def load_video_seeds(
    csv_path: str | Path,
    *,
    max_duration_minutes: float = DEFAULT_MAX_DURATION_MINUTES,
    max_non_latin_title_ratio: float = DEFAULT_MAX_NON_LATIN_TITLE_RATIO,
) -> tuple[list[VideoSeed], dict[str, int]]:
    csv_path = Path(csv_path)
    grouped_rows: dict[str, list[dict[str, str]]] = {}
    total_csv_rows = 0

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        _require_columns(reader.fieldnames, csv_path)

        for row in reader:
            total_csv_rows += 1
            video_id = row["id"].strip()
            if not video_id:
                raise ValueError(f"Row {total_csv_rows + 1} has a blank id.")
            grouped_rows.setdefault(video_id, []).append(row)

    seeds: list[VideoSeed] = []
    excluded_gte_30_count = 0
    excluded_non_latin_title_count = 0
    excluded_non_numeric_duration_count = 0
    excluded_duration_conflict_count = 0
    downloaded_no_heatmap_deleted_count = 0
    downloaded_no_heatmap_title_deleted_count = 0

    for video_id in sorted(grouped_rows):
        result = _build_seed_for_video(
            video_id,
            grouped_rows[video_id],
            max_duration_minutes=max_duration_minutes,
            max_non_latin_title_ratio=max_non_latin_title_ratio,
        )
        if result.seed is not None:
            seeds.append(result.seed)
            continue

        if result.excluded_reason == "gte_30":
            excluded_gte_30_count += 1
            if _is_downloaded_no_heatmap(result):
                downloaded_no_heatmap_deleted_count += 1
        elif result.excluded_reason == "non_latin_title":
            excluded_non_latin_title_count += 1
            if _is_downloaded_no_heatmap(result):
                downloaded_no_heatmap_deleted_count += 1
                downloaded_no_heatmap_title_deleted_count += 1
        elif result.excluded_reason == "non_numeric_duration":
            excluded_non_numeric_duration_count += 1
        elif result.excluded_reason == "duration_conflict":
            excluded_duration_conflict_count += 1
        else:
            raise AssertionError(
                f"Unhandled exclusion reason: {result.excluded_reason!r}"
            )

    return seeds, {
        "total_csv_rows": total_csv_rows,
        "unique_source_video_count": len(grouped_rows),
        "excluded_gte_30_count": excluded_gte_30_count,
        "excluded_non_latin_title_count": excluded_non_latin_title_count,
        "excluded_non_numeric_duration_count": excluded_non_numeric_duration_count,
        "excluded_duration_conflict_count": excluded_duration_conflict_count,
        "downloaded_no_heatmap_deleted_count": downloaded_no_heatmap_deleted_count,
        "downloaded_no_heatmap_title_deleted_count": (
            downloaded_no_heatmap_title_deleted_count
        ),
    }


def _create_schema(connection: sqlite3.Connection) -> None:
    status_values = _status_values_sql()
    connection.executescript(
        f"""
        PRAGMA foreign_keys = ON;

        CREATE TABLE videos (
            video_id TEXT PRIMARY KEY,
            nas_path TEXT NOT NULL,
            duration_minutes REAL NOT NULL,
            source_downloaded INTEGER NOT NULL CHECK (source_downloaded IN (0, 1)),
            source_has_heatmap INTEGER NOT NULL CHECK (source_has_heatmap IN (0, 1)),
            source_row_count INTEGER NOT NULL CHECK (source_row_count >= 1),

            audio_status TEXT NOT NULL CHECK (audio_status IN ({status_values})),
            audio_output_path TEXT,
            audio_attempts INTEGER NOT NULL DEFAULT 0 CHECK (audio_attempts >= 0),
            audio_started_at TEXT,
            audio_finished_at TEXT,
            audio_error TEXT,

            image_status TEXT NOT NULL CHECK (image_status IN ({status_values})),
            image_output_dir TEXT,
            frame_count INTEGER CHECK (frame_count IS NULL OR frame_count >= 0),
            image_attempts INTEGER NOT NULL DEFAULT 0 CHECK (image_attempts >= 0),
            image_started_at TEXT,
            image_finished_at TEXT,
            image_error TEXT,

            transcription_status TEXT NOT NULL CHECK (
                transcription_status IN ({status_values})
            ),
            transcript_path TEXT,
            text_panel_path TEXT,
            sentence_panel_path TEXT,
            gcs_uri TEXT,
            transcription_attempts INTEGER NOT NULL DEFAULT 0
                CHECK (transcription_attempts >= 0),
            transcription_started_at TEXT,
            transcription_finished_at TEXT,
            transcription_error TEXT,

            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX idx_videos_audio_status ON videos(audio_status);
        CREATE INDEX idx_videos_image_status ON videos(image_status);
        CREATE INDEX idx_videos_transcription_status
            ON videos(transcription_status);

        CREATE TABLE state_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            schema_version INTEGER NOT NULL,
            csv_path TEXT NOT NULL,
            seeded_at TEXT NOT NULL,
            total_csv_rows INTEGER NOT NULL,
            unique_source_video_count INTEGER NOT NULL,
            seeded_video_count INTEGER NOT NULL,
            excluded_gte_30_count INTEGER NOT NULL,
            excluded_non_latin_title_count INTEGER NOT NULL,
            excluded_non_numeric_duration_count INTEGER NOT NULL,
            excluded_duration_conflict_count INTEGER NOT NULL,
            downloaded_no_heatmap_deleted_count INTEGER NOT NULL,
            downloaded_no_heatmap_title_deleted_count INTEGER NOT NULL,
            existing_done_count INTEGER NOT NULL,
            queued_downloaded_count INTEGER NOT NULL,
            not_downloaded_count INTEGER NOT NULL
        );
        """
    )


def _insert_seeds(
    connection: sqlite3.Connection,
    seeds: list[VideoSeed],
    *,
    created_at: str,
) -> None:
    connection.executemany(
        """
        INSERT INTO videos (
            video_id,
            nas_path,
            duration_minutes,
            source_downloaded,
            source_has_heatmap,
            source_row_count,
            audio_status,
            image_status,
            transcription_status,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                seed.video_id,
                seed.nas_path,
                seed.duration_minutes,
                int(seed.source_downloaded),
                int(seed.source_has_heatmap),
                seed.source_row_count,
                seed.audio_status.value,
                seed.image_status.value,
                seed.transcription_status.value,
                created_at,
                created_at,
            )
            for seed in seeds
        ],
    )


def init_database(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    max_duration_minutes: float = DEFAULT_MAX_DURATION_MINUTES,
    max_non_latin_title_ratio: float = DEFAULT_MAX_NON_LATIN_TITLE_RATIO,
) -> SeedSummary:
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    seeds, counts = load_video_seeds(
        csv_path,
        max_duration_minutes=max_duration_minutes,
        max_non_latin_title_ratio=max_non_latin_title_ratio,
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    seeded_at = _utc_now()
    existing_done_count = sum(seed.source_has_heatmap for seed in seeds)
    queued_downloaded_count = sum(
        (not seed.source_has_heatmap) and seed.source_downloaded for seed in seeds
    )
    not_downloaded_count = sum(
        (not seed.source_has_heatmap) and (not seed.source_downloaded)
        for seed in seeds
    )

    summary = SeedSummary(
        csv_path=csv_path,
        db_path=db_path,
        seeded_at=seeded_at,
        total_csv_rows=counts["total_csv_rows"],
        unique_source_video_count=counts["unique_source_video_count"],
        seeded_video_count=len(seeds),
        excluded_gte_30_count=counts["excluded_gte_30_count"],
        excluded_non_latin_title_count=counts["excluded_non_latin_title_count"],
        excluded_non_numeric_duration_count=counts[
            "excluded_non_numeric_duration_count"
        ],
        excluded_duration_conflict_count=counts["excluded_duration_conflict_count"],
        downloaded_no_heatmap_deleted_count=counts[
            "downloaded_no_heatmap_deleted_count"
        ],
        downloaded_no_heatmap_title_deleted_count=counts[
            "downloaded_no_heatmap_title_deleted_count"
        ],
        existing_done_count=existing_done_count,
        queued_downloaded_count=queued_downloaded_count,
        not_downloaded_count=not_downloaded_count,
    )

    with sqlite3.connect(db_path) as connection:
        _create_schema(connection)
        _insert_seeds(connection, seeds, created_at=seeded_at)
        connection.execute(
            """
            INSERT INTO state_metadata (
                id,
                schema_version,
                csv_path,
                seeded_at,
                total_csv_rows,
                unique_source_video_count,
                seeded_video_count,
                excluded_gte_30_count,
                excluded_non_latin_title_count,
                excluded_non_numeric_duration_count,
                excluded_duration_conflict_count,
                downloaded_no_heatmap_deleted_count,
                downloaded_no_heatmap_title_deleted_count,
                existing_done_count,
                queued_downloaded_count,
                not_downloaded_count
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                SCHEMA_VERSION,
                str(csv_path),
                seeded_at,
                summary.total_csv_rows,
                summary.unique_source_video_count,
                summary.seeded_video_count,
                summary.excluded_gte_30_count,
                summary.excluded_non_latin_title_count,
                summary.excluded_non_numeric_duration_count,
                summary.excluded_duration_conflict_count,
                summary.downloaded_no_heatmap_deleted_count,
                summary.downloaded_no_heatmap_title_deleted_count,
                summary.existing_done_count,
                summary.queued_downloaded_count,
                summary.not_downloaded_count,
            ),
        )
        connection.commit()

    return summary


def _print_summary(summary: SeedSummary) -> None:
    print(f"Created SQLite state DB: {summary.db_path}")
    print(f"Seeded from CSV: {summary.csv_path}")
    print(f"Seeded at: {summary.seeded_at}")
    print()
    print(f"Total CSV rows: {summary.total_csv_rows:,}")
    print(f"Unique source videos: {summary.unique_source_video_count:,}")
    print(f"Seeded videos after filters: {summary.seeded_video_count:,}")
    print(f"Excluded >=30 min: {summary.excluded_gte_30_count:,}")
    print(f"Excluded non-Latin title ratio: {summary.excluded_non_latin_title_count:,}")
    print(
        "Excluded non-numeric duration: "
        f"{summary.excluded_non_numeric_duration_count:,}"
    )
    print(
        "Excluded duration conflicts: "
        f"{summary.excluded_duration_conflict_count:,}"
    )
    print()
    print(
        "Downloaded/no-heatmap deleted by duration or title filters: "
        f"{summary.downloaded_no_heatmap_deleted_count:,}"
    )
    print(
        "Downloaded/no-heatmap deleted by title filter: "
        f"{summary.downloaded_no_heatmap_title_deleted_count:,}"
    )
    print()
    print(f"Existing done: {summary.existing_done_count:,}")
    print(f"Queued downloaded: {summary.queued_downloaded_count:,}")
    print(f"Waiting source download: {summary.not_downloaded_count:,}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m youtube_decompose.job_state.state",
        description="Build and inspect the local SQLite video state database.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init-db",
        help="Create a clean SQLite state DB from the v11 CSV.",
    )
    init_parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Input v11 CSV path. Defaults to {DEFAULT_CSV_PATH}.",
    )
    init_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Output SQLite DB path. Defaults to {DEFAULT_DB_PATH}.",
    )
    init_parser.add_argument(
        "--max-duration-minutes",
        type=float,
        default=DEFAULT_MAX_DURATION_MINUTES,
        help=(
            "Only seed videos with duration strictly less than this value. "
            f"Defaults to {DEFAULT_MAX_DURATION_MINUTES}."
        ),
    )
    init_parser.add_argument(
        "--max-non-latin-title-ratio",
        type=float,
        default=DEFAULT_MAX_NON_LATIN_TITLE_RATIO,
        help=(
            "Only seed videos whose non-Latin title-letter ratio is less than "
            "or equal to this value. Defaults to "
            f"{DEFAULT_MAX_NON_LATIN_TITLE_RATIO}."
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init-db":
        summary = init_database(
            csv_path=args.csv,
            db_path=args.db,
            max_duration_minutes=args.max_duration_minutes,
            max_non_latin_title_ratio=args.max_non_latin_title_ratio,
        )
        _print_summary(summary)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
