from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_DB_PATH = Path("job_state/video_state.sqlite")
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from youtube_decompose.job_state.state import BATCH_STATUS_VALUES, StageStatus


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def status_counts(
    connection: sqlite3.Connection,
    *,
    area: str,
    table_name: str,
    status_column: str,
    statuses: Sequence[str] | None = None,
) -> list[tuple[str, str, int]]:
    counts = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            f"""
            SELECT {status_column}, COUNT(*)
            FROM {table_name}
            GROUP BY {status_column}
            ORDER BY {status_column}
            """
        ).fetchall()
    }
    if statuses is None:
        statuses = tuple(counts)

    rows = [(area, status, counts.get(status, 0)) for status in statuses]
    rows.extend(
        (area, status, count)
        for status, count in counts.items()
        if status not in statuses
    )
    return rows


def print_table(rows: list[tuple[str, str, int]]) -> None:
    headers = ("area", "status", "count")
    widths = [
        max(len(headers[0]), *(len(row[0]) for row in rows)),
        max(len(headers[1]), *(len(row[1]) for row in rows)),
        max(len(headers[2]), *(len(str(row[2])) for row in rows)),
    ]

    print(
        f"{headers[0]:<{widths[0]}}  "
        f"{headers[1]:<{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}"
    )
    print(
        f"{'-' * widths[0]}  "
        f"{'-' * widths[1]}  "
        f"{'-' * widths[2]}"
    )
    for area, status, count in rows:
        print(f"{area:<{widths[0]}}  {status:<{widths[1]}}  {count:>{widths[2]}}")


def build_rows(db_path: Path) -> list[tuple[str, str, int]]:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    with sqlite3.connect(db_path) as connection:
        if not table_exists(connection, "videos"):
            raise ValueError(f"SQLite DB does not contain a videos table: {db_path}")

        rows: list[tuple[str, str, int]] = []
        rows.extend(
            status_counts(
                connection,
                area="audio",
                table_name="videos",
                status_column="audio_status",
                statuses=tuple(status.value for status in StageStatus),
            )
        )
        rows.extend(
            status_counts(
                connection,
                area="image",
                table_name="videos",
                status_column="image_status",
                statuses=tuple(status.value for status in StageStatus),
            )
        )
        rows.extend(
            status_counts(
                connection,
                area="transcription",
                table_name="videos",
                status_column="transcription_status",
                statuses=tuple(status.value for status in StageStatus),
            )
        )

        if table_exists(connection, "transcription_batches"):
            rows.extend(
                status_counts(
                    connection,
                    area="stt_batch",
                    table_name="transcription_batches",
                    status_column="status",
                    statuses=BATCH_STATUS_VALUES,
                )
            )

    return sorted(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print status counts from the video analysis SQLite DB."
    )
    parser.add_argument(
        "db",
        nargs="?",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite DB path. Defaults to {DEFAULT_DB_PATH}.",
    )
    args = parser.parse_args(argv)

    try:
        rows = build_rows(args.db)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")

    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
