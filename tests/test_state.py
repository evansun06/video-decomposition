from __future__ import annotations

import csv
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import youtube_decompose.state as legacy_state
from youtube_decompose.job_state.state import (
    DEFAULT_DB_PATH,
    SCHEMA_VERSION,
    init_database,
)
from youtube_decompose.job_state.util import (
    non_latin_title_ratio,
    should_exclude_title,
    title_letter_counts,
)


FIELDNAMES = [
    "id",
    "path",
    "title",
    "is_downloaded",
    "has_heatmap",
    "contentDetails.duration",
]


def video_row(
    video_id: str,
    *,
    title: str = "English title",
    path: str | None = None,
    downloaded: str = "downloaded",
    heatmap: str = "False",
    duration: str = "1",
) -> dict[str, str]:
    return {
        "id": video_id,
        "path": path if path is not None else "\\\\nas\\" + video_id + ".mp4",
        "title": title,
        "is_downloaded": downloaded,
        "has_heatmap": heatmap,
        "contentDetails.duration": duration,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class TitleFilterTests(unittest.TestCase):
    def test_non_latin_ratio_ignores_non_letters(self) -> None:
        self.assertEqual(title_letter_counts("Funny video 😂 123!"), (10, 0))
        self.assertEqual(non_latin_title_ratio("💰🚀 123"), 0.0)

    def test_non_latin_ratio_keeps_latin_variants(self) -> None:
        self.assertFalse(should_exclude_title("Café español über 😂"))
        self.assertFalse(should_exclude_title("𝐓𝐡𝐞 𝟔 𝐁𝐞𝐬𝐭"))

    def test_non_latin_ratio_uses_strict_threshold(self) -> None:
        self.assertFalse(should_exclude_title("abcd世"))
        self.assertTrue(should_exclude_title("abc世"))
        self.assertTrue(should_exclude_title("한국 브이로그"))


class StateDatabaseTests(unittest.TestCase):
    def test_legacy_state_module_remains_compatible(self) -> None:
        self.assertIs(legacy_state.init_database, init_database)
        self.assertEqual(legacy_state.DEFAULT_DB_PATH, DEFAULT_DB_PATH)

    def test_init_database_seeds_statuses_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "v11.csv"
            db_path = temp_path / "state.sqlite"
            write_csv(
                csv_path,
                [
                    video_row(
                        "done",
                        heatmap="True",
                        duration="12.5",
                    ),
                    video_row(
                        "done",
                        heatmap="True",
                        duration="12.5",
                    ),
                    video_row("queued"),
                    video_row(
                        "waiting",
                        downloaded="not downloaded",
                        duration="29.999",
                    ),
                    video_row("exactly_30", duration="30.0"),
                    video_row("over_30", duration="31"),
                    video_row(
                        "non_latin_title",
                        title="abc世界",
                        duration="10",
                    ),
                    video_row("bad_duration", duration="P0D"),
                ],
            )

            summary = init_database(csv_path=csv_path, db_path=db_path)

            self.assertEqual(summary.seeded_video_count, 3)
            self.assertEqual(summary.existing_done_count, 1)
            self.assertEqual(summary.queued_downloaded_count, 1)
            self.assertEqual(summary.not_downloaded_count, 1)
            self.assertEqual(summary.excluded_gte_30_count, 2)
            self.assertEqual(summary.excluded_non_latin_title_count, 1)
            self.assertEqual(summary.excluded_non_numeric_duration_count, 1)
            self.assertEqual(summary.downloaded_no_heatmap_deleted_count, 3)
            self.assertEqual(summary.downloaded_no_heatmap_title_deleted_count, 1)

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT
                        video_id,
                        source_row_count,
                        audio_status,
                        image_status,
                        transcription_status
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()
                metadata = connection.execute(
                    """
                    SELECT
                        schema_version,
                        excluded_non_latin_title_count,
                        downloaded_no_heatmap_deleted_count,
                        downloaded_no_heatmap_title_deleted_count
                    FROM state_metadata
                    """
                ).fetchone()

            self.assertEqual(
                rows,
                [
                    ("done", 2, "done", "done", "done"),
                    ("queued", 1, "queued", "queued", "waiting_audio"),
                    (
                        "waiting",
                        1,
                        "waiting_source",
                        "waiting_source",
                        "waiting_source",
                    ),
                ],
            )
            self.assertEqual(metadata, (SCHEMA_VERSION, 1, 3, 1))

    def test_duplicate_path_conflict_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "v11.csv"
            db_path = temp_path / "state.sqlite"
            write_csv(
                csv_path,
                [
                    video_row("same", path=r"\\nas\one.mp4"),
                    video_row("same", path=r"\\nas\two.mp4"),
                ],
            )

            with self.assertRaisesRegex(ValueError, "Conflicting 'path' values"):
                init_database(csv_path=csv_path, db_path=db_path)

    def test_duplicate_duration_conflict_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "v11.csv"
            db_path = temp_path / "state.sqlite"
            write_csv(
                csv_path,
                [
                    video_row("conflict", duration="1"),
                    video_row("conflict", duration="2"),
                ],
            )

            summary = init_database(csv_path=csv_path, db_path=db_path)

            self.assertEqual(summary.seeded_video_count, 0)
            self.assertEqual(summary.excluded_duration_conflict_count, 1)

    def test_duplicate_title_variants_use_max_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "v11.csv"
            db_path = temp_path / "state.sqlite"
            write_csv(
                csv_path,
                [
                    video_row("same", title="English title"),
                    video_row("same", title="한국 브이로그"),
                ],
            )

            summary = init_database(csv_path=csv_path, db_path=db_path)

            self.assertEqual(summary.seeded_video_count, 0)
            self.assertEqual(summary.excluded_non_latin_title_count, 1)
            self.assertEqual(summary.downloaded_no_heatmap_deleted_count, 1)
            self.assertEqual(summary.downloaded_no_heatmap_title_deleted_count, 1)

    def test_init_database_overwrites_existing_db(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "v11.csv"
            db_path = temp_path / "state.sqlite"
            write_csv(csv_path, [video_row("one")])

            with sqlite3.connect(db_path) as connection:
                connection.execute("CREATE TABLE old_table (value TEXT)")
                connection.commit()

            init_database(csv_path=csv_path, db_path=db_path)

            with sqlite3.connect(db_path) as connection:
                old_table = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name = 'old_table'
                    """
                ).fetchone()
                video_count = connection.execute(
                    "SELECT COUNT(*) FROM videos"
                ).fetchone()[0]

            self.assertIsNone(old_table)
            self.assertEqual(video_count, 1)


if __name__ == "__main__":
    unittest.main()
