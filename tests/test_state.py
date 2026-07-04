from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import youtube_decompose.state as legacy_state
from status_counts import build_rows
from youtube_decompose.job_state.state import (
    DEFAULT_DB_PATH,
    SCHEMA_VERSION,
    ensure_database_schema,
    init_database,
)
from youtube_decompose.poll_transcription_batches import (
    main as poll_batches_main,
    poll_transcription_batches,
)
from youtube_decompose.submit_audio_extraction import (
    main as audio_submit_main,
    submit_audio_extraction,
)
from youtube_decompose.submit_gcp_stt_batches import (
    batch_size as parse_stt_batch_size,
    submit_gcp_stt_batches,
)
from youtube_decompose.submit_image_decomposition import submit_image_decomposition
from youtube_decompose.google_speech import (
    GoogleTranscriptionResult,
    write_google_transcript_outputs,
)
from youtube_decompose.config import GoogleSpeechConfig
from youtube_decompose.job_state.util import (
    non_latin_title_ratio,
    should_exclude_title,
    title_letter_counts,
)
from youtube_decompose.stages import (
    run_audio_stage_for_video,
    run_image_stage_for_video,
    run_transcription_stage_for_video,
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

    def test_init_database_creates_batch_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "v11.csv"
            db_path = temp_path / "state.sqlite"
            write_csv(csv_path, [video_row("one")])

            init_database(csv_path=csv_path, db_path=db_path)

            with sqlite3.connect(db_path) as connection:
                video_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(videos)")
                }
                batch_table = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name = 'transcription_batches'
                    """
                ).fetchone()
                metadata = connection.execute(
                    "SELECT schema_version FROM state_metadata WHERE id = 1"
                ).fetchone()

            self.assertIn("transcription_batch_id", video_columns)
            self.assertIsNotNone(batch_table)
            self.assertEqual(metadata[0], SCHEMA_VERSION)

    def test_ensure_database_schema_upgrades_existing_db(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "state.sqlite"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    CREATE TABLE videos (
                        video_id TEXT PRIMARY KEY,
                        gcs_uri TEXT
                    );
                    CREATE TABLE state_metadata (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        schema_version INTEGER NOT NULL
                    );
                    INSERT INTO state_metadata (id, schema_version) VALUES (1, 2);
                    """
                )
                connection.commit()

            ensure_database_schema(db_path)
            ensure_database_schema(db_path)

            with sqlite3.connect(db_path) as connection:
                video_columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(videos)")
                }
                batch_table = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name = 'transcription_batches'
                    """
                ).fetchone()
                metadata = connection.execute(
                    "SELECT schema_version FROM state_metadata WHERE id = 1"
                ).fetchone()

            self.assertIn("transcription_batch_id", video_columns)
            self.assertIsNotNone(batch_table)
            self.assertEqual(metadata[0], SCHEMA_VERSION)


class StagePipelineTests(unittest.TestCase):
    def _seed_one_video(
        self,
        temp_path: Path,
        *,
        video_id: str = "queued",
        path: str | None = None,
    ) -> tuple[Path, Path]:
        csv_path = temp_path / "v11.csv"
        db_path = temp_path / "state.sqlite"
        write_csv(csv_path, [video_row(video_id, path=path)])
        init_database(csv_path=csv_path, db_path=db_path)
        return db_path, temp_path / "output"

    def test_audio_stage_writes_output_path_and_promotes_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_one_video(temp_path)

            def fake_convert(video_path: str, work_dir: Path) -> Path:
                self.assertEqual(video_path, r"\\nas\queued.mp4")
                audio_dir = Path(work_dir) / "audio_temp"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / "audio_full.wav"
                audio_path.write_bytes(b"wav")
                return audio_path

            with patch(
                "youtube_decompose.stages.convert_video_to_audio",
                side_effect=fake_convert,
            ):
                audio_path = run_audio_stage_for_video(
                    "queued",
                    db_path=db_path,
                    output_root=output_root,
                )

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """
                    SELECT
                        audio_status,
                        audio_output_path,
                        audio_attempts,
                        audio_error,
                        transcription_status
                    FROM videos
                    WHERE video_id = 'queued'
                    """
                ).fetchone()

            self.assertEqual(
                audio_path,
                output_root / "queued" / "audio_temp" / "audio_full.wav",
            )
            self.assertEqual(
                row,
                (
                    "done",
                    str(audio_path),
                    1,
                    None,
                    "queued",
                ),
            )

    def test_transcription_stage_records_google_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_one_video(temp_path)
            audio_path = output_root / "queued" / "audio_temp" / "audio_full.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"wav")

            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    UPDATE videos
                    SET
                        audio_status = 'done',
                        audio_output_path = ?,
                        transcription_status = 'queued'
                    WHERE video_id = 'queued'
                    """,
                    (str(audio_path),),
                )
                connection.commit()

            def fake_transcribe(
                audio_path: Path,
                result_dir: Path,
                config: object,
            ) -> GoogleTranscriptionResult:
                result_dir = Path(result_dir)
                result_dir.mkdir(parents=True, exist_ok=True)
                transcript_path = result_dir / "script_google.txt"
                text_panel_path = result_dir / "text_panel_google.csv"
                sentence_panel_path = result_dir / "google_sentence_panel.csv"
                transcript_path.write_text("hello world", encoding="utf-8")
                text_panel_path.write_text("Text\nhello\n", encoding="utf-8")
                sentence_panel_path.write_text("Text\nhello world\n", encoding="utf-8")
                return GoogleTranscriptionResult(
                    transcript_text="hello world",
                    transcript_path=transcript_path,
                    text_panel_path=text_panel_path,
                    sentence_panel_path=sentence_panel_path,
                    gcs_uri="gs://bucket/audio.wav",
                )

            with patch(
                "youtube_decompose.stages.transcribe_audio_with_google",
                side_effect=fake_transcribe,
            ):
                result = run_transcription_stage_for_video(
                    "queued",
                    google_config=object(),
                    db_path=db_path,
                    output_root=output_root,
                )

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """
                    SELECT
                        transcription_status,
                        transcript_path,
                        text_panel_path,
                        sentence_panel_path,
                        gcs_uri,
                        transcription_attempts,
                        transcription_error
                    FROM videos
                    WHERE video_id = 'queued'
                    """
                ).fetchone()

            self.assertEqual(result.transcript_text, "hello world")
            self.assertEqual(row[0], "done")
            self.assertEqual(
                row[1],
                str(output_root / "queued" / "result_temp" / "script_google.txt"),
            )
            self.assertEqual(
                row[2],
                str(output_root / "queued" / "result_temp" / "text_panel_google.csv"),
            )
            self.assertEqual(
                row[3],
                str(
                    output_root
                    / "queued"
                    / "result_temp"
                    / "google_sentence_panel.csv"
                ),
            )
            self.assertEqual(row[4:], ("gs://bucket/audio.wav", 1, None))

    def test_image_stage_records_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_one_video(temp_path)

            def fake_convert(video_path: str, work_dir: Path, frame_rate: int) -> int:
                self.assertEqual(video_path, r"\\nas\queued.mp4")
                self.assertEqual(frame_rate, 3)
                image_dir = Path(work_dir) / "image_temp"
                image_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / "image_split-1.png").write_bytes(b"png")
                (image_dir / "image_split-2.png").write_bytes(b"png")
                return 2

            with patch(
                "youtube_decompose.stages.convert_video_to_images",
                side_effect=fake_convert,
            ):
                frame_count = run_image_stage_for_video(
                    "queued",
                    db_path=db_path,
                    output_root=output_root,
                    frame_rate=3,
                )

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """
                    SELECT
                        image_status,
                        image_output_dir,
                        frame_count,
                        image_attempts,
                        image_error
                    FROM videos
                    WHERE video_id = 'queued'
                    """
                ).fetchone()

            self.assertEqual(frame_count, 2)
            self.assertEqual(
                row,
                (
                    "done",
                    str(output_root / "queued" / "image_temp"),
                    2,
                    1,
                    None,
                ),
            )

    def test_blank_source_path_marks_audio_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_one_video(temp_path, path="")

            with self.assertRaisesRegex(ValueError, "Source NAS path is blank"):
                run_audio_stage_for_video(
                    "queued",
                    db_path=db_path,
                    output_root=output_root,
                )

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """
                    SELECT audio_status, audio_attempts, audio_error
                    FROM videos
                    WHERE video_id = 'queued'
                    """
                ).fetchone()

            self.assertEqual(row[0], "failed")
            self.assertEqual(row[1], 1)
            self.assertIn("Source NAS path is blank", row[2])

    def test_transcription_missing_audio_marks_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_one_video(temp_path)
            missing_audio = output_root / "queued" / "audio_temp" / "audio_full.wav"

            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    UPDATE videos
                    SET
                        audio_status = 'done',
                        audio_output_path = ?,
                        transcription_status = 'queued'
                    WHERE video_id = 'queued'
                    """,
                    (str(missing_audio),),
                )
                connection.commit()

            with self.assertRaisesRegex(ValueError, "Audio output does not exist"):
                run_transcription_stage_for_video(
                    "queued",
                    google_config=object(),
                    db_path=db_path,
                    output_root=output_root,
                )

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """
                    SELECT
                        transcription_status,
                        transcription_attempts,
                        transcription_error
                    FROM videos
                    WHERE video_id = 'queued'
                    """
                ).fetchone()

            self.assertEqual(row[0], "failed")
            self.assertEqual(row[1], 1)
            self.assertIn("Audio output does not exist", row[2])

    def test_unknown_video_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_one_video(temp_path)

            with self.assertRaisesRegex(ValueError, "Unknown video_id"):
                run_image_stage_for_video(
                    "missing",
                    db_path=db_path,
                    output_root=output_root,
                )


class SubmissionScriptTests(unittest.TestCase):
    def _seed_videos(
        self,
        temp_path: Path,
        video_ids: list[str],
    ) -> tuple[Path, Path]:
        csv_path = temp_path / "v11.csv"
        db_path = temp_path / "state.sqlite"
        output_root = temp_path / "output"
        write_csv(csv_path, [video_row(video_id) for video_id in video_ids])
        init_database(csv_path=csv_path, db_path=db_path)
        return db_path, output_root

    def test_audio_submit_limit_one_processes_one_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a", "b"])

            def fake_convert(video_path: str, work_dir: Path) -> Path:
                audio_dir = Path(work_dir) / "audio_temp"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / "audio_full.wav"
                audio_path.write_bytes(b"wav")
                return audio_path

            with patch(
                "youtube_decompose.stages.convert_video_to_audio",
                side_effect=fake_convert,
            ):
                summary = submit_audio_extraction(
                    db_path=db_path,
                    output_root=output_root,
                    limit=1,
                    workers=1,
                )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT video_id, audio_status, audio_output_path
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()

            self.assertEqual(summary.selected, 1)
            self.assertEqual(summary.succeeded, 1)
            self.assertEqual(rows[0][0:2], ("a", "done"))
            self.assertEqual(rows[1][0:2], ("b", "queued"))
            self.assertEqual(
                rows[0][2],
                str(output_root / "a" / "audio_temp" / "audio_full.wav"),
            )

    def test_audio_submit_without_limit_processes_all_eligible_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a", "b", "c"])

            def fake_convert(video_path: str, work_dir: Path) -> Path:
                audio_dir = Path(work_dir) / "audio_temp"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / "audio_full.wav"
                audio_path.write_bytes(b"wav")
                return audio_path

            with patch(
                "youtube_decompose.stages.convert_video_to_audio",
                side_effect=fake_convert,
            ):
                summary = submit_audio_extraction(
                    db_path=db_path,
                    output_root=output_root,
                    workers=2,
                )

            with sqlite3.connect(db_path) as connection:
                statuses = connection.execute(
                    """
                    SELECT audio_status, COUNT(*)
                    FROM videos
                    GROUP BY audio_status
                    """
                ).fetchall()

            self.assertEqual(summary.selected, 3)
            self.assertEqual(summary.succeeded, 3)
            self.assertEqual(statuses, [("done", 3)])

    def test_audio_submit_retry_failed_includes_failed_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a", "b"])
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "UPDATE videos SET audio_status = 'failed' WHERE video_id = 'b'"
                )
                connection.commit()

            def fake_convert(video_path: str, work_dir: Path) -> Path:
                audio_dir = Path(work_dir) / "audio_temp"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / "audio_full.wav"
                audio_path.write_bytes(b"wav")
                return audio_path

            with patch(
                "youtube_decompose.stages.convert_video_to_audio",
                side_effect=fake_convert,
            ):
                summary = submit_audio_extraction(
                    db_path=db_path,
                    output_root=output_root,
                    retry_failed=True,
                    workers=1,
                )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT video_id, audio_status
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()

            self.assertEqual(summary.selected, 2)
            self.assertEqual(summary.succeeded, 2)
            self.assertEqual(rows, [("a", "done"), ("b", "done")])

    def test_audio_submit_without_retry_failed_skips_failed_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a", "b"])
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "UPDATE videos SET audio_status = 'failed' WHERE video_id = 'b'"
                )
                connection.commit()

            def fake_convert(video_path: str, work_dir: Path) -> Path:
                audio_dir = Path(work_dir) / "audio_temp"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / "audio_full.wav"
                audio_path.write_bytes(b"wav")
                return audio_path

            with patch(
                "youtube_decompose.stages.convert_video_to_audio",
                side_effect=fake_convert,
            ):
                summary = submit_audio_extraction(
                    db_path=db_path,
                    output_root=output_root,
                    workers=1,
                )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT video_id, audio_status
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()

            self.assertEqual(summary.selected, 1)
            self.assertEqual(rows, [("a", "done"), ("b", "failed")])

    def test_image_submit_uses_frame_rate_and_video_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a"])

            def fake_convert(video_path: str, work_dir: Path, frame_rate: int) -> int:
                self.assertEqual(frame_rate, 7)
                image_dir = Path(work_dir) / "image_temp"
                image_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / "image_split-1.png").write_bytes(b"png")
                return 1

            with patch(
                "youtube_decompose.stages.convert_video_to_images",
                side_effect=fake_convert,
            ):
                summary = submit_image_decomposition(
                    db_path=db_path,
                    output_root=output_root,
                    frame_rate=7,
                    workers=1,
                )

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """
                    SELECT image_status, image_output_dir, frame_count
                    FROM videos
                    WHERE video_id = 'a'
                    """
                ).fetchone()

            self.assertEqual(summary.selected, 1)
            self.assertEqual(row, ("done", str(output_root / "a" / "image_temp"), 1))

    def test_image_submit_retry_failed_includes_failed_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a", "b"])
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "UPDATE videos SET image_status = 'failed' WHERE video_id = 'b'"
                )
                connection.commit()

            def fake_convert(video_path: str, work_dir: Path, frame_rate: int) -> int:
                image_dir = Path(work_dir) / "image_temp"
                image_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / "image_split-1.png").write_bytes(b"png")
                return 1

            with patch(
                "youtube_decompose.stages.convert_video_to_images",
                side_effect=fake_convert,
            ):
                summary = submit_image_decomposition(
                    db_path=db_path,
                    output_root=output_root,
                    retry_failed=True,
                    workers=1,
                )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT video_id, image_status
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()

            self.assertEqual(summary.selected, 2)
            self.assertEqual(summary.succeeded, 2)
            self.assertEqual(rows, [("a", "done"), ("b", "done")])

    def test_image_submit_without_retry_failed_skips_failed_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a", "b"])
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "UPDATE videos SET image_status = 'failed' WHERE video_id = 'b'"
                )
                connection.commit()

            def fake_convert(video_path: str, work_dir: Path, frame_rate: int) -> int:
                image_dir = Path(work_dir) / "image_temp"
                image_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / "image_split-1.png").write_bytes(b"png")
                return 1

            with patch(
                "youtube_decompose.stages.convert_video_to_images",
                side_effect=fake_convert,
            ):
                summary = submit_image_decomposition(
                    db_path=db_path,
                    output_root=output_root,
                    workers=1,
                )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT video_id, image_status
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()

            self.assertEqual(summary.selected, 1)
            self.assertEqual(rows, [("a", "done"), ("b", "failed")])

    def test_audio_submit_cli_log_file_includes_video_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a"])
            log_path = temp_path / "submit.log"

            def fake_convert(video_path: str, work_dir: Path) -> Path:
                audio_dir = Path(work_dir) / "audio_temp"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / "audio_full.wav"
                audio_path.write_bytes(b"wav")
                return audio_path

            with patch(
                "youtube_decompose.stages.convert_video_to_audio",
                side_effect=fake_convert,
            ):
                exit_code = audio_submit_main(
                    [
                        "--db",
                        str(db_path),
                        "--output-root",
                        str(output_root),
                        "--limit",
                        "1",
                        "--log-file",
                        str(log_path),
                    ]
                )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertIn("video_id=a", log_text)
            self.assertIn("Reading source video", log_text)

    def test_audio_submit_cli_retry_failed_includes_failed_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_videos(temp_path, ["a"])
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "UPDATE videos SET audio_status = 'failed' WHERE video_id = 'a'"
                )
                connection.commit()

            def fake_convert(video_path: str, work_dir: Path) -> Path:
                audio_dir = Path(work_dir) / "audio_temp"
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path = audio_dir / "audio_full.wav"
                audio_path.write_bytes(b"wav")
                return audio_path

            with patch(
                "youtube_decompose.stages.convert_video_to_audio",
                side_effect=fake_convert,
            ):
                exit_code = audio_submit_main(
                    [
                        "--db",
                        str(db_path),
                        "--output-root",
                        str(output_root),
                        "--retry-failed",
                    ]
                )

            with sqlite3.connect(db_path) as connection:
                status = connection.execute(
                    "SELECT audio_status FROM videos WHERE video_id = 'a'"
                ).fetchone()[0]

            self.assertEqual(exit_code, 0)
            self.assertEqual(status, "done")


class StatusCountsTests(unittest.TestCase):
    def test_build_rows_includes_zero_count_video_and_batch_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "v11.csv"
            db_path = temp_path / "state.sqlite"
            write_csv(csv_path, [video_row("a")])
            init_database(csv_path=csv_path, db_path=db_path)

            rows = build_rows(db_path)

        self.assertIn(("audio", "queued", 1), rows)
        self.assertIn(("audio", "done", 0), rows)
        self.assertIn(("image", "failed", 0), rows)
        self.assertIn(("transcription", "waiting_audio", 1), rows)
        self.assertIn(("transcription", "queued", 0), rows)
        self.assertIn(("stt_batch", "submitted", 0), rows)
        self.assertIn(("stt_batch", "done", 0), rows)
        self.assertIn(("stt_batch", "failed", 0), rows)


class SttSubmissionTests(unittest.TestCase):
    def test_stt_submit_rejects_batch_size_above_google_limit(self) -> None:
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "at most 5"):
            parse_stt_batch_size("6")

    def _seed_audio_done(
        self,
        temp_path: Path,
        video_ids: list[str],
        *,
        missing_audio_ids: set[str] | None = None,
    ) -> tuple[Path, Path]:
        missing_audio_ids = missing_audio_ids or set()
        csv_path = temp_path / "v11.csv"
        db_path = temp_path / "state.sqlite"
        output_root = temp_path / "output"
        write_csv(csv_path, [video_row(video_id) for video_id in video_ids])
        init_database(csv_path=csv_path, db_path=db_path)

        with sqlite3.connect(db_path) as connection:
            for video_id in video_ids:
                audio_path = output_root / video_id / "audio_temp" / "audio_full.wav"
                if video_id not in missing_audio_ids:
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    audio_path.write_bytes(b"wav")
                connection.execute(
                    """
                    UPDATE videos
                    SET
                        audio_status = 'done',
                        audio_output_path = ?,
                        transcription_status = 'queued'
                    WHERE video_id = ?
                    """,
                    (str(audio_path), video_id),
                )
            connection.commit()

        return db_path, output_root

    def test_stt_submit_upgrades_schema_and_records_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_audio_done(temp_path, ["a"])

            with sqlite3.connect(db_path) as connection:
                connection.execute("DROP TABLE transcription_batches")
                connection.commit()

            def fake_upload(
                audio_path: Path,
                config: GoogleSpeechConfig,
                object_name: str | None = None,
            ) -> str:
                return f"gs://bucket/{object_name}"

            submitted: list[tuple[tuple[str, ...], str]] = []

            def fake_submit(gcs_uris: tuple[str, ...], output_uri: str) -> str:
                submitted.append((gcs_uris, output_uri))
                return "operations/abc"

            summary = submit_gcp_stt_batches(
                db_path=db_path,
                output_root=output_root,
                batch_size=1,
                google_config=GoogleSpeechConfig(
                    bucket_name="bucket",
                    project_id="project",
                ),
                upload_func=fake_upload,
                batch_submitter=fake_submit,
            )

            with sqlite3.connect(db_path) as connection:
                batch = connection.execute(
                    """
                    SELECT operation_name, status, gcs_uris_json
                    FROM transcription_batches
                    """
                ).fetchone()
                video = connection.execute(
                    """
                    SELECT transcription_status, transcription_batch_id, gcs_uri
                    FROM videos
                    WHERE video_id = 'a'
                    """
                ).fetchone()

            self.assertEqual(summary.batches_submitted, 1)
            self.assertEqual(batch[0:2], ("operations/abc", "submitted"))
            self.assertEqual(json.loads(batch[2]), [video[2]])
            self.assertEqual(video[0], "running")
            self.assertIsNotNone(video[1])
            self.assertTrue(video[2].startswith("gs://bucket/a_audio_full_"))
            self.assertEqual(len(submitted), 1)
            self.assertTrue(submitted[0][1].startswith("gs://bucket/stt_results/"))

    def test_stt_submit_chunks_uploaded_audio_by_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_audio_done(temp_path, ["a", "b", "c"])

            def fake_upload(
                audio_path: Path,
                config: GoogleSpeechConfig,
                object_name: str | None = None,
            ) -> str:
                return f"gs://bucket/{object_name}"

            submitted: list[tuple[str, ...]] = []

            def fake_submit(gcs_uris: tuple[str, ...], output_uri: str) -> str:
                submitted.append(gcs_uris)
                return f"operations/{len(submitted)}"

            summary = submit_gcp_stt_batches(
                db_path=db_path,
                output_root=output_root,
                batch_size=2,
                google_config=GoogleSpeechConfig(
                    bucket_name="bucket",
                    project_id="project",
                ),
                upload_func=fake_upload,
                batch_submitter=fake_submit,
            )

            with sqlite3.connect(db_path) as connection:
                batch_count = connection.execute(
                    "SELECT COUNT(*) FROM transcription_batches"
                ).fetchone()[0]

            self.assertEqual(summary.selected, 3)
            self.assertEqual(summary.batches_submitted, 2)
            self.assertEqual([len(batch) for batch in submitted], [2, 1])
            self.assertEqual(batch_count, 2)

    def test_stt_submit_skips_videos_gte_20_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_audio_done(temp_path, ["a", "b"])

            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    "UPDATE videos SET duration_minutes = 20.0 WHERE video_id = 'b'"
                )
                connection.commit()

            def fake_upload(
                audio_path: Path,
                config: GoogleSpeechConfig,
                object_name: str | None = None,
            ) -> str:
                return f"gs://bucket/{object_name}"

            submitted: list[tuple[str, ...]] = []

            def fake_submit(gcs_uris: tuple[str, ...], output_uri: str) -> str:
                submitted.append(gcs_uris)
                return "operations/abc"

            summary = submit_gcp_stt_batches(
                db_path=db_path,
                output_root=output_root,
                batch_size=5,
                google_config=GoogleSpeechConfig(
                    bucket_name="bucket",
                    project_id="project",
                ),
                upload_func=fake_upload,
                batch_submitter=fake_submit,
            )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT video_id, transcription_status
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()

            self.assertEqual(summary.selected, 1)
            self.assertEqual(len(submitted), 1)
            self.assertIn("/a_audio_full_", submitted[0][0])
            self.assertEqual(rows[0], ("a", "running"))
            self.assertEqual(rows[1], ("b", "queued"))

    def test_stt_submit_missing_audio_marks_only_that_video_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_audio_done(
                temp_path,
                ["a", "b"],
                missing_audio_ids={"b"},
            )

            def fake_upload(
                audio_path: Path,
                config: GoogleSpeechConfig,
                object_name: str | None = None,
            ) -> str:
                return f"gs://bucket/{object_name}"

            summary = submit_gcp_stt_batches(
                db_path=db_path,
                output_root=output_root,
                batch_size=2,
                google_config=GoogleSpeechConfig(
                    bucket_name="bucket",
                    project_id="project",
                ),
                upload_func=fake_upload,
                batch_submitter=lambda _uris, _output_uri: "operations/abc",
            )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT video_id, transcription_status, transcription_error
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()

            self.assertEqual(summary.upload_failed, 1)
            self.assertEqual(rows[0], ("a", "running", None))
            self.assertEqual(rows[1][0:2], ("b", "failed"))
            self.assertIn("Audio output does not exist", rows[1][2])


def fake_transcript(*words: str) -> SimpleNamespace:
    word_infos = []
    for index, word in enumerate(words):
        word_infos.append(
            SimpleNamespace(
                word=word,
                start_offset=SimpleNamespace(seconds=index, nanos=0),
                end_offset=SimpleNamespace(seconds=index + 1, nanos=0),
            )
        )
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        words=word_infos,
                    )
                ]
            )
        ]
    )


def fake_transcript_text_only(*parts: str) -> SimpleNamespace:
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        transcript=part,
                        words=[],
                    )
                ]
            )
            for part in parts
        ]
    )


def fake_operation(
    *,
    done: bool = True,
    response: object | None = None,
    error: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(done=done, response=response, error=error)


class BatchPollingTests(unittest.TestCase):
    def _seed_batch(
        self,
        temp_path: Path,
        *,
        video_ids: list[str],
        gcs_uris: list[str],
    ) -> tuple[Path, Path]:
        csv_path = temp_path / "v11.csv"
        db_path = temp_path / "state.sqlite"
        output_root = temp_path / "output"
        write_csv(csv_path, [video_row(video_id) for video_id in video_ids])
        init_database(csv_path=csv_path, db_path=db_path)

        with sqlite3.connect(db_path) as connection:
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
                VALUES (
                    'batch-1',
                    'operations/abc',
                    'submitted',
                    ?,
                    '2026-01-01T00:00:00+00:00',
                    '2026-01-01T00:00:00+00:00',
                    '2026-01-01T00:00:00+00:00'
                )
                """,
                (json.dumps(gcs_uris),),
            )
            for video_id, gcs_uri in zip(video_ids, gcs_uris):
                connection.execute(
                    """
                    UPDATE videos
                    SET
                        audio_status = 'done',
                        transcription_status = 'running',
                        transcription_batch_id = 'batch-1',
                        gcs_uri = ?
                    WHERE video_id = ?
                    """,
                    (gcs_uri, video_id),
                )
            connection.commit()

        return db_path, output_root

    def test_completed_batch_writes_outputs_and_marks_done(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gcs_uris = ["gs://bucket/a.wav", "gs://bucket/b.wav"]
            db_path, output_root = self._seed_batch(
                temp_path,
                video_ids=["a", "b"],
                gcs_uris=gcs_uris,
            )
            response = SimpleNamespace(
                results={
                    gcs_uris[0]: SimpleNamespace(
                        transcript=fake_transcript("hello", "world.")
                    ),
                    gcs_uris[1]: SimpleNamespace(
                        transcript=fake_transcript("second", "video.")
                    ),
                }
            )

            summary = poll_transcription_batches(
                db_path=db_path,
                output_root=output_root,
                operation_getter=lambda _name: fake_operation(response=response),
            )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT
                        video_id,
                        transcription_status,
                        transcript_path,
                        text_panel_path,
                        sentence_panel_path,
                        transcription_error
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()
                batch = connection.execute(
                    "SELECT status, error FROM transcription_batches"
                ).fetchone()

            self.assertEqual(summary.done_batches, 1)
            self.assertEqual(summary.done_videos, 2)
            self.assertEqual(summary.failed_videos, 0)
            self.assertEqual(batch, ("done", None))
            for row in rows:
                self.assertEqual(row[1], "done")
                self.assertIsNone(row[5])
                self.assertTrue(Path(row[2]).exists())
                self.assertTrue(Path(row[3]).exists())
                self.assertTrue(Path(row[4]).exists())

    def test_repoll_completed_batch_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gcs_uri = "gs://bucket/a.wav"
            db_path, output_root = self._seed_batch(
                temp_path,
                video_ids=["a"],
                gcs_uris=[gcs_uri],
            )
            response = SimpleNamespace(
                results={
                    gcs_uri: SimpleNamespace(transcript=fake_transcript("hello."))
                }
            )

            poll_transcription_batches(
                db_path=db_path,
                output_root=output_root,
                operation_getter=lambda _name: fake_operation(response=response),
            )

            def fail_if_called(_name: str) -> SimpleNamespace:
                raise AssertionError("completed batches should not be polled again")

            summary = poll_transcription_batches(
                db_path=db_path,
                output_root=output_root,
                operation_getter=fail_if_called,
            )

            self.assertEqual(summary.checked_batches, 0)
            self.assertEqual(summary.done_videos, 0)

    def test_incomplete_batch_stays_submitted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, _output_root = self._seed_batch(
                temp_path,
                video_ids=["a"],
                gcs_uris=["gs://bucket/a.wav"],
            )

            summary = poll_transcription_batches(
                db_path=db_path,
                output_root=temp_path / "output",
                operation_getter=lambda _name: fake_operation(done=False),
            )

            with sqlite3.connect(db_path) as connection:
                batch = connection.execute(
                    "SELECT status, error FROM transcription_batches"
                ).fetchone()
                video = connection.execute(
                    "SELECT transcription_status FROM videos WHERE video_id = 'a'"
                ).fetchone()

            self.assertEqual(summary.pending_batches, 1)
            self.assertEqual(batch, ("submitted", None))
            self.assertEqual(video[0], "running")

    def test_materialization_error_keeps_batch_submitted_for_repoll(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_root = temp_path / "output"
            output_root.write_text("not a directory", encoding="utf-8")
            gcs_uri = "gs://bucket/a.wav"
            db_path, _output_root = self._seed_batch(
                temp_path,
                video_ids=["a"],
                gcs_uris=[gcs_uri],
            )
            response = SimpleNamespace(
                results={
                    gcs_uri: SimpleNamespace(transcript=fake_transcript("hello."))
                }
            )

            with self.assertRaises(OSError):
                poll_transcription_batches(
                    db_path=db_path,
                    output_root=output_root,
                    operation_getter=lambda _name: fake_operation(response=response),
                )

            with sqlite3.connect(db_path) as connection:
                batch = connection.execute(
                    "SELECT status, error FROM transcription_batches"
                ).fetchone()
                video = connection.execute(
                    "SELECT transcription_status FROM videos WHERE video_id = 'a'"
                ).fetchone()

            self.assertEqual(batch[0], "submitted")
            self.assertIsNotNone(batch[1])
            self.assertEqual(video[0], "running")

            output_root.unlink()
            summary = poll_transcription_batches(
                db_path=db_path,
                output_root=output_root,
                operation_getter=lambda _name: fake_operation(response=response),
            )

            with sqlite3.connect(db_path) as connection:
                batch = connection.execute(
                    "SELECT status, error FROM transcription_batches"
                ).fetchone()
                video = connection.execute(
                    "SELECT transcription_status FROM videos WHERE video_id = 'a'"
                ).fetchone()

            self.assertEqual(summary.done_batches, 1)
            self.assertEqual(summary.done_videos, 1)
            self.assertEqual(batch, ("done", None))
            self.assertEqual(video[0], "done")

    def test_batch_level_error_marks_batch_and_video_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path, output_root = self._seed_batch(
                temp_path,
                video_ids=["a"],
                gcs_uris=["gs://bucket/a.wav"],
            )
            error = SimpleNamespace(code=13, message="operation failed")

            summary = poll_transcription_batches(
                db_path=db_path,
                output_root=output_root,
                operation_getter=lambda _name: fake_operation(error=error),
            )

            with sqlite3.connect(db_path) as connection:
                batch = connection.execute(
                    "SELECT status, error FROM transcription_batches"
                ).fetchone()
                video = connection.execute(
                    """
                    SELECT transcription_status, transcription_error
                    FROM videos
                    WHERE video_id = 'a'
                    """
                ).fetchone()

            self.assertEqual(summary.failed_batches, 1)
            self.assertEqual(summary.failed_videos, 1)
            self.assertEqual(batch, ("failed", "operation failed"))
            self.assertEqual(video, ("failed", "operation failed"))

    def test_file_level_error_marks_only_that_video_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gcs_uris = ["gs://bucket/a.wav", "gs://bucket/b.wav"]
            db_path, output_root = self._seed_batch(
                temp_path,
                video_ids=["a", "b"],
                gcs_uris=gcs_uris,
            )
            response = SimpleNamespace(
                results={
                    gcs_uris[0]: SimpleNamespace(
                        transcript=fake_transcript("hello.")
                    ),
                    gcs_uris[1]: SimpleNamespace(
                        error=SimpleNamespace(code=3, message="bad audio")
                    ),
                }
            )

            summary = poll_transcription_batches(
                db_path=db_path,
                output_root=output_root,
                operation_getter=lambda _name: fake_operation(response=response),
            )

            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    """
                    SELECT video_id, transcription_status, transcription_error
                    FROM videos
                    ORDER BY video_id
                    """
                ).fetchall()
                batch = connection.execute(
                    "SELECT status, error FROM transcription_batches"
                ).fetchone()

            self.assertEqual(summary.done_batches, 1)
            self.assertEqual(summary.done_videos, 1)
            self.assertEqual(summary.failed_videos, 1)
            self.assertEqual(batch, ("done", None))
            self.assertEqual(rows[0], ("a", "done", None))
            self.assertEqual(rows[1], ("b", "failed", "bad audio"))

    def test_completed_gcs_output_batch_downloads_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gcs_uri = "gs://bucket/a.wav"
            db_path, output_root = self._seed_batch(
                temp_path,
                video_ids=["a"],
                gcs_uris=[gcs_uri],
            )
            response = SimpleNamespace(
                results={
                    gcs_uri: SimpleNamespace(uri="gs://bucket/results/a.json"),
                }
            )

            with patch(
                "youtube_decompose.poll_transcription_batches."
                "download_google_batch_results_from_gcs",
                return_value=fake_transcript("from", "gcs."),
            ) as download_results:
                summary = poll_transcription_batches(
                    db_path=db_path,
                    output_root=output_root,
                    google_config=GoogleSpeechConfig(
                        bucket_name="bucket",
                        project_id="project",
                    ),
                    operation_getter=lambda _name: fake_operation(response=response),
                )

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """
                    SELECT transcription_status, transcript_path
                    FROM videos
                    WHERE video_id = 'a'
                    """
                ).fetchone()

            self.assertEqual(summary.done_videos, 1)
            self.assertEqual(row[0], "done")
            self.assertEqual(Path(row[1]).read_text(encoding="utf-8"), "from gcs.")
            download_results.assert_called_once()

    def test_transcript_text_falls_back_when_word_offsets_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)

            transcript_text, transcript_path, text_panel_path, sentence_panel_path = (
                write_google_transcript_outputs(
                    fake_transcript_text_only("hello world.", "second sentence."),
                    result_dir,
                )
            )

            self.assertEqual(transcript_text, "hello world. second sentence.")
            self.assertEqual(
                transcript_path.read_text(encoding="utf-8"),
                "hello world. second sentence.",
            )
            self.assertTrue(text_panel_path.exists())
            self.assertTrue(sentence_panel_path.exists())

    def test_poller_cli_one_pass_exits_with_no_submitted_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "v11.csv"
            db_path = temp_path / "state.sqlite"
            write_csv(csv_path, [video_row("a")])
            init_database(csv_path=csv_path, db_path=db_path)

            with patch("sys.stdout"):
                exit_code = poll_batches_main(
                    [
                        "--db",
                        str(db_path),
                        "--output-root",
                        str(temp_path / "output"),
                        "--limit",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
