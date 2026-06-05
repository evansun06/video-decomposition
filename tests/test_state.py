from __future__ import annotations

import csv
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
from youtube_decompose.google_speech import GoogleTranscriptionResult
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


if __name__ == "__main__":
    unittest.main()
