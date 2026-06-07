from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .job_state.state import StageStatus
from .stages import run_audio_stage_for_video
from .submission import (
    VideoCandidate,
    WorkerSummary,
    add_common_arguments,
    configure_logging,
    fetch_stage_candidates,
    resolve_output_root,
    run_logger,
    run_video_workers,
)


LOGGER_NAME = "youtube_decompose.submit_audio_extraction"


def submit_audio_extraction(
    *,
    db_path: str | Path,
    output_root: str | Path,
    limit: int | None = None,
    workers: int = 1,
) -> WorkerSummary[Path]:
    output_root = Path(output_root)
    candidates = fetch_stage_candidates(
        db_path=db_path,
        status_column="audio_status",
        statuses=(StageStatus.QUEUED,),
        limit=limit,
    )

    def process(
        candidate: VideoCandidate,
        logger: logging.LoggerAdapter,
    ) -> Path:
        work_dir = output_root / candidate.video_id
        logger.info("Reading source video path=%s", candidate.nas_path)
        logger.info("Writing audio extraction outputs work_dir=%s", work_dir)
        audio_path = run_audio_stage_for_video(
            candidate.video_id,
            db_path=db_path,
            output_root=output_root,
        )
        logger.info("Wrote audio output path=%s", audio_path)
        return audio_path

    return run_video_workers(
        logger_name=LOGGER_NAME,
        action_name="audio extraction",
        items=candidates,
        workers=workers,
        get_video_id=lambda candidate: candidate.video_id,
        process_item=process,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m youtube_decompose.submit_audio_extraction",
        description="Submit queued SQLite videos for NAS audio extraction.",
    )
    add_common_arguments(parser)
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
        logger_name="submit_audio_extraction",
        output_root=output_root,
        level=args.log_level,
        log_file=args.log_file,
    )
    run_logger(LOGGER_NAME).info("Logging to %s", log_path)

    summary = submit_audio_extraction(
        db_path=args.db,
        output_root=output_root,
        limit=args.limit,
        workers=args.workers,
    )
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
