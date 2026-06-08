from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .job_state.state import StageStatus
from .stages import run_image_stage_for_video
from .submission import (
    VideoCandidate,
    WorkerSummary,
    add_common_arguments,
    configure_logging,
    fetch_stage_candidates,
    positive_int,
    resolve_output_root,
    run_logger,
    run_video_workers,
)


LOGGER_NAME = "youtube_decompose.submit_image_decomposition"


def submit_image_decomposition(
    *,
    db_path: str | Path,
    output_root: str | Path,
    limit: int | None = None,
    workers: int = 1,
    frame_rate: int = 10,
    retry_failed: bool = False,
) -> WorkerSummary[int]:
    output_root = Path(output_root)
    statuses = [StageStatus.QUEUED]
    if retry_failed:
        statuses.append(StageStatus.FAILED)

    candidates = fetch_stage_candidates(
        db_path=db_path,
        status_column="image_status",
        statuses=statuses,
        limit=limit,
    )

    def process(
        candidate: VideoCandidate,
        logger: logging.LoggerAdapter,
    ) -> int:
        work_dir = output_root / candidate.video_id
        logger.info("Reading source video path=%s", candidate.nas_path)
        logger.info(
            "Writing image decomposition outputs work_dir=%s frame_rate=%s",
            work_dir,
            frame_rate,
        )
        frame_count = run_image_stage_for_video(
            candidate.video_id,
            db_path=db_path,
            output_root=output_root,
            frame_rate=frame_rate,
        )
        logger.info(
            "Wrote image output dir=%s frame_count=%s",
            work_dir / "image_temp",
            frame_count,
        )
        return frame_count

    return run_video_workers(
        logger_name=LOGGER_NAME,
        action_name="image decomposition",
        items=candidates,
        workers=workers,
        get_video_id=lambda candidate: candidate.video_id,
        process_item=process,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m youtube_decompose.submit_image_decomposition",
        description="Submit queued SQLite videos for NAS image decomposition.",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--frame-rate",
        type=positive_int,
        default=10,
        help="Sampled frames per second. Defaults to 10.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Include image_status='failed' rows as eligible work.",
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
        logger_name="submit_image_decomposition",
        output_root=output_root,
        level=args.log_level,
        log_file=args.log_file,
    )
    run_logger(LOGGER_NAME).info("Logging to %s", log_path)

    summary = submit_image_decomposition(
        db_path=args.db,
        output_root=output_root,
        limit=args.limit,
        workers=args.workers,
        frame_rate=args.frame_rate,
        retry_failed=args.retry_failed,
    )
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
