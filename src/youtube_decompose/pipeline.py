from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DecomposeConfig, GoogleSpeechConfig
from .folders import setup_work_folder
from .google_speech import GoogleTranscriptionResult, transcribe_audio_with_google
from .media import convert_video_to_audio, convert_video_to_images


@dataclass(frozen=True)
class DecomposeResult:
    work_dir: Path
    image_dir: Path
    audio_dir: Path
    result_dir: Path
    audio_path: Path
    frame_count: int
    transcription: GoogleTranscriptionResult | None = None

    @property
    def transcript_path(self) -> Path | None:
        if self.transcription is None:
            return None
        return self.transcription.transcript_path


def decompose_video(
    video_path: str | Path,
    work_dir: str | Path,
    google_config: GoogleSpeechConfig | None = None,
    config: DecomposeConfig | None = None,
) -> DecomposeResult:
    """Run frame extraction, audio extraction, and optional transcription."""

    config = config or DecomposeConfig()
    paths = setup_work_folder(work_dir)

    frame_count = convert_video_to_images(
        video_path=video_path,
        work_dir=paths["root"],
        frame_rate=config.frame_rate,
    )
    audio_path = convert_video_to_audio(video_path=video_path, work_dir=paths["root"])

    transcription = None
    if config.transcribe:
        if google_config is None:
            raise ValueError("google_config is required when transcribe=True.")

        transcription = transcribe_audio_with_google(
            audio_path=audio_path,
            result_dir=paths["result"],
            config=google_config,
        )

    return DecomposeResult(
        work_dir=paths["root"],
        image_dir=paths["image"],
        audio_dir=paths["audio"],
        result_dir=paths["result"],
        audio_path=audio_path,
        frame_count=frame_count,
        transcription=transcription,
    )

