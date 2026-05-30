from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class GoogleSpeechConfig:
    """Configuration for Google Cloud Storage and Speech-to-Text clients."""

    bucket_name: str
    project_id: str | None = None
    credentials_path: str | None = None
    credentials_info: dict[str, Any] | None = None
    language_code: str = "en-US"
    model: str = "video"
    use_enhanced: bool = True
    audio_topic: str = (
        "interviews debt financial planning housing investing macroeconomics savings"
    )
    speech_context_phrases: tuple[str, ...] = (
        "YouTube video",
        "personal finance",
        "financial advice",
        "economics and finance",
    )
    operation_timeout_seconds: int = 1200

    @classmethod
    def from_env(cls) -> "GoogleSpeechConfig":
        """Create config from environment variables."""

        bucket_name = os.environ.get("GOOGLE_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GOOGLE_BUCKET_NAME is required.")

        return cls(
            bucket_name=bucket_name,
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            credentials_path=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        )


@dataclass(frozen=True)
class DecomposeConfig:
    """Configuration for frame/audio extraction and optional transcription."""

    frame_rate: int = 10
    transcribe: bool = True
