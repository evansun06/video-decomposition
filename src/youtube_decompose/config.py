from __future__ import annotations

from dataclasses import dataclass
import json
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
    location: str = "global"
    recognizer_id: str = "_"
    model: str = "chirp_3"
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

        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        credentials_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        if credentials_path and credentials_json:
            raise ValueError(
                "Set either GOOGLE_APPLICATION_CREDENTIALS or "
                "GOOGLE_SERVICE_ACCOUNT_JSON, not both."
            )

        credentials_info = None
        if credentials_json:
            try:
                credentials_info = json.loads(credentials_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "GOOGLE_SERVICE_ACCOUNT_JSON must be valid service-account JSON."
                ) from exc

            if not isinstance(credentials_info, dict):
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON must decode to an object.")

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project_id is None and credentials_info is not None:
            project_id = credentials_info.get("project_id")

        return cls(
            bucket_name=bucket_name,
            project_id=project_id,
            credentials_path=credentials_path,
            credentials_info=credentials_info,
            location=os.environ.get("GOOGLE_SPEECH_LOCATION", "global"),
            recognizer_id=os.environ.get("GOOGLE_SPEECH_RECOGNIZER", "_"),
            model=os.environ.get("GOOGLE_SPEECH_MODEL", "chirp_3"),
        )


@dataclass(frozen=True)
class DecomposeConfig:
    """Configuration for frame/audio extraction and optional transcription."""

    frame_rate: int = 10
    transcribe: bool = True
