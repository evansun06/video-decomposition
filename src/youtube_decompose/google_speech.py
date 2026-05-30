from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

from .config import GoogleSpeechConfig


@dataclass(frozen=True)
class GoogleTranscriptionResult:
    transcript_text: str
    transcript_path: Path
    gcs_uri: str


def build_google_credentials(config: GoogleSpeechConfig) -> Any | None:
    """Build explicit Google credentials or return None for ADC fallback."""

    if config.credentials_path and config.credentials_info:
        raise ValueError("Pass either credentials_path or credentials_info, not both.")

    from google.oauth2 import service_account

    if config.credentials_path:
        return service_account.Credentials.from_service_account_file(
            config.credentials_path
        )

    if config.credentials_info:
        return service_account.Credentials.from_service_account_info(
            config.credentials_info
        )

    return None


def generate_unique_filename(base_name: str) -> str:
    """Generate a unique object name for GCS uploads."""

    stem, dot, suffix = base_name.rpartition(".")
    if not dot:
        stem = base_name
        suffix = ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = str(uuid.uuid4())[:8]

    if suffix:
        return f"{stem}_{timestamp}_{random_id}.{suffix}"
    return f"{stem}_{timestamp}_{random_id}"


def upload_audio_to_gcs(
    audio_path: str | Path,
    config: GoogleSpeechConfig,
    object_name: str | None = None,
) -> str:
    """Upload an audio file to GCS and return its gs:// URI."""

    from google.cloud import storage

    audio_path = Path(audio_path)
    credentials = build_google_credentials(config)
    storage_client = storage.Client(
        project=config.project_id,
        credentials=credentials,
    )
    bucket = storage_client.bucket(config.bucket_name)
    object_name = object_name or generate_unique_filename(audio_path.name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(audio_path))

    return f"gs://{config.bucket_name}/{object_name}"


def transcribe_audio_with_google(
    audio_path: str | Path,
    result_dir: str | Path,
    config: GoogleSpeechConfig,
) -> GoogleTranscriptionResult:
    """Transcribe a local WAV file via Google Speech-to-Text v1."""

    from google.cloud import speech_v1p1beta1 as speech
    from pydub import AudioSegment

    audio_path = Path(audio_path)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    credentials = build_google_credentials(config)
    speech_client = speech.SpeechClient(credentials=credentials)
    gcs_uri = upload_audio_to_gcs(audio_path, config)

    audio_segment = AudioSegment.from_file(audio_path)
    metadata = speech.RecognitionMetadata(
        interaction_type=speech.RecognitionMetadata.InteractionType.PRESENTATION,
        original_media_type=speech.RecognitionMetadata.OriginalMediaType.VIDEO,
        audio_topic=config.audio_topic,
    )
    speech_contexts = [
        speech.SpeechContext(phrases=[phrase])
        for phrase in config.speech_context_phrases
    ]

    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=audio_segment.frame_rate,
        language_code=config.language_code,
        use_enhanced=config.use_enhanced,
        model=config.model,
        enable_word_time_offsets=False,
        enable_automatic_punctuation=True,
        enable_word_confidence=False,
        audio_channel_count=audio_segment.channels,
        enable_separate_recognition_per_channel=False,
        metadata=metadata,
        speech_contexts=speech_contexts,
    )

    operation = speech_client.long_running_recognize(
        config=recognition_config,
        audio=speech.RecognitionAudio(uri=gcs_uri),
    )
    response = operation.result(timeout=config.operation_timeout_seconds)

    transcript_text = " ".join(
        result.alternatives[0].transcript
        for result in response.results
        if result.alternatives
    ).strip()

    transcript_path = result_dir / "transcript.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")

    return GoogleTranscriptionResult(
        transcript_text=transcript_text,
        transcript_path=transcript_path,
        gcs_uri=gcs_uri,
    )
