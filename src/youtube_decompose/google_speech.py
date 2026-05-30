from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import uuid

from .config import GoogleSpeechConfig


@dataclass(frozen=True)
class GoogleTranscriptionResult:
    transcript_text: str
    transcript_path: Path
    text_panel_path: Path
    sentence_panel_path: Path
    gcs_uri: str


def build_google_credentials(config: GoogleSpeechConfig) -> Any | None:
    """Build explicit Google credentials or return None for ADC fallback."""

    if config.credentials_path and config.credentials_info:
        raise ValueError("Pass either credentials_path or credentials_info, not both.")

    if not config.credentials_path and not config.credentials_info:
        return None

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


def build_recognizer_name(config: GoogleSpeechConfig) -> str:
    """Build a Speech-to-Text v2 recognizer resource name."""

    if not config.project_id:
        raise ValueError(
            "project_id is required for Google Speech-to-Text v2. Set "
            "GOOGLE_CLOUD_PROJECT or include project_id in GOOGLE_SERVICE_ACCOUNT_JSON."
        )

    return (
        f"projects/{config.project_id}/locations/{config.location}/"
        f"recognizers/{config.recognizer_id}"
    )


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


def _duration_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())

    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", None)
    if seconds is not None or nanos is not None:
        return float(seconds or 0) + float(nanos or 0) / 1_000_000_000

    return float(value)


def _word_rows_from_google_response(response: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for result in response.results:
        if not result.alternatives:
            continue

        for word_info in result.alternatives[0].words:
            text = word_info.word
            start_offset = getattr(
                word_info, "start_offset", getattr(word_info, "start_time", None)
            )
            end_offset = getattr(
                word_info, "end_offset", getattr(word_info, "end_time", None)
            )
            onset = _duration_seconds(start_offset)
            offset = _duration_seconds(end_offset)
            rows.append(
                {
                    "Text": text,
                    "Onset": onset,
                    "Offset": offset,
                    "Duration": offset - onset,
                    "Sentence End": bool(text and text[-1] in {",", ".", "!", "?"}),
                }
            )

    return rows


def _sentence_rows_from_word_rows(
    word_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sentence_rows: list[dict[str, Any]] = []
    current_words: list[dict[str, Any]] = []

    for row in word_rows:
        current_words.append(row)

        text = row["Text"]
        if row["Sentence End"] and text.endswith((".", "!", "?")):
            sentence_rows.append(
                {
                    "Text": " ".join(word["Text"] for word in current_words),
                    "Onset": min(float(word["Onset"]) for word in current_words),
                    "Offset": max(float(word["Offset"]) for word in current_words),
                    "Duration": sum(float(word["Duration"]) for word in current_words),
                    "Sentence ID": len(sentence_rows) + 1,
                }
            )
            current_words = []

    return sentence_rows


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return f"{value:.2f}"
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {fieldname: _format_csv_value(row[fieldname]) for fieldname in fieldnames}
            )


def write_google_transcript_outputs(
    response: Any,
    result_dir: str | Path,
) -> tuple[str, Path, Path, Path]:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    word_rows = _word_rows_from_google_response(response)
    sentence_rows = _sentence_rows_from_word_rows(word_rows)
    transcript_text = " ".join(row["Text"] for row in word_rows)

    transcript_path = result_dir / "script_google.txt"
    text_panel_path = result_dir / "text_panel_google.csv"
    sentence_panel_path = result_dir / "google_sentence_panel.csv"

    transcript_path.write_text(transcript_text, encoding="utf-8")
    _write_csv(
        text_panel_path,
        ["Text", "Onset", "Offset", "Duration", "Sentence End"],
        word_rows,
    )
    _write_csv(
        sentence_panel_path,
        ["Text", "Onset", "Offset", "Duration", "Sentence ID"],
        sentence_rows,
    )

    return transcript_text, transcript_path, text_panel_path, sentence_panel_path


def transcribe_audio_with_google(
    audio_path: str | Path,
    result_dir: str | Path,
    config: GoogleSpeechConfig,
) -> GoogleTranscriptionResult:
    """Transcribe a local WAV file via Google Speech-to-Text v2 batch recognize."""

    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    audio_path = Path(audio_path)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    credentials = build_google_credentials(config)
    speech_client = SpeechClient(credentials=credentials)
    gcs_uri = upload_audio_to_gcs(audio_path, config)
    recognizer = build_recognizer_name(config)

    # Transcription Biases towards X topics eg (personal finance)
    phrase_hints = [
        phrase for phrase in config.speech_context_phrases if phrase.strip()
    ]
    adaptation = None
    if phrase_hints:
        phrase_set = cloud_speech.PhraseSet(
            phrases=[{"value": phrase} for phrase in phrase_hints]
        )
        adaptation = cloud_speech.SpeechAdaptation(
            phrase_sets=[
                cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=phrase_set
                )
            ]
        )

    recognition_config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        adaptation=adaptation,
        language_codes=[config.language_code],
        model=config.model,
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_word_confidence=True,
            enable_automatic_punctuation=True,
        ),
    )

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=recognizer,
        config=recognition_config,
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )
    operation = speech_client.batch_recognize(request=request)
    response = operation.result(timeout=config.operation_timeout_seconds)
    transcript = response.results[gcs_uri].transcript

    (
        transcript_text,
        transcript_path,
        text_panel_path,
        sentence_panel_path,
    ) = write_google_transcript_outputs(transcript, result_dir)

    return GoogleTranscriptionResult(
        transcript_text=transcript_text,
        transcript_path=transcript_path,
        text_panel_path=text_panel_path,
        sentence_panel_path=sentence_panel_path,
        gcs_uri=gcs_uri,
    )
