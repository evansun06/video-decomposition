# YouTube Decompose Migration

This folder preserves the minimum source needed to reconstruct the old
`main1_decompose.py` workflow outside the original repo.

It can:

- create per-video work folders
- cut video frames into `image_temp/`
- extract audio into `audio_temp/audio_full.wav`
- optionally transcribe that audio with Google Speech-to-Text v2
- write the original transcript outputs to `result_temp/`

It intentionally does **not** include Face++, DeepFace, speech emotion,
aggregation, cleaning, multi-file batching, SQLite status tracking, or a split
upload/transcription pipeline. The current transcription path uses Google v2
`BatchRecognize`, but still submits one audio file per call.

## Layout

```text
migration/
  README.md
  pyproject.toml
  .env.example
  src/youtube_decompose/
    config.py
    folders.py
    google_speech.py
    media.py
    pipeline.py
```

## Install

From this folder:

```bash
python -m pip install -e .
```

You also need `ffmpeg` available on `PATH`. `moviepy`, `pliers`, and `pydub`
depend on it for media extraction.

## Google Cloud Setup

The transcription path uploads `audio_full.wav` to Cloud Storage, then calls
Speech-to-Text v2 `BatchRecognize` with that `gs://` URI. Batch recognition
requires Cloud Storage input.

Set these shell variables for the commands below:

```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export BUCKET_NAME="your-unique-audio-bucket"
export SERVICE_ACCOUNT_NAME="youtube-decompose-stt"
export SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
```

Create or select the Google Cloud project, then enable billing in the console:

```bash
gcloud auth login
gcloud init
gcloud config set project "${PROJECT_ID}"
```

Enable the required APIs:

```bash
gcloud services enable speech.googleapis.com storage.googleapis.com
```

Create the Cloud Storage bucket used for uploaded audio:

```bash
gcloud storage buckets create "gs://${BUCKET_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}"
```

Create the service account used by this package:

```bash
gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
  --display-name="YouTube Decompose Speech-to-Text"
```

Grant the service account permission to call Speech-to-Text:

```bash
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/speech.client"
```

Grant the service account permission to upload/read audio objects in the bucket:

```bash
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/storage.objectUser"
```

Grant the Google-managed Speech service agent permission to read the batch audio
from the bucket:

```bash
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-speech.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

Create a JSON key for local development. Treat this file and its contents as a
secret:

```bash
gcloud iam service-accounts keys create ./gcp-service-account.json \
  --iam-account="${SERVICE_ACCOUNT_EMAIL}"
```

## Google Credentials

The package does not hard-code credentials and does not mutate environment
variables. The easiest local setup is to paste a service-account JSON into
`GOOGLE_SERVICE_ACCOUNT_JSON`:

```bash
export GOOGLE_BUCKET_NAME="${BUCKET_NAME}"
export GOOGLE_SERVICE_ACCOUNT_JSON="$(tr -d '\n' < ./gcp-service-account.json)"
```

If `project_id` is present in that JSON, `GoogleSpeechConfig.from_env()` uses it
automatically. You can also set `GOOGLE_CLOUD_PROJECT` explicitly:

```bash
export GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"
```

Supported credential methods:

1. Set `GOOGLE_SERVICE_ACCOUNT_JSON` to pasted service-account JSON.
2. Set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON path.
3. Pass `credentials_path` into `GoogleSpeechConfig`.
4. Pass an in-memory service-account dict as `credentials_info`.
5. Rely on Application Default Credentials if neither explicit option is set.

Speech-to-Text v2 defaults are `location="global"`, `recognizer_id="_"`, and
`model="chirp_3"`. Override them through `GoogleSpeechConfig` or these env vars:

```bash
export GOOGLE_SPEECH_LOCATION="global"
export GOOGLE_SPEECH_RECOGNIZER="_"
export GOOGLE_SPEECH_MODEL="chirp_3"
```

This package enables word-level timestamps so it can write
`text_panel_google.csv`. Google documents a shorter Chirp 3 batch limit when
word-level timestamps are enabled; use `GOOGLE_SPEECH_MODEL="long"` for longer
timestamped files if Chirp 3 rejects the audio length.

Copy `.env.example` to `.env` if you want a local environment file. Do not
commit real credential files.

If those variables are already present in the process environment:

```python
from youtube_decompose import GoogleSpeechConfig

google_config = GoogleSpeechConfig.from_env()
```

## Python Usage

```python
from youtube_decompose import DecomposeConfig, GoogleSpeechConfig, decompose_video

google_config = GoogleSpeechConfig(
    bucket_name="your-gcs-bucket",
    project_id="your-project-id",
    credentials_path="/path/to/service-account.json",
)

result = decompose_video(
    video_path="/path/to/video.mp4",
    work_dir="/path/to/output/video_id",
    google_config=google_config,
    config=DecomposeConfig(frame_rate=10, transcribe=True),
)

print(result.transcript_path)
print(result.transcription.text_panel_path)
print(result.transcription.sentence_panel_path)
```

Or load credentials from the environment:

```python
from youtube_decompose import DecomposeConfig, GoogleSpeechConfig, decompose_video

result = decompose_video(
    video_path="/path/to/video.mp4",
    work_dir="/path/to/output/video_id",
    google_config=GoogleSpeechConfig.from_env(),
    config=DecomposeConfig(frame_rate=10, transcribe=True),
)
```

For frame/audio extraction only:

```python
from youtube_decompose import DecomposeConfig, decompose_video

result = decompose_video(
    video_path="/path/to/video.mp4",
    work_dir="/path/to/output/video_id",
    config=DecomposeConfig(transcribe=False),
)
```

## Output

```text
work_dir/
  image_temp/
    image_split-*.png
  audio_temp/
    audio_full.wav
    audio_full_left.wav   # only if source audio is stereo
    audio_full_right.wav  # only if source audio is stereo
  result_temp/
    script_google.txt          # full transcript, only when transcribe=True
    text_panel_google.csv      # word-level timing panel, only when transcribe=True
    google_sentence_panel.csv  # sentence-level timing panel, only when transcribe=True
```

## Google References

- Speech-to-Text v2 batch recognition: https://cloud.google.com/speech-to-text/v2/docs/batch-recognize
- Speech-to-Text v2 quotas and limits: https://cloud.google.com/speech-to-text/v2/quotas
- Speech-to-Text IAM roles: https://docs.cloud.google.com/speech-to-text/v2/docs/iam
- Service-account JSON keys: https://docs.cloud.google.com/iam/docs/keys-create-delete
- Cloud Storage IAM roles: https://cloud.google.com/storage/docs/access-control/iam-roles
