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
aggregation, or cleaning. The current one-video transcription helper uses
Google v2 `BatchRecognize` with one audio file per call. The SQLite-backed
submission scripts can run local extraction and submit multi-file
Speech-to-Text batch operations, while the poller writes per-video transcript
outputs when those operations complete.

## Project Goal

The end state is to run this analysis pipeline on a Windows remote desktop that
is physically connected to the NAS containing the source MP4 files. Development
can happen on macOS, but production paths and write behavior must work from the
Windows/NAS environment.

The NAS is the source of truth for input video files. Two CSV files contain the
exact NAS paths needed to retrieve the MP4s. Before the first full run, those CSV
inputs should be diffed against the IDs that have already been analyzed. The
remaining unanalyzed IDs can then seed the SQLite tracking database.

The database should avoid storing Mac-only absolute paths as the only source of
truth. Prefer storing video IDs plus paths relative to configured roots, or store
the NAS path separately from machine-specific local/output roots.

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
    poll_transcription_batches.py
    media.py
    pipeline.py
    job_state/
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

Speech-to-Text v2 defaults are `location="us"`, `recognizer_id="_"`, and
`model="chirp_3"`. Override them through `GoogleSpeechConfig` or these env vars:

```bash
export GOOGLE_SPEECH_LOCATION="us"
export GOOGLE_SPEECH_RECOGNIZER="_"
export GOOGLE_SPEECH_MODEL="chirp_3"
```

This package enables word-level timestamps so it can write
`text_panel_google.csv`. Google documents a shorter Chirp 3 batch limit when
word-level timestamps are enabled.

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

For separated one-video stage execution with SQLite status updates:

```python
from pathlib import Path

from youtube_decompose import (
    GoogleSpeechConfig,
    run_audio_stage_for_video,
    run_image_stage_for_video,
    run_transcription_stage_for_video,
)

db_path = Path("job_state/video_state.sqlite")
output_root = Path(r"D:\youtube_analysis_output")

run_audio_stage_for_video("--B5nAoQVTI", db_path=db_path, output_root=output_root)
run_transcription_stage_for_video(
    "--B5nAoQVTI",
    google_config=GoogleSpeechConfig.from_env(),
    db_path=db_path,
    output_root=output_root,
)
run_image_stage_for_video("--B5nAoQVTI", db_path=db_path, output_root=output_root)
```

`YOUTUBE_DECOMPOSE_OUTPUT_ROOT` can also set the default output root. Each stage
uses the seeded `videos.nas_path` as the input path and writes outputs under
`output_root / video_id`, preserving `audio_temp/`, `result_temp/`, and
`image_temp/`.

## Submission Scripts

Set the NAS output root before running submission scripts:

```bash
export NASOUTPUTPATH="/path/to/nas/output"
```

Every submission script writes under:

```text
NASOUTPUTPATH/
  video_id/
    audio_temp/
    image_temp/
    result_temp/
```

Run one-video smoke tests with `--limit 1`; omit `--limit` to process every
eligible SQLite row. `--workers` controls local parallelism, and every log line
includes `video_id=...` so reads, writes, and failures can be traced to one
video.

Audio extraction:

```bash
python -m youtube_decompose.submit_audio_extraction \
  --db job_state/video_state.sqlite \
  --limit 1 \
  --workers 2
```

Image decomposition:

```bash
python -m youtube_decompose.submit_image_decomposition \
  --db job_state/video_state.sqlite \
  --frame-rate 10 \
  --limit 1 \
  --workers 2
```

Submit extracted audio to Google Speech-to-Text batches:

```bash
python -m youtube_decompose.submit_gcp_stt_batches \
  --db job_state/video_state.sqlite \
  --batch-size 5 \
  --limit 5 \
  --workers 4
```

The STT submitter selects rows with `audio_status='done'` and
`transcription_status='queued'`, uploads `audio_full.wav` to Cloud Storage,
submits `BatchRecognize` operations, records `transcription_batches`, and marks
included videos `running`. Use `--retry-failed` to include failed transcription
rows in a later retry.

## Batch Transcription Polling

The poller checks submitted Speech-to-Text operations and writes transcript
outputs once Google completes them.

The GCS URI lifecycle is:

```text
local audio_temp/audio_full.wav
  -> upload to gs://bucket/object.wav
  -> submit that URI in a Speech-to-Text BatchRecognize request with GCS output
  -> Google returns result-object URIs keyed by input audio URI
  -> poller finds videos.gcs_uri and writes result_temp transcript files
```

The minimal batch table is:

```sql
transcription_batches (
    batch_id TEXT PRIMARY KEY,
    operation_name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,          -- submitted, done, failed
    gcs_uris_json TEXT NOT NULL,   -- JSON array of gs:// URIs
    submitted_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

Poll submitted operations once:

```bash
python -m youtube_decompose.poll_transcription_batches \
  --db job_state/video_state.sqlite \
  --output-root "$NASOUTPUTPATH" \
  --limit 50
```

On Windows PowerShell, poll 10 batches every 2 minutes until no submitted
batches remain:

```powershell
.\scripts\poll_loop.ps1
```

Override the defaults when needed:

```powershell
.\scripts\poll_loop.ps1 -Limit 10 -SleepSeconds 120 -QuotaSleepSeconds 600
```

If a Google operation is still running, the batch remains `submitted`. If it is
complete, the poller writes `script_google.txt`, `text_panel_google.csv`, and
`google_sentence_panel.csv` under each video's `result_temp/`, then marks those
videos and the batch `done`. File-level failures mark only that video `failed`;
batch-level operation failures mark the batch and its unfinished videos
`failed`. If local transcript materialization fails, the batch remains
`submitted` with the error recorded so the same completed Google operation can
be polled again after the local issue is fixed.

For smoke tests, use a small submitted batch and poll one operation:

```bash
python -m youtube_decompose.poll_transcription_batches --limit 1
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

## Roadmap

1. Configure runtime paths for macOS development and Windows/NAS execution.
   - Define input CSV locations, NAS video roots, output roots, and temporary
     work roots.
   - Confirm the Windows remote desktop can read NAS MP4 paths and write output
     folders without path, permission, or file-locking issues.

2. Diff the two NAS-path CSV inputs against already-analyzed IDs.
   - Treat this as a one-time migration step.
   - Produce the seed set of video IDs and NAS MP4 paths that still need
     analysis.

3. Add SQLite state tracking and seed it from the diff result. [partially done]
   - Track video ID, NAS source path, work/output paths, local decomposition
     state, GCS URI, batch ID, transcript state, errors, and timestamps.
   - Use SQLite as the coordination layer between local decomposition and cloud
     transcription.

4. Split the code into two pipelines.
   - Local pipeline: video from NAS -> audio files + sampled frames -> SQLite
     state update.
   - Transcript/GCP pipeline: pending audio -> GCS upload -> v2 batch
     recognition -> transcript output files -> SQLite state update.

5. Use Speech-to-Text v2 batch recognition properly. [submitter and poller done]
   - Group pending audio files into batches of up to 5 GCS URIs.
   - Store submitted operation names in SQLite.
   - Poll active cloud batch operations separately from local decomposition.
   - Write `script_google.txt`, `text_panel_google.csv`, and
     `google_sentence_panel.csv` when operations complete.

6. Configure GCP and billing for the production run.
   - Enable the required APIs, create/select the bucket, configure service
     account access, and set credentials on the Windows remote desktop.
   - Keep credential JSON out of git and environment-specific config.

7. Add smoke tests.
   - Include or point to one small MP4 fixture in a local smoke-test directory.
   - Test helper functions for transcript output generation, env parsing, and
     SQLite state transitions.
   - Run one end-to-end smoke test on the Windows/NAS environment before the
     full analysis run.
