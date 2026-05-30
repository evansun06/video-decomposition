# YouTube Decompose Migration

This folder preserves the minimum source needed to reconstruct the old
`main1_decompose.py` workflow outside the original repo.

It can:

- create per-video work folders
- cut video frames into `image_temp/`
- extract audio into `audio_temp/audio_full.wav`
- optionally transcribe that audio with Google Speech-to-Text v1
- write a simple transcript to `result_temp/transcript.txt`

It intentionally does **not** include Face++, DeepFace, speech emotion,
aggregation, cleaning, batching, SQLite status tracking, or Google v2
BatchRecognize architecture.

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

## Google Credentials

The package does not hard-code credentials and does not mutate environment
variables. Use one of these methods:

1. Set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON path.
2. Pass `credentials_path` into `GoogleSpeechConfig`.
3. Pass an in-memory service-account dict as `credentials_info`.

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
    transcript.txt        # only when transcribe=True
```
