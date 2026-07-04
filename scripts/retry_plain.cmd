@echo off
setlocal
pushd "%~dp0.."
if "%VIDEO_DB%"=="" (
  echo VIDEO_DB is not set.
  exit /b 1
)
if "%NASOUTPUTPATH%"=="" (
  echo NASOUTPUTPATH is not set.
  exit /b 1
)
set LIMIT=%~1
if "%LIMIT%"=="" set LIMIT=1
set GOOGLE_SPEECH_LOCATION=us
set GOOGLE_SPEECH_MODEL=chirp_3
set GOOGLE_SPEECH_ENABLE_WORD_TIME_OFFSETS=false
set GOOGLE_SPEECH_ENABLE_ADAPTATION=false
python -m youtube_decompose.submit_gcp_stt_batches --db "%VIDEO_DB%" --output-root "%NASOUTPUTPATH%" --batch-size 1 --limit "%LIMIT%" --workers 1 --retry-failed
python status_counts.py "%VIDEO_DB%"
popd
