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
if "%LIMIT%"=="" set LIMIT=5
set GOOGLE_SPEECH_MODEL=chirp_3
python -m youtube_decompose.submit_gcp_stt_batches --db "%VIDEO_DB%" --output-root "%NASOUTPUTPATH%" --batch-size 5 --limit "%LIMIT%" --workers 4
python status_counts.py "%VIDEO_DB%"
popd
