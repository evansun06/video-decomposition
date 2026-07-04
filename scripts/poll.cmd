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
if "%LIMIT%"=="" set LIMIT=10
python -m youtube_decompose.poll_transcription_batches --db "%VIDEO_DB%" --output-root "%NASOUTPUTPATH%" --limit "%LIMIT%"
python status_counts.py "%VIDEO_DB%"
popd
