@echo off
setlocal
pushd "%~dp0.."
if "%VIDEO_DB%"=="" (
  echo VIDEO_DB is not set.
  exit /b 1
)
python status_counts.py "%VIDEO_DB%"
popd
