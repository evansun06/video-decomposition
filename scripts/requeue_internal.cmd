@echo off
setlocal
pushd "%~dp0.."
python "%~dp0requeue_internal.py"
python status_counts.py "%VIDEO_DB%"
popd
