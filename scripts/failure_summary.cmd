@echo off
setlocal
pushd "%~dp0.."
python "%~dp0failure_summary.py"
popd
