@echo off
setlocal
pushd "%~dp0.."
python "%~dp0diag.py"
popd
