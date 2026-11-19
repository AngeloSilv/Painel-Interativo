@echo off
setlocal
pushd "%~dp0.." || exit /b 1
if not exist ".venv\Scripts\python.exe" goto setup_required
".venv\Scripts\python.exe" -c "import sys; sys.exit(not (sys.version_info[:2] == (3, 13) and sys.prefix != sys.base_prefix))"
if errorlevel 1 goto setup_required
".venv\Scripts\python.exe" -m streamlit --version
if errorlevel 1 goto setup_required
if not defined STREAMLIT_SERVER_PORT set "STREAMLIT_SERVER_PORT=8501"
echo Endereco local esperado: http://127.0.0.1:%STREAMLIT_SERVER_PORT%
echo Para encerrar, pressione Ctrl+C.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.address 127.0.0.1
set "APP_EXIT=%ERRORLEVEL%"
popd
exit /b %APP_EXIT%

:setup_required
echo Ambiente indisponivel. Execute scripts\setup_local.bat e confira as mensagens. 1>&2
popd
exit /b 1
