@echo off
setlocal
pushd "%~dp0.." || exit /b 1
if exist ".venv" goto validate

py -3.13 -c "import sys; sys.exit(sys.version_info[:2] != (3, 13))" >nul 2>&1
if not errorlevel 1 (
    py -3.13 -m venv .venv
    if errorlevel 1 goto failed
    goto validate
)
python3.13 -c "import sys; sys.exit(sys.version_info[:2] != (3, 13))" >nul 2>&1
if not errorlevel 1 (
    python3.13 -m venv .venv
    if errorlevel 1 goto failed
    goto validate
)
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 13))" >nul 2>&1
if not errorlevel 1 (
    python -m venv .venv
    if errorlevel 1 goto failed
    goto validate
)
echo Python 3.13 indisponivel. Instale-o e confirme o comando py -3.13 ou python no PATH. 1>&2
goto failed

:validate
if not exist ".venv\Scripts\python.exe" goto invalid_venv
".venv\Scripts\python.exe" -c "import sys; sys.exit(not (sys.version_info[:2] == (3, 13) and sys.prefix != sys.base_prefix))"
if errorlevel 1 goto invalid_venv
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
echo Instalacao concluida. Execute scripts\run_local.bat.
popd
exit /b 0

:invalid_venv
echo A pasta .venv contem um ambiente ausente, invalido ou incompativel com Python 3.13. 1>&2
echo Preserve ou renomeie essa pasta antes de tentar criar outro ambiente. 1>&2
:failed
popd
exit /b 1
