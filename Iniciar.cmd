@echo off
setlocal
cd /d "%~dp0"
if not exist data mkdir data
if not exist ".venv\Scripts\pythonw.exe" (
    echo Ambiente nao encontrado. Execute Instalar.cmd primeiro.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import pandas,PySide6,matplotlib,xgboost,catboost,sklearn" 2>"data\inicializacao.log"
if errorlevel 1 (
    echo Nao foi possivel carregar as bibliotecas. Consulte data\inicializacao.log.
    pause
    exit /b 1
)
start "SolarDesk" ".venv\Scripts\pythonw.exe" "%~dp0app.py"
