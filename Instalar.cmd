@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    echo Instale Python 3.12 de 64 bits com o Python Launcher e tente novamente.
    pause
    exit /b 1
)
py -3.12 -m venv .venv
if errorlevel 1 goto erro
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto erro
if not exist data mkdir data
echo Instalacao concluida. Abra Iniciar.cmd.
pause
exit /b 0
:erro
echo Nao foi possivel instalar. Confira a mensagem acima.
pause
exit /b 1
