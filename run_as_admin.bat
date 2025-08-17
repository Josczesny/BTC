@echo off
REM ===================================================================
REM SCRIPT DE ELEVACAO AUTOMATICA DE PRIVILEGIOS PARA TRADING BTC
REM ===================================================================

echo 🔍 VERIFICANDO PRIVILEGIOS DE ADMINISTRADOR...

REM Verifica se já está executando como administrador
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ JA EXECUTANDO COMO ADMINISTRADOR
    goto :run_python
) else (
    echo ⚠️ NAO ESTA EXECUTANDO COMO ADMINISTRADOR
    echo 🚀 SOLICITANDO ELEVACAO DE PRIVILEGIOS...
)

REM Solicita elevação de privilégios
echo 💡 Uma janela UAC será exibida - clique em "Sim" para continuar
powershell -Command "Start-Process cmd -ArgumentList '/c cd /d %~dp0 && %~nx0 elevated %*' -Verb RunAs"
goto :eof

:run_python
REM Chegou aqui = já tem privilégios de admin
echo ✅ PRIVILEGIOS DE ADMINISTRADOR CONFIRMADOS
echo 🚀 INICIANDO SISTEMA DE TRADING BTC...

REM Ativa ambiente conda se existir
if exist "activate_env.bat" (
    echo 🔄 ATIVANDO AMBIENTE CONDA...
    call activate_env.bat
)

REM 🔧 EXECUTA O SISTEMA PRINCIPAL (main.py)
echo 🐍 EXECUTANDO SISTEMA MODULARIZADO...
python main.py %*

echo.
echo ✅ SISTEMA FINALIZADO
pause
goto :eof 