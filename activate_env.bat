@echo off
call C:\Users\Jonas\anaconda3\Scripts\activate.bat btc-auto-trader
echo Ambiente btc-auto-trader ativado!
python --version
python -c "import sys; print('Python path:', sys.executable)" 