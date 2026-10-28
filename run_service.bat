@echo off
cd /d "%~dp0\tryon-service"
echo Starting Stylinger Classical CGIP Try-On Service on port 5001...
call .venv\Scripts\activate.bat
python app.py
pause
