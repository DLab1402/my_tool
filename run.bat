@echo off
REM Go to project directory (optional)
cd /d %~dp0

REM Activate the virtual environment
call D:\my_project\myenv\Scripts\activate.bat

REM Run your Python script inside the venv
python main.py