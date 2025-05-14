@echo off

rem Run the timeline researcher workflow
echo Starting Timeline Researcher workflow...

rem Activate virtual environment if it exists
if exist venv (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

rem Run the main script
python timeline_researcher.py

rem Exit with the script's exit code
exit /b %ERRORLEVEL%
