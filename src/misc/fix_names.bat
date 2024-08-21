@echo off
setlocal enabledelayedexpansion

REM Check if the folder path is passed as an argument
if "%~1"=="" (
    echo Usage: %~nx0 folder_path
    exit /b 1
)

REM Set the folder path from the first argument
set "folder=%~1"

REM Check if the folder exists
if not exist "%folder%" (
    echo The specified folder does not exist: %folder%
    exit /b 1
)

REM Change directory to the specified folder
cd /d "%folder%"

REM Loop through all files in the folder
for %%f in (*.*) do (
    REM Get the file name without the extension
    set "filename=%%~nf"
    REM Get the file extension
    set "extension=%%~xf"
    
    REM Check if the filename contains an underscore
    if not "!filename!"=="!filename:_=!" (
        REM Get the part of the filename before the first underscore
        for /f "tokens=1 delims=_" %%a in ("!filename!") do set "newname=%%a"
        
        REM Rename the file
        ren "%%f" "!newname!!extension!"
    )
)

echo All files have been renamed in %folder%.
pause
