@echo off
setlocal enabledelayedexpansion

REM Check if a directory was passed as a parameter
if "%~1"=="" (
    echo Please provide a directory.
    echo Usage: %~nx0 "C:\path\to\your\folder"
    goto :eof
)

REM Set the directory from the first parameter
set "directory=%~1"

REM Change to the specified directory
cd /d "%directory%" || (
    echo The directory does not exist: %directory%
    goto :eof
)

REM Loop through all files in the directory
for %%f in (*.*) do (
    REM Get the filename without the extension
    set "filename=%%~nf"

    REM Replace all dots with p
    set "newname=!filename:p=.!"

    REM Append the file extension
    set "newname=!newname!%%~xf"

    REM Rename the file if the new name is different
    if "%%f" neq "!newname!" ren "%%f" "!newname!"
)

echo Renaming completed.
