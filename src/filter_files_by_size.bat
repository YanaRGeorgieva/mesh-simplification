@echo off
setlocal enabledelayedexpansion

REM Check if the correct number of arguments are provided
if "%~2"=="" (
    echo Usage: %~nx0 source_folder size_limit_MB
    echo Example: %~nx0 "C:\path\to\folder" 6
    exit /b 1
)

REM Read the source folder from the command line
set "SOURCE_FOLDER=%~1"

REM Read the size limit from the command line and convert it to bytes
set "SIZE_LIMIT_MB=%~2"
set /a SIZE_LIMIT=%SIZE_LIMIT_MB%*1024*1024

REM Define the new subfolder name
set "DEST_FOLDER=%SOURCE_FOLDER%\FilteredFiles"

REM Create the destination subfolder if it doesn't exist
if not exist "%DEST_FOLDER%" mkdir "%DEST_FOLDER%"

REM Loop through each file in the source folder
for %%f in ("%SOURCE_FOLDER%\*.*") do (
    REM Get the file size in bytes
    set "FILESIZE=%%~zf"
    
    REM Enable delayed expansion within the loop
    setlocal enabledelayedexpansion
    
    REM Perform the file size comparison and move the file if it meets the condition
    if !FILESIZE! leq %SIZE_LIMIT% (
        move "%%f" "%DEST_FOLDER%"
    )
    
    endlocal
)

echo All files under %SIZE_LIMIT_MB%MB have been moved to "%DEST_FOLDER%".
pause
