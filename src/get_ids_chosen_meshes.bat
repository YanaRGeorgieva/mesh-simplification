@echo off
setlocal enabledelayedexpansion

REM Check if the correct number of arguments are provided
if "%~2"=="" (
    echo Usage: %~nx0 input_directory output_file.csv
    echo Example: %~nx0 C:\path\to\input\directory output.csv
    exit /b 1
)

REM Read the input directory and output CSV file from the command line
set INPUT_DIR=%~1
set OUTPUT_CSV=%~2

REM Start the CSV file with the header
echo file_id > %OUTPUT_CSV%

REM Loop through all files in the input directory
for %%f in (%INPUT_DIR%\*) do (
    REM Get the file name without the extension
    set FILE_NAME=%%~nf

    REM Write the file name to the CSV file
    echo !FILE_NAME! >> %OUTPUT_CSV%
)

echo File names have been written to %OUTPUT_CSV%
pause
