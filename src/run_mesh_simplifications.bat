@echo off
setlocal enabledelayedexpansion

REM Set the Python script path
set PYTHON_SCRIPT=simplify_meshes_cli.py

REM Check if the correct number of arguments are provided
if "%~2"=="" (
    echo Usage: %~nx0 input_folder output_folder_base
    echo Example: %~nx0 C:\path\to\input\folder C:\path\to\output\folders
    exit /b 1
)

REM Read the input and output folders from the command line
set INPUT_FOLDER=%~1
set OUTPUT_FOLDERS=%~2

REM Define the lists of threshold, ratio, and penalty_weight values
set THRESHOLDS=0.05 0.1 0.2 0.3
set RATIOS=0.3 0.5 0.7 0.9
set PENALTIES=2000

REM Loop through each combination of threshold, ratio, and penalty_weight
for %%t in (%THRESHOLDS%) do (
    for %%r in (%RATIOS%) do (
        for %%p in (%PENALTIES%) do (
            REM Create an output folder named based on the current combination
            set OUTPUT_FOLDER=%OUTPUT_FOLDERS%\output_t%%t_r%%r_p%%p
            mkdir %OUTPUT_FOLDER%
            
            REM Run the Python script with the current combination
            python %PYTHON_SCRIPT% %INPUT_FOLDER% %OUTPUT_FOLDER% --threshold %%t --simplification_ratio %%r --penalty_weight %%p
            
            REM Print status
            echo Finished processing with threshold=%%t, ratio=%%r, penalty_weight=%%p
        )
    )
)

echo All combinations processed.
pause
