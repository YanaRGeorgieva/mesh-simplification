@echo off
setlocal enabledelayedexpansion

REM Check if the correct number of arguments are provided
if "%~2"=="" (
    echo Usage: %~nx0 input_folder number_of_files
    echo Example: %~nx0 C:\path\to\folder 1000
    exit /b 1
)

REM Set the input folder and the subfolder for selected files
set "INPUT_FOLDER=%~1"
set "NUMBER_FILES=%~2"
set "SUBFOLDER=%INPUT_FOLDER%\random_selection"

REM Create the subfolder if it doesn't exist
if not exist "%SUBFOLDER%" mkdir "%SUBFOLDER%"

REM Initialize counter
set /a COUNT=0

REM Get a list of all files in the folder
for %%f in ("%INPUT_FOLDER%\*") do (
    set /a COUNT+=1
    set "FILE[!COUNT!]=%%f"
)

REM Check if there are fewer than NUMBER_FILES files
if %COUNT% lss %NUMBER_FILES% (
    echo There are less than "!NUMBER_FILES!" files in the folder.
    pause
    exit /b 1
)

REM Initialize counter for moved files
set /a MOVED=0

REM Randomly select and move 1000 files
:movefiles
if %MOVED% geq %NUMBER_FILES% goto end

REM Generate a random number between 1 and COUNT
set /a "RAND=!random! %% COUNT + 1"

REM Move the randomly selected file if it hasn't been moved yet
if exist "!FILE[%RAND%]!" (
    move "!FILE[%RAND%]!" "%SUBFOLDER%" >nul
    echo Moved file: !FILE[%RAND%]!
    set /a MOVED+=1
    set "FILE[%RAND%]="
)

goto movefiles

:end
echo "!NUMBER_FILES!" files have been randomly selected and moved to %SUBFOLDER%.
pause
``
