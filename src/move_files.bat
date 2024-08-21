@echo off
setlocal enabledelayedexpansion

REM Set the source folder and CSV file from command-line arguments
set "source_folder=%~1"
set "csv_file=%~2"
set "destination_folder=%source_folder%\Subfolder"

REM Ensure the destination folder exists
if not exist "%destination_folder%" (
    echo Creating destination folder: %destination_folder%
    mkdir "%destination_folder%"
) else (
    echo Destination folder already exists.
)

REM Initialize CSV index
set "csv_index=6"

REM Initialize file list
set "file_index=0"
dir /b "%source_folder%" > filelist.txt

REM Start processing CSV file
goto :read_next_csv_line

:read_next_csv_line
    REM Read the current line from the CSV file
    set "filename_prefix="
    for /f "tokens=*" %%A in ('more +%csv_index% "%csv_file%"') do (
        set "filename_prefix=%%A"
        set /a csv_index+=1
        goto :process_files
    )

    REM If no more lines are found, exit the script
    goto :end

:process_files
    REM Reset file index to start from the first file
    set "file_index=1"

    REM Start processing files in the source folder
    goto :read_next_file

:read_next_file
    REM Read the next file from the file list
    set "current_file="
    for /f "skip=%file_index% tokens=*" %%F in (filelist.txt) do (
        set "current_file=%%F"
        goto :check_file
    )

    REM If no more files are found, move to the next CSV prefix
    goto :read_next_csv_line

:check_file
    REM Check if the filename contains the prefix from the CSV file
    echo !current_file! | findstr /i "!filename_prefix!" >nul
    if !errorlevel! == 0 (
        echo Found matching file: !current_file!
        echo Moving "!current_file!" to "%destination_folder%"
        move "%source_folder%\!current_file!" "%destination_folder%"
        goto :read_next_csv_line
    )

    REM If no match, read the next file
    set /a file_index+=1
    goto :read_next_file

:end
    del filelist.txt
    echo All files have been processed.
    pause
