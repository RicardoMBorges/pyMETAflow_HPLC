@echo off
setlocal

rem Get the directory path where the batch file is located
set "script_dir=%~dp0"

rem Check if the virtual environment already exists in the script's directory
if exist "%script_dir%pyMetaFlow_HPLC_env\Scripts\activate.bat" (
    echo pyMetaFlow_HPLC_env already exists. Activating...
    call "%script_dir%pyMetaFlow_HPLC_env\Scripts\activate.bat"
) else (
    echo Creating virtual environment in "%script_dir%pyMetaFlow_HPLC_env"...
    py -3.11 -m venv "%script_dir%pyMetaFlow_HPLC_env"
    call "%script_dir%pyMetaFlow_HPLC_env\Scripts\activate.bat"

    echo Installing the requirements...
    call "%script_dir%pyMetaFlow_HPLC_env\Scripts\python.exe" -m pip install -r "%script_dir%requirements.txt"
    
    echo Installing Jupyter Notebook with compatible packages...
    call "%script_dir%pyMetaFlow_HPLC_env\Scripts\python.exe" -m pip install jupyter notebook==6.5.2 traitlets==5.9.0 ipython==7.31.1
)

rem Start Jupyter Notebook
call "%script_dir%pyMetaFlow_HPLC_env\Scripts\jupyter-notebook.exe"

pause
