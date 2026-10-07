@echo off
rem WSmart+ Route package setup for Windows (Command Prompt).
rem
rem   scripts\windows\setup.bat
rem
rem 1. installs uv if it is missing;
rem 2. unpacks the code archive (code.zip or wsmart_route_source.zip) when logic\ is not there yet;
rem 3. creates the virtual environment (.venv), or updates it if it already exists
rem    (with an NVIDIA GPU it also installs the CUDA build of torch);
rem 4. unpacks data.zip into data\.
rem
rem The package root is the nearest parent folder that holds pyproject.toml. Written for the exported
rem package; do not run it against a development checkout.
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
:findroot
if exist "%ROOT%pyproject.toml" goto :rootfound
for %%I in ("%ROOT%..") do set "PARENT=%%~fI\"
if "%PARENT%"=="%ROOT%" (echo [setup] no pyproject.toml above %~dp0 & goto :fail)
set "ROOT=%PARENT%"
goto :findroot
:rootfound
cd /d "%ROOT%"
echo [setup] package root: %CD%

where uv >nul 2>nul
if errorlevel 1 (
    echo [setup] uv not found: installing it
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto :fail
    set "PATH=%USERPROFILE%\.local\bin;!PATH!"
)
uv --version
if errorlevel 1 goto :fail

set "PYX=import sys,zipfile,pathlib;r=pathlib.Path(sys.argv[1]);s=sys.argv[3].strip('.');z=zipfile.ZipFile(r/sys.argv[2]);n=[x for x in z.namelist() if not x.startswith('__MACOSX/')];t=r if (not s or all(x.startswith(s+'/') for x in n)) else r/s;t.mkdir(parents=True,exist_ok=True);z.extractall(t,n);print('[setup]',sys.argv[2],'unpacked:',len(n),'entries ->',t)"

if not exist "logic\pyproject.toml" (
    if exist "code.zip" (
        uv run --no-project --quiet python -c "%PYX%" "%CD%" code.zip .
    ) else if exist "wsmart_route_source.zip" (
        uv run --no-project --quiet python -c "%PYX%" "%CD%" wsmart_route_source.zip .
    )
    if not exist "logic\pyproject.toml" (echo [setup] logic\ is missing and no code archive was found & goto :fail)
)

rem Creates .venv on the first run and brings it up to date afterwards.
uv sync --package wsmart-route --extra solvers --no-dev
if errorlevel 1 goto :fail

rem On Windows the default torch wheel is CPU-only: with an NVIDIA GPU, switch to the CUDA build.
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo [setup] NVIDIA GPU found: installing the CUDA build of torch
    uv pip install --python .venv\Scripts\python.exe "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130
    if errorlevel 1 goto :fail
)

if exist "data.zip" (
    uv run --no-project --quiet python -c "%PYX%" "%CD%" data.zip data
    if errorlevel 1 goto :fail
) else (
    echo [setup] no data.zip: skipping ^(place the runtime data in %CD%\data yourself^)
)

echo [setup] done. Activate the environment with:  .venv\Scripts\activate
endlocal
exit /b 0

:fail
echo [setup] failed.
endlocal
exit /b 1
