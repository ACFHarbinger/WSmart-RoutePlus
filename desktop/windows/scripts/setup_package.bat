@echo off
rem WSmart Route+ package setup for Windows (Command Prompt).
rem
rem   scripts\windows\setup.bat
rem
rem 1. installs the Microsoft Visual C++ x64 runtime if it is missing (torch needs it; asks for administrator rights);
rem 2. installs uv if it is missing;
rem 3. unpacks the code archive (code.zip or wsmart_route_source.zip) when logic\ is not there yet (the 1.0.0 release's pyproject.toml sits at the root);
rem 4. creates the virtual environment (.venv) with the Python in .python-version, or updates it if it
rem    already exists (with an NVIDIA GPU it also installs the CUDA build of torch);
rem 5. unpacks data.zip into data\ and creates executables\ for the Windows executable.
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

rem torch's DLLs need the Visual C++ 2015-2022 x64 runtime, which a clean Windows does not have.
set "VCURL=https://aka.ms/vs/17/release/vc_redist.x64.exe"
call :vcruntime
if errorlevel 1 (
    echo [setup] Microsoft Visual C++ runtime not found: installing it ^(Windows asks for administrator rights^)
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$f=Join-Path $env:TEMP 'vc_redist.x64.exe'; Invoke-WebRequest '%VCURL%' -OutFile $f -UseBasicParsing; $p=Start-Process -FilePath $f -ArgumentList '/install','/passive','/norestart' -Verb RunAs -Wait -PassThru; Remove-Item $f -ErrorAction SilentlyContinue; if ($p.ExitCode -notin 0,1638,3010) { exit 1 }"
    call :vcruntime
    if errorlevel 1 (echo [setup] the Visual C++ runtime could not be installed. Approve the administrator prompt, or run %VCURL% as administrator with /install /passive /norestart, and run setup again & goto :fail)
)

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

if not exist "logic\__init__.py" (
    if exist "code.zip" (
        uv run --no-project --quiet python -c "%PYX%" "%CD%" code.zip .
    ) else if exist "wsmart_route_source.zip" (
        uv run --no-project --quiet python -c "%PYX%" "%CD%" wsmart_route_source.zip .
    )
    if not exist "logic\__init__.py" (echo [setup] logic\ is missing and no code archive was found & goto :fail)
)

rem Creates .venv on the first run and brings it up to date afterwards.
uv sync --extra solvers --extra neural --extra test --no-dev
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

if not exist "executables" mkdir "executables"

echo [setup] done. Activate the environment with:  .venv\Scripts\activate
endlocal
exit /b 0

:vcruntime
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed 2>nul | find "0x1" >nul && exit /b 0
if exist "%WINDIR%\System32\vcruntime140_1.dll" exit /b 0
exit /b 1

:fail
echo [setup] failed.
endlocal
exit /b 1
