# WSmart+ Route package setup for Windows (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
#
# 1. installs the Microsoft Visual C++ x64 runtime if it is missing (torch needs it; asks for administrator rights);
# 2. installs uv if it is missing;
# 3. unpacks the code archive (code.zip or wsmart_route_source.zip) when logic\ is not there yet (the 1.0.0 release's pyproject.toml sits at the root);
# 4. creates the virtual environment (.venv) with the Python in .python-version, or updates it if it
#    already exists (with an NVIDIA GPU it also installs the CUDA build of torch);
# 5. unpacks data.zip into data\ and creates executables\ for the Windows executable.
#
# The package root is the nearest parent folder that holds pyproject.toml. Written for the exported
# package; do not run it against a development checkout.
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$What, [scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "[setup] $What failed (exit code $LASTEXITCODE)" }
}

$root = $PSScriptRoot
while (-not (Test-Path (Join-Path $root "pyproject.toml"))) {
    $parent = Split-Path $root -Parent
    if (-not $parent -or $parent -eq $root) { throw "[setup] no pyproject.toml above $PSScriptRoot" }
    $root = $parent
}
Set-Location $root
Write-Host "[setup] package root: $root"

# torch's DLLs need the Visual C++ 2015-2022 x64 runtime, which a clean Windows does not have.
function Test-VCRuntime {
    $key = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
    return ($key -and $key.Installed -eq 1) -or (Test-Path "$env:WINDIR\System32\vcruntime140_1.dll")
}
$vcUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
if (-not (Test-VCRuntime)) {
    Write-Host "[setup] Microsoft Visual C++ runtime not found: installing it (Windows asks for administrator rights)"
    $installer = Join-Path $env:TEMP "vc_redist.x64.exe"
    $code = "none: the installer did not run"
    # An elevated shell installs directly; otherwise Windows shows an administrator prompt (UAC) to approve
    $elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    $verb = if ($elevated) { @{} } else { @{ Verb = "RunAs" } }
    try {
        Invoke-WebRequest $vcUrl -OutFile $installer -UseBasicParsing
        $code = (Start-Process -FilePath $installer -ArgumentList "/install", "/passive", "/norestart" @verb -Wait -PassThru).ExitCode
    } catch {
        Write-Host "[setup] $($_.Exception.Message)"
    }
    Remove-Item $installer -ErrorAction SilentlyContinue
    # 3010: installed, restart pending; 1638: a newer version is already installed
    if (($code -notin 0, 1638, 3010) -or -not (Test-VCRuntime)) {
        throw ("[setup] the Visual C++ runtime could not be installed (exit code $code). Approve the administrator " +
            "prompt when setup shows it, or download $vcUrl, run it as administrator with " +
            "'/install /passive /norestart', and run setup again")
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[setup] uv not found: installing it"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Invoke-Checked "uv" { uv --version }

$pyx = "import sys,zipfile,pathlib;r=pathlib.Path(sys.argv[1]);s=sys.argv[3].strip('.');z=zipfile.ZipFile(r/sys.argv[2]);n=[x for x in z.namelist() if not x.startswith('__MACOSX/')];t=r if (not s or all(x.startswith(s+'/') for x in n)) else r/s;t.mkdir(parents=True,exist_ok=True);z.extractall(t,n);print('[setup]',sys.argv[2],'unpacked:',len(n),'entries ->',t)"
function Expand-PackageArchive {
    param([string]$Archive, [string]$Folder)
    Invoke-Checked "unpacking $Archive" { uv run --no-project --quiet python -c $pyx $root $Archive $Folder }
}

if (-not (Test-Path "logic\__init__.py")) {
    foreach ($archive in "code.zip", "wsmart_route_source.zip") {
        if (Test-Path $archive) { Expand-PackageArchive $archive "."; break }
    }
    if (-not (Test-Path "logic\__init__.py")) { throw "[setup] logic\ is missing and no code archive was found" }
}

# Creates .venv on the first run and brings it up to date afterwards.
Invoke-Checked "uv sync" { uv sync --extra solvers --extra neural --extra test --no-dev }

# On Windows the default torch wheel is CPU-only: with an NVIDIA GPU, switch to the CUDA build.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "[setup] NVIDIA GPU found: installing the CUDA build of torch"
    Invoke-Checked "CUDA torch" { uv pip install --python .venv\Scripts\python.exe "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130 }
}

if (Test-Path "data.zip") {
    Expand-PackageArchive "data.zip" "data"
} else {
    Write-Host "[setup] no data.zip: skipping (place the runtime data in $root\data yourself)"
}

New-Item -ItemType Directory -Force -Path "executables" | Out-Null

Write-Host "[setup] done. Activate the environment with:  .venv\Scripts\Activate.ps1"
