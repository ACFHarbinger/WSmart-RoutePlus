# WSmart+ Route package setup for Windows (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
#
# 1. installs uv if it is missing;
# 2. unpacks the code archive (code.zip or wsmart_route_source.zip) when logic\ is not there yet;
# 3. creates the virtual environment (.venv), or updates it if it already exists
#    (with an NVIDIA GPU it also installs the CUDA build of torch);
# 4. unpacks data.zip into data\.
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

if (-not (Test-Path "logic\pyproject.toml")) {
    foreach ($archive in "code.zip", "wsmart_route_source.zip") {
        if (Test-Path $archive) { Expand-PackageArchive $archive "."; break }
    }
    if (-not (Test-Path "logic\pyproject.toml")) { throw "[setup] logic\ is missing and no code archive was found" }
}

# Creates .venv on the first run and brings it up to date afterwards.
Invoke-Checked "uv sync" { uv sync --package wsmart-route --extra solvers --no-dev }

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

Write-Host "[setup] done. Activate the environment with:  .venv\Scripts\Activate.ps1"
