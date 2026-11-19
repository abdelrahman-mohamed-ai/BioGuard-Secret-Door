$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'requirements installation failed' }
& $python -m ipykernel install --user --name bioguard-task2 --display-name 'Python (BioGuard Task 2)'
if ($LASTEXITCODE -ne 0) { throw 'kernel registration failed' }
Write-Host 'READY. Open with: .\.venv\Scripts\python.exe -m jupyter lab'
