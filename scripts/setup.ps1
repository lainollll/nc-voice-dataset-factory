param([switch]$Development, [switch]$Models, [switch]$Exports, [switch]$Speaker, [string]$Python311 = '')
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$environmentPath = Join-Path $projectRoot '.venv'
$pythonPath = Join-Path $environmentPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $environmentPath)) {
    if ($Python311) { & $Python311 -m venv $environmentPath }
    else { & py -3.11 -m venv $environmentPath }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed.' }
}
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project environment is incomplete; inspect .venv manually.' }
& $pythonPath -c 'import sys; raise SystemExit(sys.version_info[:2] != (3,11) or sys.maxsize <= 2**32)'
if ($LASTEXITCODE -ne 0) { throw 'This project environment is not 64-bit Python 3.11. Use a fresh application folder; existing environments are not replaced automatically.' }
$requirementsName = if ($Development) { 'requirements-dev.txt' } else { 'requirements-core.txt' }
& $pythonPath -m pip install -r (Join-Path $projectRoot $requirementsName)
if ($LASTEXITCODE -ne 0) { throw 'Core dependency installation failed.' }
if ($Models) {
    & $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements-models.txt') -c (Join-Path $projectRoot 'requirements-core.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Model dependency installation failed.' }
    & $pythonPath -m pip install --no-deps 'chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2'
    if ($LASTEXITCODE -ne 0) { throw 'Pinned Chatterbox source installation failed.' }
}
if ($Speaker) {
    & $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements-speaker.txt') -c (Join-Path $projectRoot 'requirements-core.txt') -c (Join-Path $projectRoot 'constraints-models.txt') --extra-index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { throw 'Speaker dependency installation failed.' }
}
if ($Exports) {
    & $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements-export.txt') -c (Join-Path $projectRoot 'requirements-core.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Export dependency installation failed.' }
}
& $pythonPath -m pip install --no-deps -e $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Application installation failed.' }
Write-Host 'Setup complete. No model weights were downloaded. Double-click Start_NC_Voice_Dataset_Factory.cmd.'
