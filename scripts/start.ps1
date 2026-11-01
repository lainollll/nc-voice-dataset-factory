param([string]$Project = '')
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run scripts\setup.ps1 first.' }
if ($Project) { & $pythonPath -X faulthandler -m nc_factory --project $Project }
else { & $pythonPath -X faulthandler -m nc_factory }
exit $LASTEXITCODE
