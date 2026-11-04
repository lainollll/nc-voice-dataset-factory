$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$logDirectory = Join-Path $projectRoot 'setup-logs'
$exitCode = 0
$transcribing = $false

try {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $logPath = Join-Path $logDirectory ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.log')
    Start-Transcript -LiteralPath $logPath -ErrorAction Stop | Out-Null
    $transcribing = $true
    Write-Host "Installing NC Voice Dataset Factory in: $projectRoot"
    Write-Host "Setup log: $logPath"
    Write-Host 'Keep this window open while packages download and install.'

    Push-Location -LiteralPath $projectRoot
    try {
        $global:LASTEXITCODE = 0
        & (Join-Path $PSScriptRoot 'setup_windows.ps1')
        if ($LASTEXITCODE -ne 0) {
            $exitCode = $LASTEXITCODE
            throw "Dependency setup returned exit code $exitCode."
        }
        $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
            throw 'Setup did not create the application Python environment.'
        }
        & $pythonPath -B -c 'import nc_factory; from PySide6.QtWidgets import QApplication; import numpy, soundfile'
        if ($LASTEXITCODE -ne 0) { throw 'Application imports failed after dependency setup.' }
        Write-Host 'Runtime verification passed. The application is ready to open.'
        Write-Host 'Speech and transcription model weights are selected in the in-app guide.'
    } finally {
        Pop-Location
    }
} catch {
    if ($exitCode -eq 0) { $exitCode = 1 }
    Write-Host ''
    Write-Host 'Setup needs attention. The application is not ready.'
    Write-Host $_.Exception.Message
    Write-Host 'Check the setup log, resolve the error, then run Repair_Setup.cmd again.'
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
}
exit $exitCode
