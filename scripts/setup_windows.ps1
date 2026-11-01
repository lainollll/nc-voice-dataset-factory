param([switch]$Development)
$ErrorActionPreference = 'Stop'

function Refresh-SetupPath {
    # Refresh only this process; preserve custom paths already supplied by the user.
    $paths = @($env:Path, [Environment]::GetEnvironmentVariable('Path', 'User'), [Environment]::GetEnvironmentVariable('Path', 'Machine'))
    foreach ($base in @($env:LOCALAPPDATA, $env:ProgramFiles)) {
        if (-not $base) { continue }
        $links = if ($base -eq $env:LOCALAPPDATA) { Join-Path $base 'Microsoft\WinGet\Links' } else { Join-Path $base 'WinGet\Links' }
        if (Test-Path -LiteralPath $links -PathType Container) { $paths += $links }
    }
    $seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $entries = foreach ($value in $paths) {
        if (-not $value) { continue }
        foreach ($entry in $value.Split(';')) {
            $expanded = [Environment]::ExpandEnvironmentVariables($entry).Trim()
            if ($expanded -and $seen.Add($expanded.TrimEnd('\'))) { $expanded }
        }
    }
    $env:Path = $entries -join ';'
}

function Read-Python311Path {
    param([string]$Command, [string[]]$Prefix = @())
    try {
        $response = & $Command @Prefix -c 'import sys,json; assert sys.version_info[:2] == (3,11) and sys.maxsize > 2**32; print(json.dumps(sys.executable))' 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $response) { return $null }
        $executable = ($response | Select-Object -Last 1) | ConvertFrom-Json
        if ($executable -is [string] -and (Test-Path -LiteralPath $executable -PathType Leaf)) { return $executable }
    } catch { return $null }
    return $null
}

function Find-Python311 {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $found = Read-Python311Path -Command $launcher.Source -Prefix @('-3.11')
        if ($found) { return $found }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    # Avoid starting the Microsoft Store through its Python execution alias.
    if ($python -and $python.Source -notlike '*\WindowsApps\*') {
        $found = Read-Python311Path -Command $python.Source
        if ($found) { return $found }
    }
    $candidates = @()
    if ($env:LOCALAPPDATA) { $candidates += Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe' }
    if ($env:ProgramFiles) { $candidates += Join-Path $env:ProgramFiles 'Python311\python.exe' }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $found = Read-Python311Path -Command $candidate
            if ($found) { return $found }
        }
    }
    return $null
}

function Test-SetupTool {
    param([string]$Name, [string[]]$Arguments)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) { return $false }
    try {
        & $command.Source @Arguments *> $null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Install-SetupPrerequisite {
    param([string]$PackageId)
    $manager = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $manager) {
        throw "WinGet is missing. Install or update 'App Installer' from Microsoft Store, then double-click setup.cmd again. Needed package: $PackageId"
    }
    Write-Host "Installing missing prerequisite: $PackageId"
    & $manager.Source install --id $PackageId --exact --architecture x64 --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "Installation failed for $PackageId (WinGet exit $LASTEXITCODE). Resolve the message above and run setup.cmd again." }
    Refresh-SetupPath
}

$python311 = Find-Python311
if (-not $python311) {
    Install-SetupPrerequisite -PackageId 'Python.Python.3.11'
    $python311 = Find-Python311
    if (-not $python311) { throw 'Python 3.11 is not available after installation. Close this window and run setup.cmd again.' }
}
foreach ($tool in @(
    @{ Name = 'git'; Arguments = @('--version'); PackageId = 'Git.Git' },
    @{ Name = 'ffmpeg'; Arguments = @('-version'); PackageId = 'Gyan.FFmpeg' }
)) {
    if (-not (Test-SetupTool -Name $tool.Name -Arguments $tool.Arguments)) {
        Install-SetupPrerequisite -PackageId $tool.PackageId
        if (-not (Test-SetupTool -Name $tool.Name -Arguments $tool.Arguments)) {
            throw "$($tool.Name) is not on PATH after installation. Close this window and run setup.cmd again."
        }
    }
}
Write-Host 'Prerequisites ready. Installing the pinned GUI, CUDA/TTS, ASR, export and speaker runtime packages...'
& (Join-Path $PSScriptRoot 'setup.ps1') -Python311 $python311 -Models -Exports -Speaker -Development:$Development
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Use the in-app Getting Started guide to select/download model weights and a permitted voice.'
