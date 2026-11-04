param([string]$CompilerPath = '', [string]$OutputDirectory = '')
$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

if (-not $CompilerPath) {
    $compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($compiler) { $CompilerPath = $compiler.Source }
    foreach ($base in @(${env:ProgramFiles(x86)}, $env:ProgramFiles, (Join-Path $env:LOCALAPPDATA 'Programs'))) {
        if ($CompilerPath -or -not $base) { continue }
        foreach ($version in @('7', '6')) {
            $candidate = Join-Path $base "Inno Setup $version\ISCC.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { $CompilerPath = $candidate; break }
        }
    }
}
if (-not $CompilerPath -or -not (Test-Path -LiteralPath $CompilerPath -PathType Leaf)) {
    throw 'Inno Setup compiler not found. Install Inno Setup 6.7+ or pass -CompilerPath with the full ISCC.exe path. See docs/BUILD_INSTALLER.md.'
}
$CompilerPath = (Resolve-Path -LiteralPath $CompilerPath).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot 'dist' }
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$metadata = Get-Content -LiteralPath (Join-Path $projectRoot 'pyproject.toml') -Raw
$match = [regex]::Match($metadata, '(?m)^version\s*=\s*"(\d+\.\d+\.\d+)"\s*$')
if (-not $match.Success) { throw 'Expected a three-part project version in pyproject.toml.' }
$appVersion = $match.Groups[1].Value
$filename = "NC-Voice-Dataset-Factory-$appVersion-Setup"
$buildRoot = Join-Path $projectRoot ('.artifacts\installer-build-' + [guid]::NewGuid().ToString('N'))
$payloadRoot = Join-Path $buildRoot 'payload'
$compiledRoot = Join-Path $buildRoot 'compiled'
New-Item -ItemType Directory -Path $payloadRoot, $compiledRoot, $OutputDirectory -Force | Out-Null

function Copy-PayloadFile {
    param([string]$Source, [string]$Destination)
    $sourcePath = Join-Path $projectRoot $Source
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) { throw "Missing installer input: $Source" }
    $targetPath = Join-Path $payloadRoot $Destination
    New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath
}

# Explicit inputs: never recurse over the application root or copy local data.
foreach ($name in @(
    'README.md', 'pyproject.toml', 'setup.cmd', 'Start_NC_Voice_Dataset_Factory.cmd',
    'constraints-models.txt', 'requirements-core.txt', 'requirements-dev.txt',
    'requirements-models.txt', 'requirements-export.txt', 'requirements-speaker.txt',
    'requirements-runtime-lock.txt'
)) { Copy-PayloadFile $name $name }
foreach ($name in @('setup.ps1', 'setup_windows.ps1', 'start.ps1', 'install_runtime.ps1')) {
    Copy-PayloadFile "scripts\$name" "scripts\$name"
}
foreach ($name in @('INSTALLATION.md', 'USER_GUIDE.md', 'BUILD_INSTALLER.md', 'VALIDATION.md', 'VOICE_AUTHORIZATION.md')) {
    Copy-PayloadFile "docs\$name" "docs\$name"
}
Copy-PayloadFile 'installer\INSTALL.txt' 'INSTALL.txt'
Copy-PayloadFile 'installer\Repair_Setup.cmd' 'Repair_Setup.cmd'
foreach ($name in @('checkbox_round_active.svg', 'checkbox_round_inactive.svg')) {
    Copy-PayloadFile "src\nc_factory\ui\assets\$name" "src\nc_factory\ui\assets\$name"
}
foreach ($sourceFile in Get-ChildItem -LiteralPath (Join-Path $projectRoot 'src\nc_factory') -Recurse -File -Filter '*.py') {
    if ($sourceFile.FullName -match '[\\/](__pycache__|\.venv)[\\/]') { continue }
    $relative = $sourceFile.FullName.Substring($projectRoot.Length + 1)
    Copy-PayloadFile $relative $relative
}

& $CompilerPath '/Qp' "/DPayloadRoot=$payloadRoot" "/DAppVersion=$appVersion" "/O$compiledRoot" "/F$filename" (Join-Path $projectRoot 'installer\NCVoiceDatasetFactory.iss')
if ($LASTEXITCODE -ne 0) { throw "Installer compiler failed with exit code $LASTEXITCODE." }
$compiled = Join-Path $compiledRoot ($filename + '.exe')
if (-not (Test-Path -LiteralPath $compiled -PathType Leaf)) { throw 'Compiler did not produce the installer.' }
$installerPath = Join-Path $OutputDirectory ($filename + '.exe')
Copy-Item -LiteralPath $compiled -Destination $installerPath -Force
$hasher = [System.Security.Cryptography.SHA256]::Create()
$stream = [System.IO.File]::OpenRead($installerPath)
try {
    $hash = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
} finally {
    $stream.Dispose()
    $hasher.Dispose()
}
Set-Content -LiteralPath ($installerPath + '.sha256') -Encoding ASCII -Value "$hash  $filename.exe"
$files = @(Get-ChildItem -LiteralPath $payloadRoot -Recurse -File | ForEach-Object {
    $_.FullName.Substring($payloadRoot.Length + 1)
} | Sort-Object)
@{ version = $appVersion; files = $files } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath ($installerPath + '.payload.json') -Encoding UTF8
Write-Host "Installer: $installerPath"
Write-Host "SHA-256:   $hash"
Write-Host "Payload:   $($files.Count) files; staged at $payloadRoot"
