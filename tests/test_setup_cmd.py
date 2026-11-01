from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows setup entry point")
ROOT = Path(__file__).resolve().parents[1]


def fixture_app(tmp_path):
    root = tmp_path / "Friend app with spaces"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "setup.cmd", root / "setup.cmd")
    shutil.copyfile(ROOT / "scripts/setup_windows.ps1", root / "scripts/setup_windows.ps1")
    (root / "scripts/setup.ps1").write_text(
        """param([switch]$Models,[switch]$Exports,[switch]$Speaker,[switch]$Development,[string]$Python311)
@{models=[bool]$Models; exports=[bool]$Exports; speaker=[bool]$Speaker; development=[bool]$Development; python=$Python311; cwd=(Get-Location).Path} | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot '..\\observed.json')
exit 0
""",
        encoding="utf-8",
    )
    tools = root / "fixture-tools"
    tools.mkdir()
    interpreter = tools / "python311.exe"
    interpreter.write_bytes(b"test-only path; nested setup is stubbed")
    return root, tools, interpreter


def tool_files(tools, interpreter, installed):
    (tools / "py.cmd").write_text(
        f"@echo off\necho {json.dumps(str(interpreter))}\nexit /b {0 if installed else 1}\n"
    )
    (tools / "python.cmd").write_text("@echo off\nexit /b 1\n")
    for name in ("git", "ffmpeg"):
        (tools / f"{name}.cmd").write_text(f"@echo off\nexit /b {0 if installed else 1}\n")


def run_setup(root, tools, *args):
    environment = os.environ.copy()
    environment["PATH"] = str(tools) + os.pathsep + environment["PATH"]
    environment["LOCALAPPDATA"] = str(root / "fake-local-app-data")
    environment["ProgramFiles"] = str(root / "fake-program-files")
    # Fake commands shadow any real tools so no package downloads/installations occur.
    return subprocess.run(
        ["cmd.exe", "/d", "/c", "call", str(root / "setup.cmd"), *args],
        cwd=root.parent,
        env=environment,
        input="\n",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )


def test_setup_cmd_installs_all_runtime_components_without_typing_commands(tmp_path):
    root, tools, interpreter = fixture_app(tmp_path)
    tool_files(tools, interpreter, True)
    (tools / "winget.cmd").write_text("@echo off\nexit /b 90\n")
    result = run_setup(root, tools, "-Development")
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads((root / "observed.json").read_text(encoding="utf-8-sig"))
    assert all(observed[k] for k in ("models", "exports", "speaker", "development"))
    assert Path(observed["cwd"]) == root and Path(observed["python"]) == interpreter


def test_setup_bootstraps_missing_prerequisites_using_exact_winget_ids(tmp_path):
    root, tools, interpreter = fixture_app(tmp_path)
    tool_files(tools, interpreter, False)
    (tools / "install-mock.ps1").write_text(
        """param([string]$PackageId)
$folder=$PSScriptRoot
Add-Content -LiteralPath (Join-Path $folder 'packages.txt') -Value $PackageId
switch ($PackageId) {
 'Python.Python.3.11' { Set-Content -LiteralPath (Join-Path $folder 'py.cmd') -Value ('@echo off' + "`r`necho " + ((Join-Path $folder 'python311.exe') | ConvertTo-Json) + "`r`nexit /b 0") }
 'Git.Git' { Set-Content -LiteralPath (Join-Path $folder 'git.cmd') -Value "@echo off`r`nexit /b 0" }
 'Gyan.FFmpeg' { Set-Content -LiteralPath (Join-Path $folder 'ffmpeg.cmd') -Value "@echo off`r`nexit /b 0" }
 default { exit 99 }
}
exit 0
""",
        encoding="utf-8",
    )
    (tools / "winget.cmd").write_text(
        '@echo off\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-mock.ps1" -PackageId "%3"\nexit /b %errorlevel%\n'
    )
    result = run_setup(root, tools)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tools / "packages.txt").read_text().splitlines() == [
        "Python.Python.3.11",
        "Git.Git",
        "Gyan.FFmpeg",
    ]
    assert (root / "observed.json").exists()


def test_setup_dependency_install_failure_is_visible_and_stops_runtime_install(tmp_path):
    root, tools, interpreter = fixture_app(tmp_path)
    tool_files(tools, interpreter, False)
    (tools / "winget.cmd").write_text("@echo off\nexit /b 5\n")
    result = run_setup(root, tools)
    assert result.returncode != 0
    assert "Python.Python.3.11" in result.stdout + result.stderr
    assert not (root / "observed.json").exists()


def test_setup_cmd_preserves_runtime_installer_failure_code(tmp_path):
    root, tools, interpreter = fixture_app(tmp_path)
    tool_files(tools, interpreter, True)
    (tools / "winget.cmd").write_text("@echo off\nexit /b 90\n")
    (root / "scripts/setup.ps1").write_text("exit 7\n")
    result = run_setup(root, tools)
    assert result.returncode == 7 and "error 7" in result.stdout


def test_real_python_probe_returns_64bit_311_path_without_running_installer(tmp_path):
    script = tmp_path / "probe.ps1"
    script.write_text(
        """param([string]$Bootstrap,[string]$Python)
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile($Bootstrap,[ref]$tokens,[ref]$errors)
$function=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Read-Python311Path'},$true)
. ([ScriptBlock]::Create($function.Extent.Text))
$found=Read-Python311Path -Command $Python
if (-not $found) { Write-Error 'Real Python 3.11 probe failed'; exit 7 }
$found | ConvertTo-Json
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Bootstrap",
            str(ROOT / "scripts/setup_windows.ps1"),
            "-Python",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(json.loads(result.stdout)) == Path(sys.executable)
