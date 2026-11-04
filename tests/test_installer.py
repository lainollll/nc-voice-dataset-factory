from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sysconfig
import venv
import uuid

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows installer")
ROOT = Path(__file__).resolve().parents[1]


def run_powershell(script: Path, *arguments: str, **kwargs) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), sysconfig.get_path("purelib"))
    )
    return subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(script), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=45,
        **kwargs,
    )


def runtime_fixture(tmp_path: Path, setup: str) -> Path:
    root = tmp_path / "App folder with spaces"
    (root / "scripts").mkdir(parents=True)
    wrapper = ROOT / "scripts/install_runtime.ps1"
    if wrapper.exists():
        shutil.copyfile(wrapper, root / "scripts/install_runtime.ps1")
    (root / "scripts/setup_windows.ps1").write_text(setup, encoding="utf-8")
    return root


def runtime_log(root: Path) -> str:
    logs = list((root / "setup-logs").glob("*.log"))
    assert len(logs) == 1
    return logs[0].read_text(encoding="utf-8-sig")


def test_installer_runtime_logs_success_with_spaces_in_application_path(tmp_path):
    root = runtime_fixture(tmp_path, "Write-Host 'Existing setup was called'; exit 0\n")
    venv.EnvBuilder(with_pip=False).create(root / ".venv")
    result = run_powershell(root / "scripts/install_runtime.ps1", cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    log = runtime_log(root)
    assert "Existing setup was called" in log
    assert "Runtime verification passed" in log


def test_installer_runtime_preserves_dependency_failure_and_log(tmp_path):
    root = runtime_fixture(tmp_path, "Write-Host 'Download failed'; exit 7\n")
    result = run_powershell(root / "scripts/install_runtime.ps1")
    assert result.returncode == 7, result.stdout + result.stderr
    log = runtime_log(root)
    assert "Download failed" in log and "not ready" in log


def test_installer_runtime_catches_setup_exception(tmp_path):
    root = runtime_fixture(tmp_path, "throw 'Prerequisite unavailable'\n")
    result = run_powershell(root / "scripts/install_runtime.ps1")
    assert result.returncode != 0
    assert "Prerequisite unavailable" in runtime_log(root)


def test_installer_runtime_rejects_success_without_environment(tmp_path):
    root = runtime_fixture(tmp_path, "exit 0\n")
    result = run_powershell(root / "scripts/install_runtime.ps1")
    assert result.returncode != 0
    assert "environment" in runtime_log(root).lower()


def build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "Source tree with spaces"
    root.mkdir()
    for folder in ("src/nc_factory", "scripts", "docs", "installer"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "pyproject.toml", "setup.cmd", "Start_NC_Voice_Dataset_Factory.cmd",
                 "constraints-models.txt", "requirements-core.txt", "requirements-dev.txt",
                 "requirements-models.txt", "requirements-export.txt", "requirements-speaker.txt",
                 "requirements-runtime-lock.txt"):
        shutil.copyfile(ROOT / name, root / name)
    (root / "src/nc_factory/__init__.py").write_text("__version__ = '0.1.0'\n")
    assets = root / "src/nc_factory/ui/assets"
    assets.mkdir(parents=True)
    for name in ("checkbox_round_active.svg", "checkbox_round_inactive.svg"):
        (assets / name).write_bytes(b"fixture asset; not rendered")
    for name in ("setup.ps1", "setup_windows.ps1", "start.ps1", "install_runtime.ps1"):
        (root / "scripts" / name).write_text("exit 0\n")
    for name in ("INSTALLATION.md", "USER_GUIDE.md", "BUILD_INSTALLER.md", "VALIDATION.md", "VOICE_AUTHORIZATION.md"):
        (root / "docs" / name).write_text("Installation guide fixture\n")
    for name in ("NCVoiceDatasetFactory.iss", "INSTALL.txt", "Repair_Setup.cmd"):
        (root / "installer" / name).write_text("Fixture\n")
    builder = ROOT / "scripts/build_installer.ps1"
    if builder.exists():
        shutil.copyfile(builder, root / "scripts/build_installer.ps1")
    compiler = root / "fake compiler.ps1"
    compiler.write_text(
        """$output = ($args | Where-Object { $_ -like '/O*' }).Substring(2)
$filename = ($args | Where-Object { $_ -like '/F*' }).Substring(2)
$payload = ($args | Where-Object { $_ -like '/DPayloadRoot=*' }).Substring(14)
$args | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'compiler-arguments.json')
Get-ChildItem -LiteralPath $payload -Recurse -File | ForEach-Object { $_.FullName.Substring($payload.Length + 1) } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'payload-files.json')
Set-Content -Encoding ASCII -LiteralPath (Join-Path $output ($filename + '.exe')) -Value 'compiled setup fixture'
exit 0
""",
        encoding="utf-8",
    )
    return root, compiler


def test_installer_build_excludes_local_data_and_writes_matching_checksum(tmp_path):
    root, compiler = build_fixture(tmp_path)
    for name in (".env", "projects/private.wav", "DATASET/private.wav", ".venv/private.exe",
                 "src/nc_factory/__pycache__/private.pyc", "src/nc_factory/private.local.json",
                 "docs/private.log", "dist/old-setup.exe"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not be packaged")
    output = tmp_path / "Output with spaces"
    result = run_powershell(root / "scripts/build_installer.ps1", "-CompilerPath", str(compiler),
                            "-OutputDirectory", str(output), cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    executable = output / "NC-Voice-Dataset-Factory-0.1.0-Setup.exe"
    assert executable.is_file()
    assert executable.with_suffix(".exe.sha256").read_text().split()[0] == hashlib.sha256(
        executable.read_bytes()
    ).hexdigest()
    files = json.loads((root / "payload-files.json").read_text(encoding="utf-8-sig"))
    assert "src\\nc_factory\\__init__.py" in files
    assert "scripts\\install_runtime.ps1" in files
    assert "src\\nc_factory\\ui\\assets\\checkbox_round_active.svg" in files
    assert "src\\nc_factory\\ui\\assets\\checkbox_round_inactive.svg" in files
    assert "INSTALL.txt" in files and "Repair_Setup.cmd" in files
    assert not any("private" in name or name.startswith("dist\\") for name in files)
    for link in re.findall(r"\]\(([^)]+)\)", (root / "README.md").read_text(encoding="utf-8")):
        if not link.startswith(("https://", "http://", "#")):
            assert link.replace("/", "\\") in files, f"Missing linked documentation: {link}"


def test_installer_build_fails_when_compiler_fails(tmp_path):
    root, compiler = build_fixture(tmp_path)
    compiler.write_text("exit 12\n")
    result = run_powershell(root / "scripts/build_installer.ps1", "-CompilerPath", str(compiler))
    assert result.returncode != 0
    assert "compiler failed" in (result.stdout + result.stderr).lower()
    assert not list((root / "dist").glob("*.sha256"))


@pytest.mark.parametrize("runtime_exit", [0, 9])
def test_compiled_installer_shortcuts_failure_exit_and_data_preservation(tmp_path, runtime_exit):
    compiler = Path(os.environ.get(
        "NC_INSTALLER_COMPILER", ROOT / ".artifacts/installer-tools/Inno Setup 6/ISCC.exe"
    ))
    if not compiler.is_file():
        pytest.skip("Set NC_INSTALLER_COMPILER to ISCC.exe for compiled-installer integration tests")
    root, _ = build_fixture(tmp_path)
    identity = f"NCInstallerTest-{uuid.uuid4().hex}"
    (root / "installer/NCVoiceDatasetFactory.iss").write_text(
        f'#define AppId "{identity}"\n#define AppGroup "{identity}"\n'
        f'#include "{ROOT / "installer/NCVoiceDatasetFactory.iss"}"\n',
        encoding="utf-8",
    )
    (root / "scripts/install_runtime.ps1").write_text(f"exit {runtime_exit}\n")
    built = run_powershell(root / "scripts/build_installer.ps1", "-CompilerPath", str(compiler))
    assert built.returncode == 0, built.stdout + built.stderr
    executable = root / "dist/NC-Voice-Dataset-Factory-0.1.0-Setup.exe"
    destination = tmp_path / "Installed app with spaces"
    log = tmp_path / "installation.log"
    shortcuts = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs" / identity
    arguments = [str(executable), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/SP-", "/NORESTART",
                 f"/DIR={destination}", f"/GROUP={identity}", f"/LOG={log}"]
    preserved = ("projects/keep.txt", "models/keep.bin", "lfm3-training/runs/keep.txt",
                 ".venv/keep.txt", "exports/keep.txt", "setup-logs/keep.log")
    try:
        installed = subprocess.run(arguments, capture_output=True, timeout=45)
        assert installed.returncode == (0 if runtime_exit == 0 else 1), log.read_text(errors="replace")
        assert (destination / "src/nc_factory/__init__.py").is_file()
        assert (shortcuts / "NC Voice Dataset Factory.lnk").is_file()
        assert (shortcuts / "Repair setup.lnk").is_file()
        assert (shortcuts / "Installation help.lnk").is_file()
        if runtime_exit:
            assert "Dependency setup failed" in log.read_text(encoding="utf-8-sig", errors="replace")
        for name in preserved:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"user data must survive")
        if runtime_exit == 0:
            reinstalled = subprocess.run(arguments, capture_output=True, timeout=45)
            assert reinstalled.returncode == 0
            assert all((destination / name).read_bytes() == b"user data must survive" for name in preserved)
    finally:
        uninstaller = destination / "unins000.exe"
        if uninstaller.exists():
            removed = subprocess.run(
                [str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
                capture_output=True, timeout=45,
            )
            assert removed.returncode == 0
    assert not (destination / "src/nc_factory/__init__.py").exists()
    assert not (shortcuts / "NC Voice Dataset Factory.lnk").exists()
    assert all((destination / name).read_bytes() == b"user data must survive" for name in preserved)
