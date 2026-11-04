#ifndef PayloadRoot
  #error Build using scripts/build_installer.ps1
#endif
#ifndef AppVersion
  #error AppVersion must be supplied by the build script
#endif
#ifndef AppId
  #define AppId "NeuralCompanion.NCVoiceDatasetFactory"
#endif
#ifndef AppGroup
  #define AppGroup "NC Voice Dataset Factory"
#endif

[Setup]
AppId={#AppId}
AppName=NC Voice Dataset Factory
AppVersion={#AppVersion}
AppPublisher=NeuralCompanion
AppPublisherURL=https://github.com/lainollll/nc-voice-dataset-factory
DefaultDirName={localappdata}\Programs\NC Voice Dataset Factory
DefaultGroupName={#AppGroup}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
MinVersion=10.0.22000
DisableDirPage=no
DisableProgramGroupPage=yes
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
OutputBaseFilename=NC-Voice-Dataset-Factory-{#AppVersion}-Setup
InfoBeforeFile={#PayloadRoot}\INSTALL.txt
SetupLogging=yes
CloseApplications=no
RestartApplications=no
UninstallDisplayName=NC Voice Dataset Factory
ExtraDiskSpaceRequired=21474836480

[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#PayloadRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\NC Voice Dataset Factory"; Filename: "{app}\Start_NC_Voice_Dataset_Factory.cmd"; WorkingDir: "{app}"
Name: "{group}\Repair setup"; Filename: "{app}\Repair_Setup.cmd"; WorkingDir: "{app}"
Name: "{group}\Installation help"; Filename: "{sys}\notepad.exe"; Parameters: """{app}\INSTALL.txt"""; WorkingDir: "{app}"
Name: "{autodesktop}\NC Voice Dataset Factory"; Filename: "{app}\Start_NC_Voice_Dataset_Factory.cmd"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Start_NC_Voice_Dataset_Factory.cmd"; Description: "Open NC Voice Dataset Factory"; Flags: postinstall shellexec skipifsilent; Check: RuntimeReady

[Code]
var
  RuntimeInstalled: Boolean;

function RuntimeReady: Boolean;
begin
  Result := RuntimeInstalled;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ExitCode, ShowCommand: Integer;
  Parameters: String;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := 'Downloading and installing application dependencies...';
    Parameters := '-NoLogo -NoProfile -ExecutionPolicy Bypass -File "' +
      ExpandConstant('{app}\scripts\install_runtime.ps1') + '"';
    ShowCommand := SW_SHOWNORMAL;
    if WizardSilent then ShowCommand := SW_HIDE;
    RuntimeInstalled := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      Parameters, ExpandConstant('{app}'), ShowCommand, ewWaitUntilTerminated, ExitCode);
    RuntimeInstalled := RuntimeInstalled and (ExitCode = 0);
    if not RuntimeInstalled then
      Log('Dependency setup failed. Review setup-logs and run Repair_Setup.cmd.');
  end;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpFinished) and not RuntimeInstalled then
  begin
    WizardForm.FinishedHeadingLabel.Caption := 'Setup needs attention';
    WizardForm.FinishedLabel.Caption :=
      'Application files were copied, but dependency setup failed. The app is not ready.' + #13#10#13#10 +
      'Open Repair setup from the Start menu, or run Repair_Setup.cmd in the installation folder. ' +
      'Review the setup-logs folder for the error details.';
  end;
end;

function GetCustomSetupExitCode: Integer;
begin
  if RuntimeInstalled then Result := 0 else Result := 1;
end;
