; Concert Remaster installer (NSIS 3). Build it with windows/build.sh, which stages the files.
;
; Installs the app for the current user (no admin rights needed), then runs the app's own
; setup, which downloads Python, the AI libraries for the chosen graphics card, ffmpeg and
; every AI model. Its progress shows live in the installer window.

Target amd64-unicode   ; the app itself is 64-bit only (PyTorch)
!include "MUI2.nsh"
!include "Sections.nsh"
!include "LogicLib.nsh"

!ifndef STAGE
  !error "Build with windows/build.sh (it passes -DSTAGE=<staged files> -DOUTFILE=<exe> -DVERSION=<x.y.z>)"
!endif

!define APP "Concert Remaster"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\ConcertRemaster"

Name "${APP}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\${APP}"
InstallDirRegKey HKCU "Software\ConcertRemaster" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
BrandingText "${APP} ${VERSION}"
ShowInstDetails show
ShowUninstDetails show
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${APP}"
VIAddVersionKey "FileDescription" "${APP} installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "MIT licence"

!define MUI_ICON "${STAGE}\app.ico"
!define MUI_UNICON "${STAGE}\app.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Welcome to ${APP} ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "Make your concert recordings sound like the released songs, on your own PC.$\r$\n$\r$\nThis installs the app and then downloads what it needs, once: Python, the AI libraries for your graphics card, ffmpeg and the AI models (about 20 GB in total, so allow some time on a slow connection).$\r$\n$\r$\nAfter that, everything runs offline. The internet is only used, if you allow it, to look songs up and download their original versions for comparison.$\r$\n$\r$\nIf the download is interrupted, just start ${APP}: it offers to continue where it stopped."
!define MUI_COMPONENTSPAGE_SMALLDESC
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP}.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Start ${APP}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Var GpuChoice

Section "!${APP}" SecApp
  SectionIn RO
  AddSize 6000000  ; Python + AI libraries, downloaded by the setup step
  SetOutPath "$INSTDIR"
  File "/oname=${APP}.exe" "${STAGE}\${APP}.exe"
  SetOutPath "$INSTDIR\concert-remaster"
  File /r "${STAGE}\concert-remaster\*.*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\ConcertRemaster" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${APP}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "${APP}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\${APP}.exe,0"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" 20000000
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
  CreateDirectory "$SMPROGRAMS\${APP}"
  CreateShortcut "$SMPROGRAMS\${APP}\${APP}.lnk" "$INSTDIR\${APP}.exe" "" "$INSTDIR\${APP}.exe" 0
  CreateShortcut "$SMPROGRAMS\${APP}\Uninstall ${APP}.lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Desktop shortcut" SecDesktop
  CreateShortcut "$DESKTOP\${APP}.lnk" "$INSTDIR\${APP}.exe" "" "$INSTDIR\${APP}.exe" 0
SectionEnd

SectionGroup /e "Graphics card for the AI" SecGpu
  Section "Detect automatically" SecAuto
  SectionEnd
  Section /o "NVIDIA (CUDA)" SecNvidia
  SectionEnd
  Section /o "AMD / Intel (DirectML)" SecAmd
  SectionEnd
  Section /o "No graphics card (CPU only)" SecCpu
  SectionEnd
SectionGroupEnd

Section "Download all AI models now (about 12 GB)" SecModels
  AddSize 12500000
SectionEnd

Section "-Setup"
  StrCpy $0 "auto"
  ${If} ${SectionIsSelected} ${SecNvidia}
    StrCpy $0 "nvidia"
  ${ElseIf} ${SectionIsSelected} ${SecAmd}
    StrCpy $0 "amd"
  ${ElseIf} ${SectionIsSelected} ${SecCpu}
    StrCpy $0 "cpu"
  ${EndIf}
  StrCpy $1 ""
  ${IfNot} ${SectionIsSelected} ${SecModels}
    StrCpy $1 "-SkipModels"
  ${EndIf}
  DetailPrint "Setting up Python, the AI libraries ($0) and the models. This takes a while; the details follow."
  System::Call 'Kernel32::SetEnvironmentVariable(t "TQDM_DISABLE", t "1")i'
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\concert-remaster\windows\setup.ps1" -Gpu $0 $1 -NoShortcut'
  Pop $2
  ${If} $2 != 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "The setup did not finish (usually the internet connection or disk space).$\r$\n$\r$\nStart ${APP} when ready: it offers to continue where it stopped.$\r$\n$\r$\nDetails: $INSTDIR\concert-remaster\setup.log" /SD IDOK
  ${EndIf}
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecApp} "The app, its launcher and Start-menu entries."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "A ${APP} icon on the desktop."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecGpu} "Which processor runs the AI models. Automatic picks NVIDIA (fastest), then AMD/Intel, then the CPU."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecAuto} "Pick the best available: NVIDIA via CUDA, AMD/Intel via DirectML, otherwise the CPU."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecNvidia} "GeForce/RTX cards (e.g. an RTX 3060 laptop): by far the fastest."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecAmd} "Radeon cards (e.g. RX 580) and Intel graphics through DirectML. Models that don't run there fall back to the CPU."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCpu} "Works everywhere, but separation is many times slower."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecModels} "Recommended. Without it, each model downloads the first time it is used."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Function .onInit
  StrCpy $GpuChoice ${SecAuto}
  FindWindow $0 "" "${APP}"
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "Please close ${APP} before installing." /SD IDOK
    Abort
  ${EndIf}
FunctionEnd

Function .onSelChange
  !insertmacro StartRadioButtons $GpuChoice
    !insertmacro RadioButton ${SecAuto}
    !insertmacro RadioButton ${SecNvidia}
    !insertmacro RadioButton ${SecAmd}
    !insertmacro RadioButton ${SecCpu}
  !insertmacro EndRadioButtons
FunctionEnd

Function un.onInit
  FindWindow $0 "" "${APP}"
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "Please close ${APP} before uninstalling." /SD IDOK
    Abort
  ${EndIf}
FunctionEnd

Section "Uninstall"
  Delete "$DESKTOP\${APP}.lnk"
  RMDir /r "$SMPROGRAMS\${APP}"
  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 "Also delete your projects, exported songs and the downloaded AI models?$\r$\n$\r$\nChoose No to keep them in $INSTDIR\concert-remaster for a later reinstall." /SD IDNO IDYES everything
    RMDir /r "$INSTDIR\concert-remaster\.venv"
    RMDir /r "$INSTDIR\concert-remaster\tools"
    RMDir /r "$INSTDIR\concert-remaster\src"
    RMDir /r "$INSTDIR\concert-remaster\windows"
    RMDir /r "$INSTDIR\concert-remaster\window"
    RMDir /r "$INSTDIR\concert-remaster\logs"
    Delete "$INSTDIR\concert-remaster\pyproject.toml"
    Delete "$INSTDIR\concert-remaster\README.md"
    Delete "$INSTDIR\concert-remaster\setup.log"
    Delete "$INSTDIR\concert-remaster\gpu.txt"
    Goto finish
  everything:
    RMDir /r "$INSTDIR\concert-remaster"
  finish:
  Delete "$INSTDIR\${APP}.exe"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINST_KEY}"
  DeleteRegKey HKCU "Software\ConcertRemaster"
SectionEnd
