Unicode true

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"

!ifndef ARG_CLIENT_BINARY
  !error "ARG_CLIENT_BINARY is required"
!endif
!ifndef ARG_SERVER_BINARY
  !error "ARG_SERVER_BINARY is required"
!endif
!ifndef ARG_DOWNLOAD_BINARY
  !error "ARG_DOWNLOAD_BINARY is required"
!endif

!ifndef INFO_PRODUCTVERSION
  !define INFO_PRODUCTVERSION "1.0.0"
!endif
!ifndef INFO_FILEVERSION
  !define INFO_FILEVERSION "1.0.0.0"
!endif
!ifndef INFO_COMPANYNAME
  !define INFO_COMPANYNAME "znsoft"
!endif
!ifndef INFO_PRODUCTNAME
  !define INFO_PRODUCTNAME "PaddleOCR-VL"
!endif
!ifndef INFO_ARCH
  !define INFO_ARCH "x64"
!endif
!ifndef OUTFILE
  !define OUTFILE "..\..\dist\windows\PaddleOCR-VL-${INFO_PRODUCTVERSION}-windows-${INFO_ARCH}-setup.exe"
!endif
!ifndef UNINST_KEY_NAME
  !define UNINST_KEY_NAME "${INFO_COMPANYNAME}${INFO_PRODUCTNAME}"
!endif
!ifndef ICON_FILE
  !define ICON_FILE "..\..\cmd\paddleocrvl-client\build\windows\icon.ico"
!endif

!define CLIENT_EXE "paddleocrvl-client.exe"
!define SERVER_EXE "paddleocrvl-server.exe"
!define DOWNLOAD_EXE "paddleocrvl-download.exe"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${UNINST_KEY_NAME}"

Name "${INFO_PRODUCTNAME}"
OutFile "${OUTFILE}"
InstallDir "$PROGRAMFILES64\${INFO_COMPANYNAME}\${INFO_PRODUCTNAME}"
RequestExecutionLevel admin
ShowInstDetails show
ShowUninstDetails show

VIProductVersion "${INFO_FILEVERSION}"
VIFileVersion "${INFO_FILEVERSION}"
VIAddVersionKey "CompanyName" "${INFO_COMPANYNAME}"
VIAddVersionKey "FileDescription" "${INFO_PRODUCTNAME} Installer"
VIAddVersionKey "ProductVersion" "${INFO_PRODUCTVERSION}"
VIAddVersionKey "FileVersion" "${INFO_PRODUCTVERSION}"
VIAddVersionKey "ProductName" "${INFO_PRODUCTNAME}"

!define MUI_ICON "${ICON_FILE}"
!define MUI_UNICON "${ICON_FILE}"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

!if "${INFO_ARCH}" == "x64"
  Function .onInit
    ${IfNot} ${IsNativeAMD64}
      MessageBox MB_OK "${INFO_PRODUCTNAME} ${INFO_ARCH} requires AMD64 Windows."
      Quit
    ${EndIf}
  FunctionEnd
!endif

Section "Install"
  SetShellVarContext all
  SetOutPath "$INSTDIR"

  IfFileExists "$INSTDIR\${SERVER_EXE}" 0 old_service_done
    DetailPrint "Stopping existing PaddleOCR-VL Windows service"
    ExecWait '$\"$INSTDIR\${SERVER_EXE}$\" service stop'
    DetailPrint "Removing existing PaddleOCR-VL Windows service"
    ExecWait '$\"$INSTDIR\${SERVER_EXE}$\" service uninstall'
  old_service_done:

  File "/oname=${CLIENT_EXE}" "${ARG_CLIENT_BINARY}"
  File "/oname=${SERVER_EXE}" "${ARG_SERVER_BINARY}"
  File "/oname=${DOWNLOAD_EXE}" "${ARG_DOWNLOAD_BINARY}"

  !ifdef ARG_WEBVIEW2_BOOTSTRAPPER
    InitPluginsDir
    File "/oname=$pluginsdir\MicrosoftEdgeWebview2Setup.exe" "${ARG_WEBVIEW2_BOOTSTRAPPER}"
    DetailPrint "Installing Microsoft Edge WebView2 Runtime"
    ExecWait '$\"$pluginsdir\MicrosoftEdgeWebview2Setup.exe$\" /silent /install'
  !endif

  CreateDirectory "$APPDATA\PaddleOCRVL"
  CreateDirectory "$APPDATA\PaddleOCRVL\models"

  DetailPrint "Ensuring PaddleOCR-VL model files are present"
  ExecWait '$\"$INSTDIR\${DOWNLOAD_EXE}$\" -out $\"$APPDATA\PaddleOCRVL\models$\"' $0
  ${If} $0 != 0
    DetailPrint "Model download failed with exit code $0"
    Abort "PaddleOCR-VL model download failed. Check network access and try again."
  ${EndIf}

  DetailPrint "Installing PaddleOCR-VL Windows service"
  ExecWait '$\"$INSTDIR\${SERVER_EXE}$\" service install -model-dir $\"$APPDATA\PaddleOCRVL\models$\" -admin-config $\"$APPDATA\PaddleOCRVL\paddleocrvl-admin.json$\" -addr 127.0.0.1:8080' $0
  ${If} $0 != 0
    DetailPrint "Service install failed with exit code $0"
    Abort "PaddleOCR-VL service install failed."
  ${Else}
    DetailPrint "Starting PaddleOCR-VL Windows service"
    ExecWait '$\"$INSTDIR\${SERVER_EXE}$\" service start' $0
    ${If} $0 != 0
      DetailPrint "Service start failed with exit code $0"
      Abort "PaddleOCR-VL service start failed."
    ${EndIf}
    DetailPrint "Waiting for PaddleOCR-VL service readiness"
    ExecWait '$\"$INSTDIR\${SERVER_EXE}$\" wait-ready -addr 127.0.0.1:8080 -timeout 30m' $0
    ${If} $0 != 0
      DetailPrint "Service readiness check failed with exit code $0"
      Abort "PaddleOCR-VL service did not become ready."
    ${EndIf}
  ${EndIf}

  CreateDirectory "$SMPROGRAMS\${INFO_PRODUCTNAME}"
  CreateShortcut "$SMPROGRAMS\${INFO_PRODUCTNAME}\${INFO_PRODUCTNAME} Client.lnk" "$INSTDIR\${CLIENT_EXE}"
  CreateShortcut "$DESKTOP\${INFO_PRODUCTNAME} Client.lnk" "$INSTDIR\${CLIENT_EXE}"

  WriteUninstaller "$INSTDIR\uninstall.exe"
  SetRegView 64
  WriteRegStr HKLM "${UNINST_KEY}" "Publisher" "${INFO_COMPANYNAME}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayName" "${INFO_PRODUCTNAME}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion" "${INFO_PRODUCTVERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\${CLIENT_EXE}"
  WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
  WriteRegStr HKLM "${UNINST_KEY}" "QuietUninstallString" "$\"$INSTDIR\uninstall.exe$\" /S"
SectionEnd

Section "Uninstall"
  SetShellVarContext all

  IfFileExists "$INSTDIR\${SERVER_EXE}" 0 service_done
    DetailPrint "Stopping PaddleOCR-VL Windows service"
    ExecWait '$\"$INSTDIR\${SERVER_EXE}$\" service stop'
    DetailPrint "Removing PaddleOCR-VL Windows service"
    ExecWait '$\"$INSTDIR\${SERVER_EXE}$\" service uninstall'
  service_done:

  Delete "$SMPROGRAMS\${INFO_PRODUCTNAME}\${INFO_PRODUCTNAME} Client.lnk"
  RMDir "$SMPROGRAMS\${INFO_PRODUCTNAME}"
  Delete "$DESKTOP\${INFO_PRODUCTNAME} Client.lnk"
  Delete "$INSTDIR\${CLIENT_EXE}"
  Delete "$INSTDIR\${SERVER_EXE}"
  Delete "$INSTDIR\${DOWNLOAD_EXE}"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"

  SetRegView 64
  DeleteRegKey HKLM "${UNINST_KEY}"
SectionEnd
