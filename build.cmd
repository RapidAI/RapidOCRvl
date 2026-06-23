@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Build Windows binaries for this repository.
rem Usage:
rem   build.cmd [amd64^|arm64] [cli^|all^|nsis] [version] [makensis]
rem
rem Defaults:
rem   arch = amd64
rem   mode = all
rem   version = 1.0.0
rem   makensis = makensis

set "ARCH=%~1"
set "MODE=%~2"
set "VERSION=%~3"
set "MAKENSIS=%~4"

if "%ARCH%"=="" set "ARCH=amd64"
if "%MODE%"=="" set "MODE=all"
if "%VERSION%"=="" set "VERSION=1.0.0"
if "%MAKENSIS%"=="" set "MAKENSIS=makensis"

if /I not "%ARCH%"=="amd64" if /I not "%ARCH%"=="arm64" (
  echo Unsupported architecture: %ARCH%
  echo Usage: build.cmd [amd64^|arm64] [cli^|all^|nsis] [version] [makensis]
  exit /b 2
)

if /I not "%MODE%"=="cli" if /I not "%MODE%"=="all" if /I not "%MODE%"=="nsis" (
  echo Unsupported mode: %MODE%
  echo Usage: build.cmd [amd64^|arm64] [cli^|all^|nsis] [version] [makensis]
  exit /b 2
)

pushd "%~dp0" || exit /b 1

where go >nul 2>nul
if errorlevel 1 (
  echo Required tool not found on PATH: go
  popd
  exit /b 1
)

if /I "%MODE%"=="nsis" goto :build_nsis

set "DIST=%CD%\dist\windows\%ARCH%"
if not exist "%DIST%" mkdir "%DIST%"

set "OLD_GOOS=%GOOS%"
set "OLD_GOARCH=%GOARCH%"
set "OLD_CGO_ENABLED=%CGO_ENABLED%"
set "GOOS=windows"
set "GOARCH=%ARCH%"
set "CGO_ENABLED=0"
set "LDFLAGS=-s -w"

echo Building Go binaries for windows/%ARCH%...

call :build_go paddleocrvl-server   .\cmd\paddleocrvl-server   || goto :fail
call :build_go paddleocrvl-go       .\cmd\paddleocrvl-go       || goto :fail
call :build_go paddleocrvl-download .\cmd\paddleocrvl-download || goto :fail
call :build_go paddleocrvl-inspect  .\cmd\paddleocrvl-inspect  || goto :fail
call :build_go paddleocrvl-convert  .\cmd\paddleocrvl-convert  || goto :fail
call :build_go paddleocrvl-bench    .\cmd\paddleocrvl-bench    || goto :fail

if /I "%MODE%"=="all" (
  where wails >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Wails was not found on PATH; skipped paddleocrvl-client.exe.
    echo Install Wails and run this script again, or use: build.cmd %ARCH% cli
  ) else (
    echo.
    echo Building Wails client for windows/%ARCH%...
    pushd "%CD%\cmd\paddleocrvl-client" || goto :fail
    wails build -platform "windows/%ARCH%" -clean
    if errorlevel 1 (
      popd
      goto :fail
    )
    popd

    if exist "%CD%\cmd\paddleocrvl-client\build\bin\paddleocrvl-client.exe" (
      copy /Y "%CD%\cmd\paddleocrvl-client\build\bin\paddleocrvl-client.exe" "%DIST%\paddleocrvl-client.exe" >nul
      if errorlevel 1 (
        echo Failed to copy paddleocrvl-client.exe to %DIST%.
        echo Close any running copy of the program and try again.
        goto :fail
      )
      echo   - %DIST%\paddleocrvl-client.exe
    ) else (
      echo Wails build completed, but the expected client binary was not found.
      goto :fail
    )
  )
)

goto :success

:build_nsis
where powershell >nul 2>nul
if errorlevel 1 (
  echo Required tool not found on PATH: powershell
  popd
  exit /b 1
)

echo Building NSIS installer for windows/%ARCH%, version %VERSION%...
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\packaging\windows\build-nsis.ps1" -Version "%VERSION%" -Arch "%ARCH%" -Makensis "%MAKENSIS%"
if errorlevel 1 goto :fail_no_env

echo.
echo NSIS installer completed: %CD%\dist\windows
popd
exit /b 0

:build_go
set "NAME=%~1"
set "PKG=%~2"
go build -trimpath -ldflags "%LDFLAGS%" -o "%DIST%\%NAME%.exe" "%PKG%"
if errorlevel 1 exit /b 1
echo   - %DIST%\%NAME%.exe
exit /b 0

:success
call :restore_env
echo.
echo Build completed: %DIST%
popd
exit /b 0

:fail
set "BUILD_EXIT=%ERRORLEVEL%"
if "%BUILD_EXIT%"=="0" set "BUILD_EXIT=1"
call :restore_env
echo.
echo Build failed with exit code %BUILD_EXIT%.
popd
exit /b %BUILD_EXIT%

:fail_no_env
set "BUILD_EXIT=%ERRORLEVEL%"
if "%BUILD_EXIT%"=="0" set "BUILD_EXIT=1"
echo.
echo Build failed with exit code %BUILD_EXIT%.
popd
exit /b %BUILD_EXIT%

:restore_env
if "%OLD_GOOS%"=="" (
  set "GOOS="
) else (
  set "GOOS=%OLD_GOOS%"
)
if "%OLD_GOARCH%"=="" (
  set "GOARCH="
) else (
  set "GOARCH=%OLD_GOARCH%"
)
if "%OLD_CGO_ENABLED%"=="" (
  set "CGO_ENABLED="
) else (
  set "CGO_ENABLED=%OLD_CGO_ENABLED%"
)
exit /b 0
