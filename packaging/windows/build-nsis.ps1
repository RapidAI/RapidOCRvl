param(
  [string]$Version = "1.0.0",
  [ValidateSet("amd64", "arm64")]
  [string]$Arch = "amd64",
  [string]$WebView2Bootstrapper = "",
  [string]$Makensis = "makensis"
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$dist = Join-Path $repo "dist\windows"
$server = Join-Path $dist "paddleocrvl-server.exe"
$download = Join-Path $dist "paddleocrvl-download.exe"
$client = Join-Path $repo "cmd\paddleocrvl-client\build\bin\paddleocrvl-client.exe"
$installerArch = if ($Arch -eq "amd64") { "x64" } else { "arm64" }
$installer = Join-Path $dist "PaddleOCR-VL-$Version-windows-$installerArch-setup.exe"
$nsi = Join-Path $PSScriptRoot "paddleocrvl.nsi"
$icon = Join-Path $repo "cmd\paddleocrvl-client\build\windows\icon.ico"
$versionNumbers = @([regex]::Matches(($Version -replace '^[vV]', '').Split('-')[0].Split('+')[0], '\d+') | Select-Object -First 4 | ForEach-Object { $_.Value })
while ($versionNumbers.Count -lt 4) {
  $versionNumbers += "0"
}
$fileVersion = ($versionNumbers[0..3] -join ".")

New-Item -ItemType Directory -Force -Path $dist | Out-Null

$oldGoos = $env:GOOS
$oldGoarch = $env:GOARCH

Push-Location $repo
try {
  foreach ($tool in @("go", "wails", $Makensis)) {
    if (!(Get-Command $tool -ErrorAction SilentlyContinue)) {
      throw "Required tool not found on PATH: $tool"
    }
  }
  if ($WebView2Bootstrapper -ne "" -and !(Test-Path $WebView2Bootstrapper)) {
    throw "WebView2 bootstrapper not found: $WebView2Bootstrapper"
  }
  if (!(Test-Path $icon)) {
    throw "Installer icon not found: $icon"
  }

  $env:GOOS = "windows"
  $env:GOARCH = $Arch
  go build -trimpath -ldflags "-s -w" -o $server .\cmd\paddleocrvl-server
  go build -trimpath -ldflags "-s -w" -o $download .\cmd\paddleocrvl-download

  Push-Location (Join-Path $repo "cmd\paddleocrvl-client")
  try {
    wails build -platform "windows/$Arch" -clean
  } finally {
    Pop-Location
  }

  if (!(Test-Path $client)) {
    throw "Wails client binary not found: $client"
  }

  $args = @(
    "-DARG_CLIENT_BINARY=$client",
    "-DARG_SERVER_BINARY=$server",
    "-DARG_DOWNLOAD_BINARY=$download",
    "-DINFO_PRODUCTVERSION=$Version",
    "-DINFO_FILEVERSION=$fileVersion",
    "-DINFO_ARCH=$installerArch",
    "-DICON_FILE=$icon",
    "-DOUTFILE=$installer"
  )
  if ($WebView2Bootstrapper -ne "") {
    $args += "-DARG_WEBVIEW2_BOOTSTRAPPER=$WebView2Bootstrapper"
  }
  $args += $nsi

  & $Makensis @args
  if ($LASTEXITCODE -ne 0) {
    throw "makensis failed with exit code $LASTEXITCODE"
  }

  Write-Host "Created $installer"
} finally {
  Pop-Location
  if ($null -eq $oldGoos) { Remove-Item Env:\GOOS -ErrorAction SilentlyContinue } else { $env:GOOS = $oldGoos }
  if ($null -eq $oldGoarch) { Remove-Item Env:\GOARCH -ErrorAction SilentlyContinue } else { $env:GOARCH = $oldGoarch }
}
