$ErrorActionPreference = "SilentlyContinue"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$vbs = Join-Path $dir "watchdog-hidden.vbs"
$task = "US-111-Proxy-Watchdog"

if (-not (Test-Path $vbs)) {
    Write-Host "[FAIL] watchdog-hidden.vbs not found"
    exit 1
}

$tr = 'wscript.exe "' + $vbs + '"'
schtasks /Create /TN $task /TR $tr /SC MINUTE /MO 1 /F
schtasks /Query /TN $task /FO LIST
Write-Host ""
Write-Host "Remove with: schtasks /Delete /TN $task /F"
