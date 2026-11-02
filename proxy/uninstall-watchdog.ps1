$ErrorActionPreference = "SilentlyContinue"
$task = "US-111-Proxy-Watchdog"
schtasks /Delete /TN $task /F
Write-Host "[OK] watchdog task removed"
