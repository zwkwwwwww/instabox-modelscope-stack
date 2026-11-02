$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$vbs = Join-Path $dir "watchdog-hidden.vbs"
$startup = [Environment]::GetFolderPath("Startup")
$lnk = Join-Path $startup "US-111-Proxy.lnk"

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnk)
$sc.TargetPath = "wscript.exe"
$sc.Arguments = '"' + $vbs + '"'
$sc.WorkingDirectory = $dir
$sc.WindowStyle = 7
$sc.Description = "US-111 proxy (hidden)"
$sc.Save()
Write-Host "[OK] autostart installed: $lnk"
Write-Host "Remove with: Remove-Item '$lnk'"
