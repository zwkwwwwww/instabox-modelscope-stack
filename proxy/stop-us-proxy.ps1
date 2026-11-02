$ErrorActionPreference = "SilentlyContinue"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfg = Join-Path $dir "config.ps1"
if (-not (Test-Path $cfg)) {
    Write-Host "[FAIL] missing config.ps1; copy config.example.ps1 and edit it"
    exit 1
}
. $cfg

$conns = Get-NetTCPConnection -LocalPort $ProxyPort -State Listen -ErrorAction SilentlyContinue
if ($conns) {
    foreach ($c in $conns) {
        $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
        if ($p) {
            Write-Host "[STOP] $($p.ProcessName) (PID $($c.OwningProcess))"
            Stop-Process -Id $c.OwningProcess -Force
        }
    }
} else {
    Write-Host "[INFO] nothing listening on $ProxyPort"
}

$orphans = Get-CimInstance Win32_Process -Filter "Name='instabox.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match ("port-forward\s+" + [regex]::Escape($InstaboxBox) + "\s+" + $ProxyPort + ":") }
foreach ($o in $orphans) {
    Write-Host "[STOP] instabox orphan (PID $($o.ProcessId))"
    Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Host "[OK] proxy stopped"
