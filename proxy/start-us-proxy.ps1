$ErrorActionPreference = "SilentlyContinue"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfg = Join-Path $dir "config.ps1"
if (-not (Test-Path $cfg)) {
    Write-Host "[FAIL] missing config.ps1; copy config.example.ps1 and edit it"
    exit 1
}
. $cfg

$log = Join-Path $dir "forward.log"
$proxy = "http://${ProxyUser}:${ProxyPass}@127.0.0.1:${ProxyPort}"
$lock = Join-Path $dir "start.lock"

function Test-ProxyHealth {
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { return $true }
    $code = & curl.exe -s -o NUL -w "%{http_code}" --max-time 25 -x $proxy "https://www.google.com/generate_204" 2>$null
    return ([string]$code -eq "204")
}

function Stop-Forward {
    $conns = Get-NetTCPConnection -LocalPort $ProxyPort -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }
    $orphans = Get-CimInstance Win32_Process -Filter "Name='instabox.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match ("port-forward\s+" + [regex]::Escape($InstaboxBox) + "\s+" + $ProxyPort + ":") }
    foreach ($o in $orphans) { Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Wait-Port {
    param([int]$Seconds)
    for ($i = 0; $i -lt $Seconds; $i++) {
        Start-Sleep -Seconds 1
        if (Get-NetTCPConnection -LocalPort $ProxyPort -State Listen -ErrorAction SilentlyContinue) { return $true }
    }
    return $false
}

# Free-plan runtime pauses enforce a cooldown during which no wake-up is possible.
# Detect it first so the watchdog exits quickly and does not hold the lock.
if (Test-Path $InstaboxExe) {
    $raw = & $InstaboxExe get $InstaboxBox --json 2>$null | Out-String
    $st = $null
    try { $st = $raw | ConvertFrom-Json } catch { }
    if ($st -is [array]) { $st = $st[0] }
    if ($st -and $st.cooldownUntil) {
        $until = ([datetime]::Parse($st.cooldownUntil)).ToLocalTime()
        if ($until -gt (Get-Date)) {
            $left = [int](($until - (Get-Date)).TotalSeconds)
            Write-Host ("[WAIT] box cooling down until {0:HH:mm:ss} ({1}s left)" -f $until, $left)
            exit 0
        }
    }
}

# One start run at a time: the watchdog and a manual click can otherwise kill
# each other's ssh process. A stale lock (dead PID) is ignored.
if (Test-Path $lock) {
    $other = Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($other -and ($other -ne $PID) -and (Get-Process -Id ([int]$other) -ErrorAction SilentlyContinue)) {
        Write-Host "[SKIP] another start run is in progress (PID $other)"
        exit 0
    }
}
Set-Content -Path $lock -Value $PID -Encoding ASCII

# Idempotent, but verify the tunnel is actually passing traffic:
# a stale ssh can keep the port listening while every request hangs.
if (Get-NetTCPConnection -LocalPort $ProxyPort -State Listen -ErrorAction SilentlyContinue) {
    if (Test-ProxyHealth) {
        Write-Host "[OK] proxy already running on 127.0.0.1:$ProxyPort (healthy)"
        exit 0
    }
    Write-Host "[WARN] port $ProxyPort is listening but the tunnel is unhealthy; restarting it"
    Stop-Forward
    Start-Sleep -Seconds 3
}

# Under TUN mode the box hostname resolves to a fake-ip by design; the router
# must send DOMAIN-SUFFIX,<your-box-domain>,DIRECT so the SSH dial stays direct.

# Wake a paused box (30 min idle or the free tier's 480-minute runtime limit).
# --no-wait returns immediately; during cooldown this fails harmlessly.
if (Test-Path $InstaboxExe) {
    $rp = Start-Process -FilePath $InstaboxExe -ArgumentList "resume", $InstaboxBox, "--no-wait" -NoNewWindow -PassThru -ErrorAction SilentlyContinue
    if ($rp) { $null = $rp.WaitForExit(8000) }
}

# Generate a hidden launcher from the template so no console window is shown.
$vbsTemplate = Join-Path $dir "forward-hidden.vbs.template"
$vbs = Join-Path $dir "forward-hidden.vbs"
if (Test-Path $vbsTemplate) {
    $t = Get-Content $vbsTemplate -Raw
    $t = $t.Replace("__EXE__", $InstaboxExe).Replace("__BOX__", $InstaboxBox)
    $t = $t.Replace("__PORT__", [string]$ProxyPort).Replace("__LOG__", $log)
    Set-Content -Path $vbs -Value $t -Encoding ASCII
}

$wscript = Join-Path $env:WINDIR "System32\wscript.exe"
$useVbs = (Test-Path $vbs)

# Two attempts: at logon the network may not be ready for the first one.
for ($attempt = 1; $attempt -le 2; $attempt++) {
    if ($useVbs) {
        Start-Process -FilePath $wscript -ArgumentList "`"$vbs`"" -WindowStyle Hidden -ErrorAction SilentlyContinue | Out-Null
    } else {
        $cmdline = "cmd.exe /c """"$InstaboxExe"" port-forward $InstaboxBox ${ProxyPort}:${ProxyPort} >> ""$log"" 2>&1"""
        $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdline }
        if ($r.ReturnValue -ne 0) {
            Write-Host "[FAIL] Win32_Process.Create returned $($r.ReturnValue)"
            exit 1
        }
    }
    if (Wait-Port -Seconds 30) {
        Start-Sleep -Seconds 5
        if (Test-ProxyHealth) {
            Write-Host "[OK] proxy started on 127.0.0.1:$ProxyPort"
            exit 0
        }
        Start-Sleep -Seconds 15
        if (Test-ProxyHealth) {
            Write-Host "[OK] proxy started on 127.0.0.1:$ProxyPort (after warm-up)"
            exit 0
        }
        Start-Sleep -Seconds 15
        if (Test-ProxyHealth) {
            Write-Host "[OK] proxy started on 127.0.0.1:$ProxyPort (after second warm-up)"
            exit 0
        }
        Write-Host "[WARN] attempt ${attempt}: tunnel up but not passing traffic"
    } else {
        Write-Host "[WARN] attempt ${attempt}: port did not come up"
    }
    Stop-Forward
    Start-Sleep -Seconds 5
}

Write-Host "[FAIL] port $ProxyPort not usable. Log: $log"
exit 1
