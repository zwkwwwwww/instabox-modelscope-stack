# Copy this file to config.ps1 and fill in your own values.
# config.ps1 is git-ignored.

$InstaboxBox = "111"
$ProxyPort   = 3128
$ProxyUser   = "proxyuser"
$ProxyPass   = "change-me"

# Path to the instabox CLI. Default location on Windows:
$InstaboxExe = Join-Path $env:LOCALAPPDATA "Programs\instabox\instabox.exe"
