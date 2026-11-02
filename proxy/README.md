# proxy

Windows 端 InstaBox 出口代理脚本。

```powershell
copy config.example.ps1 config.ps1   # fill in box/user/pass
powershell -ExecutionPolicy Bypass -File .\start-us-proxy.ps1
powershell -ExecutionPolicy Bypass -File .\install-watchdog.ps1
```

完整说明见 [../docs/instabox-proxy.md](../docs/instabox-proxy.md)。
