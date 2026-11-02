# InstaBox 美国出口代理

## 前置条件

- InstaBox 账号和一台盒子（本文以免费版 4C/8G 为例）
- Windows + OpenSSH + PowerShell 5.1+
- Clash Verge（或任何支持 HTTP 代理的客户端）
- 可选：Python 3.10+（桌面面板）

## 1. 在盒子上装 gost

SSH 进盒子（或用 `instabox exec`），下载 gost v3：

```bash
mkdir -p /data/bin
cd /tmp
curl -fL -o gost.tar.gz \
  https://github.com/go-gost/gost/releases/download/v3.0.0/gost_3.0.0_linux_amd64.tar.gz
tar -xzf gost.tar.gz -C /data/bin gost
chmod +x /data/bin/gost
```

创建 systemd 服务（把 `USER` / `PASS` 换成你自己的）：

```ini
# /etc/systemd/system/gost.service
[Unit]
Description=gost http proxy (US egress)
After=network-online.target

[Service]
ExecStart=/data/bin/gost -L http://USER:PASS@:3128
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now gost
ss -tlnp | grep 3128
```

> 免费版不开放原始 TCP 端口，所以 3128 只能通过 SSH 隧道访问，不会直接暴露在公网。

## 2. Windows 端配置

把 `proxy/config.example.ps1` 复制为 `proxy/config.ps1`，填写：

```powershell
$InstaboxBox = "111"          # 你的盒子名/ID
$ProxyPort   = 3128
$ProxyUser   = "proxyuser"
$ProxyPass   = "change-me"
$InstaboxExe = Join-Path $env:LOCALAPPDATA "Programs\instabox\instabox.exe"
```

启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\proxy\start-us-proxy.ps1
```

看到 `[OK] proxy started on 127.0.0.1:3128` 即可。验证：

```powershell
curl.exe -x http://proxyuser:change-me@127.0.0.1:3128 https://api.ipify.org
```

返回盒子的出口 IP 就成功了。

## 3. Clash 节点与 TUN

```yaml
proxies:
  - name: US-Box
    type: http
    server: 127.0.0.1
    port: 3128
    username: proxyuser
    password: change-me
    udp: false
```

TUN 设置：

```yaml
tun:
  enable: true
  stack: gvisor
  auto-route: true
  dns-hijack:
    - any:53
```

DNS（TUN 模式下必须用可直连的加密 DNS）：

```yaml
dns:
  enable: true
  enhanced-mode: fake-ip
  fake-ip-range: 198.18.0.1/16
  default-nameserver: [223.5.5.5, 119.29.29.29]
  nameserver:
    - https://doh.pub/dns-query
    - https://dns.alidns.com/dns-query
    - tls://223.5.5.5
  fallback:
    - https://doh.pub/dns-query
    - https://dns.alidns.com/dns-query
  fallback-filter:
    geoip: true
    geoip-code: CN
```

### 必须加的规则

1. **盒子 SSH 走 DIRECT**：TUN 下盒子域名会解析成 fake-ip，若走代理会形成
   `SSH → 代理 → 隧道 → SSH` 的死循环。

```yaml
prepend:
  - DOMAIN-SUFFIX,<your-box-host>,DIRECT   # 例如 xxxx.instabox.sh
  - IP-CIDR,<ssh-server-ip>/32,DIRECT,no-resolve
```

2. **非国内 UDP 拒绝**：HTTP 代理 + SSH 只能转 TCP，QUIC/UDP 会被直连泄漏真实 IP。

```yaml
# 放在 GEOIP,CN,DIRECT 之后
- NETWORK,UDP,REJECT
```

3. **redir-port 冲突**：有些订阅自带 `redir-port: 7892`，与 mixed-port 相同会导致
   混合端口绑定失败、连上就 reset。把它改成 `0`。

4. **GitHub 等站点**：如果隧道对某些站点太慢，可以让这些域名单独走机场快节点：

```yaml
- DOMAIN-SUFFIX,github.com,<fast-node-name>
```

## 4. 守护与自启

```powershell
# 每分钟检查一次，掉线自动重建（普通用户权限即可）
powershell -ExecutionPolicy Bypass -File .\proxy\install-watchdog.ps1

# 可选：登录时自动拉起
powershell -ExecutionPolicy Bypass -File .\proxy\install-autostart.ps1
```

守护使用 `wscript.exe` 启动隐藏的 PowerShell，不会闪黑框。

## 5. 桌面面板（可选）

```powershell
# 先准备 proxy/config.json
pythonw.exe .\proxy\proxy_gui.py
```

面板显示运行状态、出口 IP、延迟、守护状态，并提供启动/停止/重启/测试/日志按钮。

## 6. 工作原理要点

- `start-us-proxy.ps1` 先做**冷却检测**：盒子在 480 分钟强制暂停的冷却期内，直接输出
  `[WAIT]` 并退出，不占用锁。
- **单实例锁** `start.lock`：手动启动和每分钟守护不会并发重启、互相杀掉 ssh。
- **健康检查**：端口在监听不代表隧道可用；脚本会真正通过 3128 请求一次，失败则重启。
- **隐藏启动**：`forward-hidden.vbs.template` 生成 VBS，用窗口样式 0 启动 cmd，避免黑框。
- WMI 无法可靠启动 `wscript.exe`（GUI 子系统），所以用 `Start-Process wscript.exe`。

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 端口在但所有请求超时 | ssh 转发进程假活 | 守护会重启；或手动 `stop-us-proxy.ps1` 后重跑 |
| `[WAIT] box cooling down` | 免费版 480 分钟上限 | 等冷却结束（约 10 分钟），守护会自动恢复 |
| DNS 全变成 198.18.x，直连超时 | TUN 留下 fake-ip 但没有路由 | 关闭/重启 TUN；或停掉遗留的 mihomo 服务 |
| 测 IP 显示国内 | QUIC/UDP 直连泄漏 | 加 `NETWORK,UDP,REJECT` |
| 连 7892 立即 reset | redir-port 与 mixed-port 相同 | `redir-port: 0` |
| 黑框常驻 | 隧道父进程 cmd 可见 | 用本仓库的 VBS 隐藏启动 |
