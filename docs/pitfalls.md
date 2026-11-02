# 踩过的坑

按系统分类，每条都带现象和修复。

## InstaBox / 代理

### 1. SSH 隧道「假活」
**现象**：3128 在监听，但所有请求超时；`Get-NetTCPConnection` 显示大量 `CloseWait`。
**原因**：ssh 转发进程卡死，端口还在。
**修复**：健康检查必须真正发一次请求（`curl -x 3128 ...`），失败就杀掉监听进程重建；加单实例锁，避免手动和守护互相杀 ssh。

### 2. TUN 的 fake-ip 没有路由
**现象**：所有 DNS 返回 `198.18.x.x`，ssh/命令行直连超时，浏览器却正常。
**原因**：代理客户端留下 Mihomo TUN 适配器，DNS 被劫持，但路由表只有 `198.18.0.0/30`，没有 `/16`。
**修复**：停掉遗留的代理服务/禁用 TUN 适配器，清 DNS 缓存；或正确启用 TUN 的 auto-route。

### 3. QUIC/UDP 绕过代理
**现象**：curl 测出口是美国，浏览器测 IP 却是本地真实 IP。
**原因**：浏览器走 HTTP/3（QUIC/UDP 443），而 SSH/HTTP 代理只支持 TCP，mihomo 把这条 UDP 直连了。
**修复**：在 `GEOIP,CN,DIRECT` 之后加 `NETWORK,UDP,REJECT`，让浏览器回退 TCP；国内 UDP 仍直连。

### 4. redir-port 与 mixed-port 冲突
**现象**：7892 端口一连接就 reset；核心日志 `Start Mixed server error: bind: Only one usage...`。
**原因**：订阅自带 `redir-port: 7892`，与混合端口相同；Windows 不支持 redir，但它仍会占用端口。
**修复**：把 `redir-port` 改成 `0`。

### 5. PowerShell 5.1 读 UTF-8 无 BOM 的 .ps1
**现象**：带中文注释的脚本报「意外的标记 `}`」。
**修复**：脚本用纯 ASCII，或在保存时加 UTF-8 BOM。本仓库选择纯 ASCII。

### 6. WMI 启动 wscript 静默失败
**现象**：`Win32_Process.Create` 返回 0，但 VBS 完全没执行。
**原因**：WMI 对 GUI 子系统进程的支持不可靠。
**修复**：用 `Start-Process wscript.exe -WindowStyle Hidden` 从用户会话启动。

### 7. 黑框常驻
**现象**：一个空的 `cmd.exe` 窗口一直在。
**原因**：`cmd /c instabox port-forward` 是隧道的父进程，WMI 创建在用户会话可见。
**修复**：VBS 用窗口样式 0 启动 cmd；VBS 自己由 wscript 启动（无控制台）。

### 8. 免费盒子 480 分钟强制暂停
**现象**：启动脚本连续失败，`instabox resume` 返回 `machine_cooldown`。
**修复**：脚本先读 `cooldownUntil`，冷却中直接输出 `[WAIT]` 并退出；守护每分钟重试，冷却结束自动恢复。免费版这个限制改不了。

## 魔搭 / Cloudflare

### 9. UI 创建的隧道 token 无效
**现象**：cloudflared 日志 `Invalid tunnel secret`；`POST /token` 返回 405。
**修复**：用 API 创建 `cfd_tunnel`（`config_src: cloudflare`），响应里直接带有效 token。

### 10. cloudflared 下载截断导致段错误
**现象**：`cloudflared --version` 直接 Segmentation fault。
**原因**：`curl --max-time` 截断了 40MB 二进制。
**修复**：校验文件大小（`cloudflared-linux-amd64` 约 40274445 字节）或 sha256；超时给足、失败重试；走 gh-proxy 镜像。

### 11. pkill 误杀自己
**现象**：部署脚本执行到一半，外层 shell 被杀。
**原因**：`pkill -f chat_server.py` 会匹配到任务脚本自身的命令行（heredoc 文本也在 cmdline 里）。
**修复**：不要用 pkill；用 `fuser -k 8899/tcp` 或按 PID 杀；必须用时用 `chat_server[.]py` 且避免脚本文本包含原串。

### 12. msagent 单任务命令 64KB 上限
**现象**：大 base64 部署被截断，sha256 不符。
**修复**：分片 `echo -n chunk >> /tmp/x.b64`，每片 ≤22KB，最后 decode + 校验。

### 13. NAS 配额 EDQUOT 延迟
**现象**：删除大文件后仍写不进，连小文件都失败。
**原因**：配额统计/快照有延迟。
**修复**：等一段时间；用 `du -sh /mnt/workspace/*` 找大目录；临时把服务放 `/tmp` 跑。

### 14. Cloudflare 100 秒请求上限
**现象**：同步生图返回 524。
**修复**：改成异步任务：POST 立即返回 job id，GET 轮询进度/结果。

### 15. 容器被回收
**现象**：服务突然 502/530，依赖和进程都没了。
**修复**：所有代码/模型/配置/wheel 放 `/mnt/workspace`；恢复只跑 `setup.sh` + `tunnel.sh`。

### 16. 子进程中文输出乱码
**现象**：GUI 日志里 `schtasks` 输出是 `�ɹ�...`。
**原因**：中文 Windows 的 `schtasks` 输出 GBK，按 UTF-8 解码失败。
**修复**：捕获 bytes，先试 UTF-8 再试 GBK。
