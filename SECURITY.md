# 安全说明

## 不要提交的秘密

以下文件已被 `.gitignore` 排除，**永远不要**提交：

- `proxy/config.ps1`、`proxy/config.json`（代理用户名/密码、盒子名）
- `modelscope/config/auth`（网页 Basic Auth）
- `modelscope/config/apikey`（OpenAI 兼容 API Key）
- `tunnel-token`（Cloudflare 隧道 token，等同于隧道控制权）
- `msagent/token`（任务队列 token，等同于远程命令执行权）
- 任何 `.env`、日志、模型、wheel 缓存

如果你 fork 或复制本仓库，提交前运行：

```bash
grep -rnE 'sk-[a-f0-9]{20,}|token|password|secret' . --exclude-dir=.git
```

## /api/exec 的风险

`chat_server.py` 提供了一个 `POST /api/exec` 端点，可以在容器里执行 shell 命令。
它只应由你自己使用，并且必须满足：

- 只通过 HTTPS（Cloudflare 隧道）访问
- 有强 Basic Auth 密码或 API Key
- 不需要时直接删除该端点或加开关

一旦泄露，攻击者可以在你的容器/NAS 上执行任意命令。

## 代理与隧道

- **不要**把 gost 的 3128 直接发布到公网；本方案只通过 SSH 隧道访问。
- Cloudflare 隧道 token 泄露 = 别人可以把流量转到你的容器；泄露后立即在 Cloudflare 删除隧道并重建。
- `msagent` 的 token 泄露 = 别人可以下发任意命令；定期轮换，必要时停掉服务。

## 平台条款

- InstaBox 免费额度用于代理/隧道可能违反其服务条款，可能导致封号。
- ModelScope 免费实例有资源和使用限制；不要跑挖矿、扫描、攻击等滥用行为。
- 请遵守你所在地区的法律法规。

## 报告问题

如果发现本仓库中的脚本存在安全问题，请开 issue（不要附带真实密钥）。
