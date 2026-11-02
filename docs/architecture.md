# 架构

## 总览

```
┌────────────────────────── Windows ──────────────────────────┐
│                                                             │
│  Clash Verge (TUN)                                          │
│    │  规则：境外 -> Proxy(US-111) / GitHub -> TW1           │
│    ▼                                                        │
│  127.0.0.1:3128  ← ssh.exe -L（SSH 加密隧道）                │
│    ▲                                                        │
│  start-us-proxy.ps1 / watchdog / GUI                        │
│    │  instabox port-forward 111 3128:3128                   │
└────┼────────────────────────────────────────────────────────┘
     │ SSH :2222
     ▼
┌──────────────── InstaBox 盒子 (Debian, 4C/8G, 美国) ─────────┐
│  gost -L http://USER:PASS@:3128     ← 出口代理               │
│  msagent.server :8899               ← 任务队列（可选）        │
│  /data/…                            ← 持久盘                  │
└─────────────────────────────────────────────────────────────┘

┌──────────────── ModelScope DSW 容器 (8C/30G, 免费 CPU) ──────┐
│  chat_server.py :8899                                       │
│    ├─ llama.cpp GGUF 对话                                    │
│    ├─ diffusers / OpenVINO 生图                              │
│    └─ /v1/chat/completions, /v1/embeddings                  │
│  cloudflared tunnel run --token …                           │
│  /mnt/workspace/chat  ← NAS 持久化（代码/配置/模型/wheels）   │
└─────────────────────────────────────────────────────────────┘
     ▲
     │ HTTPS
  chat.example.com  ← Cloudflare 命名隧道 + DNS CNAME
```

## 数据流

### 代理
1. 应用把请求发给 `127.0.0.1:3128`（HTTP 代理协议）。
2. 该端口由 `ssh.exe -L` 打开，流量进入 SSH 加密隧道。
3. 隧道终点是盒子的 `localhost:3128`，由 gost 接收。
4. gost 连接目标站点；HTTPS 内容是端到端 TLS，gost 只看到域名/IP。

### 聊天
1. 浏览器/客户端请求 `https://chat.example.com/...`。
2. Cloudflare 隧道把请求转给容器 `127.0.0.1:8899`。
3. `chat_server.py` 鉴权（Basic 或 Bearer key），串行排队执行推理。
4. 模型从 NAS 加载（GGUF / diffusers 目录），结果流式返回。

## 端口与持久化

| 位置 | 端口 | 用途 |
|---|---|---|
| Windows | 3128 | SSH 转发入口（仅 127.0.0.1） |
| InstaBox | 3128 | gost 代理（仅隧道可达） |
| InstaBox | 8899 | msagent（可通过 port publish 暴露 HTTPS） |
| 容器 | 8899 | chat_server.py |
| 容器 | 53/UDP | mihomo DNS（仅 TUN 模式，Windows 侧） |

| 路径 | 内容 |
|---|---|
| `C:\Users\<you>\us-proxy\` | 代理脚本、日志、配置（config.ps1/json） |
| `/mnt/workspace/chat` | chat_server.py、setup.sh、tunnel.sh、auth、apikey、settings.json、wheels、bin |
| `/mnt/workspace/gguf` | GGUF 模型 |
| `/mnt/workspace/img` | diffusers / OpenVINO 图片模型 |
| `/data/msagent`（盒子） | msagent server.py、token、结果日志 |

## 启动顺序

1. InstaBox 盒子运行 `gost`（systemd 自启）。
2. Windows 登录：启动项/守护任务拉起 `start-us-proxy.ps1` → SSH 隧道。
3. Clash Verge：服务模式 + TUN（或系统代理）把流量导入 3128。
4. 容器恢复后：`setup.sh` 启动聊天服务 → `tunnel.sh` 启动 Cloudflare 隧道。
5. 可选：容器里跑 msagent worker，长轮询盒子上的任务队列。

## 设计取舍

- 用 SSH 隧道而不是在盒子上开公网端口：免费版不开放原始 TCP，且 SSH 自带加密与认证。
- 用 gost 而不是完整 VPN：客户端零安装，Clash/浏览器原生支持 HTTP 代理；代价是没有 UDP。
- 用 NAS 持久化：容器无状态，代码/模型/配置全部放 `/mnt/workspace`。
- 用 Cloudflare 命名隧道：固定域名 + 自动 TLS，不需要公网 IP 或端口映射。
