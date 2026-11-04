# InstaBox + ModelScope 自建 AI 服务栈

用一台免费的 **InstaBox** 小机器做美国出口 + 任务调度，用 **ModelScope（魔搭）免费 CPU 实例**跑本地大模型聊天和生图，通过 **Cloudflare 命名隧道**暴露成固定域名。

> 这个仓库记录的是「方法 + 可复用脚本」，不包含任何账号、密钥、域名或节点信息。所有敏感项都需要你自己填写。

## 两套系统

### 1. InstaBox 美国出口代理

```
Clash / 浏览器
   │ HTTP CONNECT (127.0.0.1:3128)
   ▼
ssh.exe -L 127.0.0.1:3128:localhost:3128   ← SSH 加密隧道
   ▼
InstaBox 盒子 (美国)  gost HTTP 代理 :3128
   ▼
目标网站
```

特点：
- 免费盒子、固定美国 IP、无需在盒子上装 VPN 服务端
- Windows 端有幂等启动、健康检查、单实例锁、每分钟守护、无黑框启动和 tkinter 面板
- 附 Clash Verge TUN 的坑位说明（fake-ip、redir-port、QUIC 泄漏）

适合：需要美国出口可达性的场景、API 调用、备用出口。
不适合：流媒体解锁（机房 IP）、游戏、大流量下载、7×24 关键服务（免费版每 8 小时强制暂停约 10 分钟）。

### 2. ModelScope CPU 聊天/生图站

```
浏览器 / API
   │ HTTPS (chat.example.com)
   ▼
Cloudflare 命名隧道
   ▼
ModelScope DSW 容器 :8899  chat_server.py
   │ llama.cpp GGUF / diffusers / OpenVINO
   ▼
NAS /mnt/workspace （模型、代码、配置、wheel 缓存）
```

能力：
- 多模型 GGUF 对话（流式、硬开关思考、上下文管理、队列）
- 多会话聊天页：每个会话独立上下文，记录保存在服务端 NAS，浏览器本地缓存；支持搜索、时间分组、置顶、重命名、删除和侧栏折叠
- 生成任务与页面解耦：关闭网页后继续生成，完成后写入云端记录；支持停止生成
- Intelligent UI 工具协议：系统提示词定义图表/对比/计算器/SVG/流程图工具及触发条件，模型在回答中主动调用，前端按白名单渲染
- OpenAI 兼容 `/v1/chat/completions`、`/v1/embeddings`
- 异步生图（SDXL-Turbo、FLUX.1-schnell INT4 OpenVINO）
- Basic Auth + API Key；网页管理 key/密码/设置
- 容器被回收后，靠 NAS 上的代码/模型/wheel 快速恢复
- 可选的 msagent：在 InstaBox 上排队下发任务，远程控制容器

## 快速开始

| 目标 | 文档 |
|---|---|
| 搭美国出口代理 | [docs/instabox-proxy.md](docs/instabox-proxy.md) |
| 搭魔搭聊天/生图站 | [docs/modelscope-chat.md](docs/modelscope-chat.md) |
| 理解整体架构 | [docs/architecture.md](docs/architecture.md) |
| 踩过的坑 | [docs/pitfalls.md](docs/pitfalls.md) |
| 安全事项 | [SECURITY.md](SECURITY.md) |

## 目录

```
proxy/                 Windows 端代理脚本（PowerShell + VBS + tkinter GUI）
modelscope/            魔搭容器端：chat_server.py、setup.sh、tunnel.sh、msagent
docs/                  架构、部署、避坑
```

## 重要限制与免责声明

- **免费 InstaBox**：空闲 30 分钟暂停；连续运行 480 分钟强制暂停，冷却约 10 分钟，期间无法唤醒。不要把它当 7×24 生产服务。
- **机房 IP**：InstaBox 的 IP 属于云厂商（如 Latitude.sh），不是住宅 IP；流媒体、部分注册和风控严格的站点会识别。API 类场景通常没问题。
- **ModelScope 免费容器**：会被周期性回收，NAS 之外的数据会丢；长任务需要保持页面/隧道活跃。
- **合规**：请遵守 InstaBox、ModelScope、Cloudflare 的服务条款以及你所在地区的法律法规。用免费额度做代理/隧道可能违反平台条款，风险自负。
- **安全**：`/api/exec` 是远程 shell；一旦泄露等同于服务器控制权。务必使用强密码、API Key，并只通过 HTTPS 暴露。

## License

MIT
