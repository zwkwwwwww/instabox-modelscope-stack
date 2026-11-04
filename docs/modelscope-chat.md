# 魔搭（ModelScope）CPU 聊天/生图站

目标：在 ModelScope 免费 DSW CPU 实例上跑一个可切换模型、支持流式对话和生图的服务，用 Cloudflare 命名隧道暴露成固定 HTTPS 域名。

## 前置条件

- ModelScope 账号，创建一个 **DSW CPU 实例**（免费 8C/30G，Ubuntu 22.04）
- 实例里的 `/mnt/workspace` 是**持久 NAS**：容器被回收后数据还在
- 一个域名，NS 托管在 Cloudflare（用于命名隧道）

## 1. 目录规划

```bash
mkdir -p /mnt/workspace/chat/{bin,wheels,static}
mkdir -p /mnt/workspace/gguf
mkdir -p /mnt/workspace/img
```

把本仓库的 `modelscope/chat_server.py`、`setup.sh`、`tunnel.sh`、`requirements.txt`
复制到 `/mnt/workspace/chat/`，并 `chmod +x setup.sh tunnel.sh`。

## 2. 下载模型

### GGUF（llama.cpp）

从 ModelScope 或 HF 镜像下载到 `/mnt/workspace/gguf/<name>/`，例如：

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-30B-A3B-GGUF \
  --include "*Q4_K_M.gguf" --local_dir /mnt/workspace/gguf/30b
```

`chat_server.py` 里的 `MODELS` 字典决定了可选模型。按你的实际路径修改：

```python
MODELS = {
    "qwen3-30b-a3b": {"path": "/mnt/workspace/gguf/30b/....gguf", ...},
}
```

> 经验：4C/8G 上 30B-A3B（MoE，激活 3B）约 7 tok/s，27B dense 约 3 tok/s，
> 4B 约 15-20 tok/s，1.7B 可到 30 tok/s。写代码建议 4B + 4096 输出。

### Diffusers（生图）

```bash
modelscope download --model AI-ModelScope/sdxl-turbo --local_dir /mnt/workspace/img/sdxl-turbo
```

FLUX.1-schnell INT4 OpenVINO 需要额外的 OpenVINO 模型目录；可以自己转换或用公开的
OpenVINO 仓库，放到 `/mnt/workspace/img/flux1-schnell-int4-ov`。

## 3. 配置与启动

```bash
cd /mnt/workspace/chat

# Basic Auth（用户名:密码）
echo 'YOUR_USER:YOUR_PASSWORD' > auth
chmod 600 auth

# 可选：设置默认参数
cat > settings.json <<'JSON'
{"think_level":"high","max_queue":8,"max_tokens_default":4096,"ctx_size":16384}
JSON

bash setup.sh
curl -u YOUR_USER:YOUR_PASSWORD http://127.0.0.1:8899/api/status
```

`setup.sh` 会优先从 `wheels/` 离线安装依赖，然后重启 8899 端口上的服务。
**不要用 `pkill -f chat_server.py`**：任务脚本自身命令行也含这个字符串，会误杀外层 shell。

## 4. Cloudflare 命名隧道

### 4.1 用 API 创建隧道（UI 创建的 token 可能无效）

```bash
ACCOUNT_ID=your_account_id
TUNNEL=$(curl -s -X POST \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/cfd_tunnel" \
  -d '{"name":"muse-chat","config_src":"cloudflare"}')
# 从响应里取 id 和 token
```

把 token 保存到 `/mnt/workspace/chat/tunnel-token`（`chmod 600`）。

### 4.2 配置 ingress

```bash
curl -s -X PUT \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID/configurations" \
  -d '{"config":{"ingress":[{"hostname":"chat.example.com","service":"http://localhost:8899"},{"service":"http_status:404"}]}}'
```

### 4.3 DNS

```bash
curl -s -X POST \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
  -d "{\"type\":\"CNAME\",\"name\":\"chat\",\"content\":\"$TUNNEL_ID.cfargotunnel.com\",\"proxied\":true}"
```

### 4.4 启动隧道

```bash
bash tunnel.sh
```

`tunnel.sh` 会优先使用 NAS 上缓存的 `bin/cloudflared`，下载时走 gh-proxy 镜像并**校验文件大小**（截断的二进制会段错误）。

## 5. 使用

### Web UI（浏览器）

打开 `https://chat.example.com`，通过 Basic Auth 登录后是内置的聊天页：

- 左侧会话列表支持新建、切换、删除、搜索、置顶、双击重命名；按「置顶 / 今天 / 昨天 / 最近 7 天 / 更早」分组
- **每个会话独立上下文**，保存在浏览器 `localStorage`（每个会话最多 80 条）；`Ctrl/⌘+K` 聚焦搜索，左上角按钮可收起侧栏
- 旧版单历史 `muse_hist` 会自动迁移成一个会话
- 思考开关按模型能力显示；MiniCPM 的「关」通过原生聊天模板硬关闭，不是提示词软引导
- 思考内容生成时展开，回复完成后自动折叠；点标题可展开/收起
- 生成过程中可以切到别的会话，回复仍写回发起请求的原会话；切回即可看到
- 顶栏可加载模型、设置单次最大输出；API Key、访问密码和服务参数可在页面里修改

会话只存在当前浏览器，不占 NAS；换设备或清缓存后不会同步。

### 对话（OpenAI 兼容）

```bash
curl -s https://chat.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

### 向量

```bash
curl -s https://chat.example.com/v1/embeddings \
  -H "Authorization: Bearer sk-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-m3","input":["hello"]}'
```

### 生图（异步，避免 Cloudflare 100 秒超时）

```bash
# 提交
curl -s -X POST https://chat.example.com/api/image \
  -H "Authorization: Bearer sk-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"sdxl-turbo","prompt":"a cat","steps":4,"width":512,"height":512}'
# 轮询 GET /api/image/result?id=...
```

## 6. 容器回收后的恢复

ModelScope 免费容器会被周期性回收。恢复只需两条命令：

```bash
bash /mnt/workspace/chat/setup.sh
bash /mnt/workspace/chat/tunnel.sh
```

模型、代码、配置、wheel 缓存、cloudflared 都在 NAS 上，不需要重新下载。

## 7. 可选：msagent 远程任务队列

在 InstaBox 盒子上跑 `msagent/server.py`，让容器里的 worker 长轮询拉任务：

```bash
# 盒子
mkdir -p /data/msagent
cp modelscope/msagent/server.py /data/msagent/
cp modelscope/msagent/msagent.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now msagent

# 把 8899 发布成公网 HTTPS
instabox port publish <box> 8899
```

容器里启动 worker（服务端 `/agent` 会注入 BASE/TOKEN）：

```bash
curl -fsSL 'https://<published-host>/agent?token=<TOKEN>' | nohup python3 - >/tmp/msagent.log 2>&1 &
```

注意：网关会掐断空闲连接，长轮询等待要控制在 **8 秒**以内，否则任务会丢。

## 8. 性能与限制

- 免费 CPU 没有 GPU，30B 级模型只能跑量化 GGUF，速度个位数 tok/s
- 生图 SDXL-Turbo 512px 约 1-2 分钟，FLUX INT4 约 3-4 分钟
- 上下文越长，prefill 越慢（16K 可能要几分钟）
- 容器无状态，只有 `/mnt/workspace` 持久
