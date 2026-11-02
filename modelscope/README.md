# modelscope

魔搭 DSW CPU 容器端：聊天/生图服务、Cloudflare 隧道、可选的 msagent。

```bash
mkdir -p /mnt/workspace/chat
# copy chat_server.py setup.sh tunnel.sh requirements.txt there
echo 'USER:PASS' > /mnt/workspace/chat/auth
bash /mnt/workspace/chat/setup.sh
bash /mnt/workspace/chat/tunnel.sh
```

完整说明见 [../docs/modelscope-chat.md](../docs/modelscope-chat.md)。
