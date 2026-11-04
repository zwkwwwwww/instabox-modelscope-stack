#!/usr/bin/env python3
# ModelScope chat: model-switchable chat UI over llama.cpp GGUF models on NAS.
# Runs in the ModelScope container on :8888. One model resident at a time (RAM-safe).
import base64, json, os, re, secrets, sys, time, gc, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

MODELS = {
    "qwen3-30b-a3b": {"path": "/mnt/workspace/gguf/30b/Qwen3-30B-A3B-Q4_K_M.gguf",
                      "label": "Qwen3-30B-A3B · 约7 tok/s"},
    "qwen3.8-27b":   {"path": "/mnt/workspace/gguf/27b/Qwen3.8-27B-Q4_K_M.gguf",
                      "label": "Qwen3.8-27B · 约3.3 tok/s · 最强"},
    "qwen3-4b":      {"path": "/mnt/workspace/gguf/4b/Qwen3-4B-Q4_K_M.gguf",
                      "label": "Qwen3-4B · 约15-20 tok/s · 快"},
    "qwen3-1.7b":    {"path": "/mnt/workspace/gguf/17b/Qwen3-1.7B-Q4_K_M.gguf",
                      "label": "Qwen3-1.7B · 约30 tok/s · 秒回"},
    "minicpm4.1-8b": {"path": "/mnt/workspace/gguf/minicpm41/MiniCPM4.1-8B-Q4_K_M.gguf",
                      "label": "MiniCPM4.1-8B · 约6-9 tok/s · 高效"},
}
IMG_MODELS = {
    "sdxl-turbo": {"path": "/mnt/workspace/img/sdxl-turbo", "label": "SDXL-Turbo · 4步 · 高质量", "steps": 4, "guidance": 0.0},
    "flux-schnell-ov": {"path": "/mnt/workspace/img/flux1-schnell-int4-ov", "label": "FLUX.1-schnell INT4 · 4步 · 最强", "steps": 4, "guidance": 0.0, "engine": "openvino"},
}
img_state = {"name": None, "pipe": None, "loading": None}
IMG_JOBS = {}
IMG_STATS = {}

AUTH_FILE = "/mnt/workspace/chat/auth"
if not os.path.exists(AUTH_FILE):
    with open(AUTH_FILE, "w") as f:
        f.write("chat:" + secrets.token_urlsafe(6)[:8])
AUTH_LINE = open(AUTH_FILE).read().strip()
AUTH_B64 = "Basic " + base64.b64encode(AUTH_LINE.encode()).decode()

APIKEY_FILE = "/mnt/workspace/chat/apikey"
if not os.path.exists(APIKEY_FILE):
    with open(APIKEY_FILE, "w") as f:
        f.write("sk-" + secrets.token_hex(24))
API_KEY = open(APIKEY_FILE).read().strip()

SETTINGS_FILE = "/mnt/workspace/chat/settings.json"
DEFAULT_SETTINGS = {"think_level": "high", "max_queue": 8, "max_tokens_default": 4096, "ctx_size": 16384}
try:
    settings = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_FILE):
        settings.update(json.load(open(SETTINGS_FILE)))
except Exception:
    settings = dict(DEFAULT_SETTINGS)


def save_settings():
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, ensure_ascii=False)
    except Exception:
        pass


def think_directive(level):
    if level == "off":
        return {"role": "system", "content": "/no_think"}
    if level == "low":
        return {"role": "system", "content": "/think" + chr(10) + "先直接给出答案；只做一两句极简推理，不要展开步骤。"}
    if level == "medium":
        return {"role": "system", "content": "/think" + chr(10) + "思考保持适中，抓住关键点即可，不要过度展开。"}
    return {"role": "system", "content": "/think"}

state = {"name": None, "llm": None, "loading": None,
         "load_started": None, "last_load_seconds": None,
         "load_progress": None, "load_rss_mb": None}
load_lock = threading.Lock()
gen_lock = threading.Lock()
q_lock = threading.Lock()
q_state = {"waiting": 0}
MAX_QUEUE = 8


def load_model(name):
    if name not in MODELS:
        return
    with load_lock:
        if state["llm"] is not None and state["name"] == name:
            return
        state["loading"] = name
        state["load_started"] = time.time()
        state["load_progress"] = 0
        state["load_rss_mb"] = 0
        stop_ev = threading.Event()

        def _rss_kb():
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1])
            except Exception:
                pass
            return 0

        try:
            if img_state["pipe"] is not None:
                img_state["pipe"] = None
                img_state["name"] = None
            if state["llm"] is not None:
                state["llm"] = None
                state["name"] = None
                gc.collect()
            base_kb = _rss_kb()

            def _sampler(p, ev, base):
                try:
                    total = max(1, os.path.getsize(p))
                except Exception:
                    total = 1
                while not ev.is_set():
                    kb = _rss_kb()
                    used_mb = max(0, kb - base) // 1024
                    state["load_rss_mb"] = used_mb
                    pct = int(max(0, kb - base) * 1024 / total * 100)
                    prev = state.get("load_progress") or 0
                    state["load_progress"] = min(99, max(prev, pct))
                    ev.wait(1)

            threading.Thread(target=_sampler, args=(MODELS[name]["path"], stop_ev, base_kb), daemon=True).start()
            from llama_cpp import Llama
            t0 = time.time()
            nctx = max(2048, min(65536, int(settings.get("ctx_size") or 16384)))
            llm = Llama(model_path=MODELS[name]["path"], n_ctx=nctx,
                        n_threads=8, verbose=False)
            state["ctx"] = nctx
            state["llm"] = llm
            state["name"] = name
            state["last_load_seconds"] = round(time.time() - t0)
            state["load_progress"] = 100
            print("loaded", name, "in", state["last_load_seconds"], "s", flush=True)
        except Exception as e:
            print("load failed:", repr(e), flush=True)
        finally:
            stop_ev.set()
            state["loading"] = None
            state["load_started"] = None


def load_img_model(name):
    with load_lock:
        if img_state["pipe"] is not None and img_state["name"] == name:
            return img_state["pipe"]
        if state["llm"] is not None:
            state["llm"] = None
            state["name"] = None
        if img_state["pipe"] is not None:
            img_state["pipe"] = None
        img_state["name"] = None
        gc.collect()
        cfg = IMG_MODELS[name]
        if cfg.get("engine") == "openvino":
            from optimum.intel import OVDiffusionPipeline
            pipe = OVDiffusionPipeline.from_pretrained(cfg["path"])
        else:
            import torch
            from diffusers import AutoPipelineForText2Image
            pipe = AutoPipelineForText2Image.from_pretrained(
                cfg["path"], torch_dtype=torch.float32, local_files_only=True)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        pipe.to("cpu")
        img_state["pipe"] = pipe
        img_state["name"] = name
        print("img model loaded:", name, flush=True)
        return pipe


IMG_PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>生图 · 魔搭</title>
<style>
:root{color-scheme:dark}
body{margin:0;font:15px/1.6 system-ui,-apple-system,'Segoe UI',sans-serif;background:#0f1115;color:#e6e6e6;padding:14px}
#bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
select,input,button{background:#171a21;color:#e6e6e6;border:1px solid #333a48;border-radius:8px;padding:8px 10px;font-size:14px}
input{min-width:220px;flex:1}
button{cursor:pointer}
#go{background:#2f6feb;border-color:#2f6feb;color:#fff;padding:8px 24px}
a{color:#7aa2ff;text-decoration:none;font-size:14px}
#bar2{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
#warm{font-size:12px;color:#7aa2ff}
#pWrap{width:320px;height:10px;background:#171a21;border:1px solid #333a48;border-radius:6px;overflow:hidden;display:none}
#pBar{height:100%;width:0%;background:linear-gradient(90deg,#2f6feb,#7aa2ff);transition:width .6s linear}
#pText{font-size:12px;color:#9aa4b2}
.indet{background:linear-gradient(90deg,#2f6feb,#7aa2ff,#2f6feb);background-size:200% 100%;animation:slidebg 1.2s linear infinite}
@keyframes slidebg{0%{background-position:0% 0}100%{background-position:-200% 0}}
#status{font-size:13px;color:#9aa4b2;margin:8px 0}
#out img{max-width:100%;border-radius:10px;border:1px solid #2a2f3a;margin-top:10px}
details{font-size:13px;color:#9aa4b2;margin-top:18px;max-width:900px}
summary{cursor:pointer;color:#7aa2ff}
pre{background:#0f1115;border:1px solid #2a2f3a;border-radius:8px;padding:10px;overflow-x:auto;font-size:12px}
</style></head><body>
<div id="bar">
 <select id="model" title="模型"></select>
 <input id="prompt" placeholder="提示词（中文也可，英文更敏感）" />
 <input id="neg" placeholder="负面提示（可选）" />
 <select id="size"><option value="512">512x512</option><option value="640">640x640</option><option value="768">768x768</option><option value="1024">1024x1024</option></select>
 <select id="steps"><option value="1">1步</option><option value="2">2步</option><option value="4" selected>4步</option><option value="8">8步</option><option value="20">20步</option><option value="30">30步</option></select>
 <button id="warmBtn" title="手动预热当前选中的模型">🔥 预热</button>
 <button id="go">生成</button>
 <a href="/">← 返回聊天</a>
</div>
<div id="bar2">
 <span id="warm">预热：—</span>
 <div id="pWrap"><div id="pBar"></div></div>
 <span id="pText"></span>
</div>
<div id="status">选择参数后点击生成。页面打开时会自动预热当前模型；生图会卸载聊天模型，切回聊天会自动重载。</div>
<div id="out"><img id="img" style="display:none" /></div>
<details><summary>API 调用（支持 API Key，适合脚本）</summary>
<pre># 1) 提交任务（API Key 在网页聊天页 🔑 处获取）
curl -X POST https://YOUR_DOMAIN/api/image -H "Authorization: Bearer sk-你的KEY" -H "Content-Type: application/json" -d '{"model":"flux-schnell-ov","prompt":"a cat","steps":4,"width":512,"height":512}'
# 返回: {"id":"xxxxxxxx","status":"running"}
# 2) 轮询结果（running 时返回进度，完成时直接返回图片）
curl -H "Authorization: Bearer sk-你的KEY" "https://YOUR_DOMAIN/api/image/result?id=xxxxxxxx" -o out.png
# 模型列表: GET /api/imgmodels    预热: POST /api/image/preload {"model":"flux-schnell-ov"} 或 GET /api/image/preload?model=flux-schnell-ov
# 预热状态: GET /api/image/warm   也可用网页账号密码做 Basic 认证</pre>
</details>
<script>
const $=id=>document.getElementById(id);
let models=[],warmTimer=null;
function hideProg(delay){
 if(delay){setTimeout(function(){$('pWrap').style.display='none';$('pText').textContent='';},delay);}
 else{$('pWrap').style.display='none';$('pText').textContent='';}
}
fetch('api/imgmodels').then(r=>r.json()).then(j=>{
 models=j.models||[];
 const sel=$('model');
 for(const m of models){const o=document.createElement('option');o.value=m.id;o.textContent=m.label;sel.appendChild(o);}
 if(models[0]){$('steps').value=String(models[0].steps);}
 sel.onchange=function(){
  const mm=models.find(x=>x.id===sel.value);
  if(mm){$('steps').value=String(mm.steps);}
  if(warmTimer){clearInterval(warmTimer);warmTimer=null;}
  hideProg();$('img').style.display='none';
  preload();
 };
 preload();
}).catch(function(){});
function preload(){
 const mm=$('model').value; if(!mm)return;
 fetch('api/image/preload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:mm})}).catch(function(){});
 startWarmPoll();
}
function startWarmPoll(){
 if(warmTimer)clearInterval(warmTimer);
 warmTimer=setInterval(async function(){
  try{
   const j=await (await fetch('api/image/warm')).json();
   if(j.loading){
    $('warm').textContent='🔥 预热中：'+j.loading+'…';
    if($('go').disabled===false){
     $('pWrap').style.display='block';
     $('pBar').classList.add('indet');
     $('pBar').style.width='100%';
     $('pText').textContent='预热 '+j.loading;
    }
   }else{
    $('pBar').classList.remove('indet');
    $('warm').textContent=j.loaded?('✅ 已预热：'+j.loaded):'预热：—';
    if((($('pText').textContent)||'').indexOf('预热')===0){hideProg();}
    clearInterval(warmTimer);warmTimer=null;
   }
  }catch(e){clearInterval(warmTimer);warmTimer=null;}
 },3000);
}
$('warmBtn').onclick=function(){preload();};
$('go').onclick=async()=>{
 const pp=$('prompt').value.trim(); if(!pp){$('status').textContent='请输入提示词';return;}
 if(warmTimer){clearInterval(warmTimer);warmTimer=null;}
 $('go').disabled=true;
 $('pBar').classList.remove('indet');
 $('pWrap').style.display='block'; $('pBar').style.width='2%'; $('pText').textContent=''; $('status').textContent='提交中…';
 const t0=Date.now();
 try{
  const body={model:$('model').value,prompt:pp,negative_prompt:$('neg').value.trim(),steps:parseInt($('steps').value,10),width:parseInt($('size').value,10),height:parseInt($('size').value,10)};
  const r=await fetch('api/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const jj=await r.json();
  if(!r.ok||!jj.id){$('status').textContent='失败：'+JSON.stringify(jj).slice(0,200);$('go').disabled=false;hideProg();return;}
  let blob=null;
  while(true){
   await new Promise(function(res){setTimeout(res,2000);});
   const rr=await fetch('api/image/result?id='+jj.id);
   const ct=rr.headers.get('content-type')||'';
   if(rr.ok&&ct.indexOf('image')>=0){blob=await rr.blob();break;}
   if(rr.status>=500){let e='';try{e=JSON.stringify(await rr.json()).slice(0,200);}catch(x){}$('status').textContent='失败：'+e;hideProg();break;}
   let st={};try{st=await rr.json();}catch(x){}
   const pct=st.pct||0;
   $('pBar').style.width=pct+'%';
   const ph=(st.phase==='loading')?' · 加载模型中':'';
   $('pText').textContent=pct+'% · 已等待 '+Math.round((Date.now()-t0)/1000)+' 秒'+ph;
   $('status').textContent='生成中…（FLUX 约 3-4 分钟 / SDXL 约 1.5 分钟）';
  }
  if(blob){
   $('pBar').style.width='100%';
   $('pText').textContent='100%';
   const url=URL.createObjectURL(blob);
   const img=$('img'); img.src=url; img.style.display='block';
   const sec=((Date.now()-t0)/1000).toFixed(1);
   $('status').textContent='✅ 完成，用时 '+sec+' 秒（'+body.width+'x'+body.height+'，'+body.steps+' 步）';
   hideProg(3000);
  }
 }catch(e){$('status').textContent='出错：'+e.message;hideProg();}
 finally{$('go').disabled=false;}
};
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();$('go').click();}});
</script></body></html>"""


IMG_WANT = {"want": None, "started": False}


def _warm_worker():
    while True:
        time.sleep(1)
        want = IMG_WANT["want"]
        if not want:
            continue
        if img_state["pipe"] is not None and img_state["name"] == want:
            IMG_WANT["want"] = None
            continue
        IMG_WANT["want"] = None
        try:
            img_state["loading"] = want
            try:
                load_img_model(want)
            finally:
                img_state["loading"] = None
        except Exception as e:
            print("preload failed:", repr(e), flush=True)


EMB = {"tok": None, "model": None}


def get_embedder():
    with load_lock:
        if EMB["model"] is not None:
            return EMB["tok"], EMB["model"]
        import torch
        from transformers import AutoTokenizer, AutoModel
        path = "/mnt/workspace/gguf/bge-m3"
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModel.from_pretrained(path, local_files_only=True)
        model.eval()
        EMB["tok"] = tok
        EMB["model"] = model
        print("embedder loaded: bge-m3", flush=True)
        return tok, model


HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>魔搭 Chat</title>
<link rel="stylesheet" href="static/katex.min.css">
<style>
:root{
 color-scheme:dark;
 --bg:#0b0c0e;--panel:#0f1012;--card:#141518;--card2:#1a1c1f;
 --border:#232529;--border2:#2d3035;--text:#e8e8e6;--muted:#8b8f97;--dim:#666b73;
 --accent:#9ece6a;--accent-soft:rgba(158,206,106,.12);--blue:#7aa2ff;--warn:#e0a43a;--danger:#f7768e;
 --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace;
 --sans:-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei UI','PingFang SC',sans-serif;
}
*{box-sizing:border-box}
html,body{height:100%;height:100dvh}
body{margin:0;background:var(--bg);color:var(--text);font:14.5px/1.6 var(--sans);overflow:hidden;-webkit-font-smoothing:antialiased}
button,input,select,textarea{font:inherit}
#app{display:flex;height:100vh;height:100dvh;overflow:hidden}

/* ---------- sidebar ---------- */
#sidebar{width:252px;flex:0 0 252px;background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:16px 12px;gap:14px;min-height:0}
.brand{display:flex;align-items:center;gap:10px;padding:2px 6px 4px}
.logo{width:26px;height:26px;border-radius:8px;background:var(--text);color:#0b0c0e;display:flex;align-items:center;justify-content:center;font:700 13px/1 var(--mono);letter-spacing:-.04em}
.brand-name{font-weight:650;font-size:14px;letter-spacing:.01em}
.brand-sub{font-size:11px;color:var(--dim);margin-top:1px}
#clearBtn{display:flex;align-items:center;gap:8px;width:100%;border:0;border-radius:9px;background:var(--text);color:#0b0c0e;padding:9px 11px;font-size:13px;font-weight:600;cursor:pointer;transition:background .15s}
#clearBtn:hover{background:#fff}
#clearBtn svg{width:15px;height:15px}
.side-label{font:600 10px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--dim);padding:0 8px;margin:2px 0 6px}
.threads{display:flex;flex-direction:column;gap:4px;min-height:0;overflow:auto}
.thread{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:8px;background:var(--card);border:1px solid var(--border);color:var(--text);font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.thread .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);flex:0 0 auto}
.side-note{font-size:11px;color:var(--dim);padding:0 8px;line-height:1.5}
.side-nav{margin-top:auto;display:flex;flex-direction:column;gap:2px;padding-top:8px;border-top:1px solid var(--border)}
.side-nav a,.side-nav button{display:flex;align-items:center;gap:9px;width:100%;border:0;background:transparent;color:var(--muted);padding:8px 9px;border-radius:7px;font-size:12.5px;text-decoration:none;cursor:pointer;text-align:left}
.side-nav a:hover,.side-nav button:hover{background:var(--card);color:var(--text)}
.side-nav svg{width:15px;height:15px;opacity:.8;flex:0 0 auto}
.side-foot{display:flex;align-items:center;gap:7px;padding:9px 9px 2px;color:var(--dim);font-size:11px;border-top:1px solid var(--border)}
.side-foot .pulse{width:6px;height:6px;border-radius:50%;background:var(--accent)}

/* ---------- main ---------- */
#main{flex:1;min-width:0;display:flex;flex-direction:column;background:var(--bg);overflow:hidden}
#bar{display:flex;align-items:center;gap:8px;padding:10px 22px;border-bottom:1px solid var(--border);background:rgba(15,16,18,.82);backdrop-filter:blur(10px);flex-wrap:wrap;min-height:54px}
#menuBtn{display:none;align-items:center;justify-content:center;width:32px;height:32px;border:1px solid var(--border);background:var(--card);color:var(--muted);border-radius:8px;cursor:pointer}
.ctl{appearance:none;background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12.5px;outline:none;cursor:pointer;transition:border-color .15s,background .15s}
.ctl:hover{border-color:var(--border2);background:var(--card2)}
.ctl:focus{border-color:#4a4f58}
#loadBtn{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:6px 11px;font-size:12.5px;cursor:pointer}
#loadBtn:hover:not(:disabled){background:var(--card2);border-color:var(--border2)}
#loadBtn:disabled{opacity:.45;cursor:default}
#model{max-width:230px}
#ctxInfo{font-size:11.5px;color:var(--dim);white-space:nowrap;margin-left:2px}
#status{margin-left:auto;display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--muted);max-width:42%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#status::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);flex:0 0 auto}
#progWrap{display:none;align-items:center;gap:10px;padding:7px 22px;border-bottom:1px solid var(--border);background:var(--panel)}
#progBar{height:3px;flex:0 0 180px;background:var(--accent);border-radius:2px;width:0;transition:width .8s linear}
#progText{font:11px/1.4 var(--mono);color:var(--muted);white-space:nowrap}

/* ---------- messages ---------- */
#log{flex:1;overflow-y:auto;padding:26px max(22px,calc((100% - 940px)/2)) 40px;display:flex;flex-direction:column;gap:0;scroll-behavior:smooth}
#log::-webkit-scrollbar,.threads::-webkit-scrollbar{width:8px}
#log::-webkit-scrollbar-thumb,.threads::-webkit-scrollbar-thumb{background:#2a2d32;border-radius:8px}
#log::-webkit-scrollbar-track,.threads::-webkit-scrollbar-track{background:transparent}
.msg{position:relative;max-width:none;padding:18px 0;border-bottom:1px solid rgba(35,37,41,.65);white-space:pre-wrap;word-break:break-word}
.msg::before{display:block;font:600 10px/1 var(--mono);letter-spacing:.13em;text-transform:uppercase;color:var(--dim);margin-bottom:9px}
.msg.me{max-width:86%;align-self:flex-end;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:13px 15px;margin:6px 0 18px;white-space:pre-wrap}
.msg.me::before{content:'YOU';color:var(--muted)}
.msg.ai{padding:8px 0 22px;border-bottom:1px solid rgba(35,37,41,.65)}
.msg.ai::before{content:'ASSISTANT'}
.msg.sys{align-self:center;max-width:92%;border:1px solid var(--border);background:var(--panel);border-radius:999px;padding:5px 12px;margin:8px 0;color:var(--dim);font-size:11.5px;text-align:center}
.msg.sys::before{display:none}
.think{color:var(--dim);font:12.5px/1.65 var(--mono);border-left:2px solid var(--border2);padding:8px 0 8px 12px;margin:0 0 12px;white-space:pre-wrap;max-height:260px;overflow:auto}
.md{line-height:1.75;font-size:14.5px}
.md h1,.md h2,.md h3,.md h4{margin:18px 0 8px;line-height:1.3;font-weight:650;letter-spacing:-.01em}
.md h1{font-size:1.42em}.md h2{font-size:1.24em}.md h3{font-size:1.1em}
.md p{margin:9px 0}
.md ul,.md ol{margin:9px 0;padding-left:24px}
.md li{margin:4px 0}
.md code{background:#0d0e10;border:1px solid var(--border);border-radius:5px;padding:1px 5px;font-family:var(--mono);font-size:.9em;color:#d7e0c8}
.md pre{background:#0d0e10;border:1px solid var(--border);border-radius:10px;padding:13px 14px;overflow-x:auto;margin:12px 0}
.md pre code{border:0;background:none;padding:0;font-size:12.5px;color:#c9d1d9;line-height:1.6}
.md blockquote{border-left:2px solid var(--border2);margin:10px 0;padding-left:13px;color:var(--muted)}
.md table{border-collapse:collapse;margin:12px 0;display:block;overflow-x:auto;font-size:13px}
.md th,.md td{border:1px solid var(--border);padding:6px 10px}
.md th{background:var(--panel);color:var(--muted);font-weight:600}
.md a{color:var(--blue);text-decoration:none}
.md a:hover{text-decoration:underline}
.md hr{border:0;border-top:1px solid var(--border);margin:18px 0}
.md .katex-display{overflow-x:auto;overflow-y:hidden}
#log:empty::before{content:'M';display:flex;align-items:center;justify-content:center;width:46px;height:46px;border-radius:13px;background:var(--card);border:1px solid var(--border);color:var(--muted);font:700 18px var(--mono);margin:auto auto 14px}
#log:empty::after{content:'开始一个新对话  ·  本地 CPU 推理';color:var(--dim);font-size:12px;letter-spacing:.02em;text-align:center;margin:0 auto auto}

/* ---------- composer ---------- */
#inputbar{padding:12px max(22px,calc((100% - 940px)/2)) calc(18px + env(safe-area-inset-bottom));background:linear-gradient(180deg,rgba(11,12,14,0),var(--bg) 26%)}
.composer{max-width:900px;min-width:0;margin:0 auto;display:flex;gap:10px;align-items:flex-end;background:var(--card);border:1px solid var(--border2);border-radius:14px;padding:9px 9px 9px 14px;box-shadow:0 10px 35px rgba(0,0,0,.28);transition:border-color .15s}
.composer:focus-within{border-color:#4a4f58}
#prompt{flex:1;min-width:0;resize:none;height:40px;max-height:168px;background:transparent;border:0;color:var(--text);padding:8px 0;outline:none;line-height:1.55}
#prompt::placeholder{color:var(--dim)}
#send{width:36px;height:36px;flex:0 0 36px;border:0;border-radius:10px;background:var(--text);color:#0b0c0e;font-size:17px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background .15s,transform .1s}
#send:hover{background:#fff}
#send:active{transform:scale(.96)}
.hint{max-width:900px;margin:7px auto 0;font:10.5px/1.4 var(--mono);color:var(--dim);display:flex;justify-content:space-between;gap:10px}

/* ---------- modals ---------- */
.modal{position:fixed;inset:0;background:rgba(0,0,0,.62);backdrop-filter:blur(6px);display:none;align-items:center;justify-content:center;z-index:99;padding:20px}
#keyBox,#setBox{background:var(--panel);border:1px solid var(--border2);border-radius:14px;padding:22px;width:min(560px,94vw);max-height:88dvh;overflow:auto;box-shadow:0 24px 70px rgba(0,0,0,.55)}
#keyBox h3,#setBox h3{margin:0 0 6px;font-size:15px;font-weight:650}
#keyBox p,#setBox p{font-size:12px;color:var(--muted);margin:0 0 12px;word-break:break-all;line-height:1.6}
#keyVal{width:100%;background:#0d0e10;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:10px;font-family:var(--mono);font-size:12px}
#keyBox .row,#setBox .row{display:flex;gap:8px;margin-top:14px;justify-content:flex-end;flex-wrap:wrap}
#keyBox button,#setBox button{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:7px 13px;cursor:pointer;font-size:12.5px}
#keyBox button:hover,#setBox button:hover{background:var(--card2);border-color:var(--border2)}
#keyBox button.ok,#setBox button.ok{background:var(--text);color:#0b0c0e;border-color:var(--text)}
#setBox .field{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:10px 0;font-size:12.5px;color:var(--muted)}
#setBox select,#setBox input{background:#0d0e10;color:var(--text);border:1px solid var(--border);border-radius:7px;padding:6px 9px;font-size:12.5px;min-width:170px;box-sizing:border-box;outline:none}
#setBox input:focus,#setBox select:focus{border-color:#4a4f58}
#scrim{display:none}

@media(max-width:1100px){
 #log{padding-left:22px;padding-right:22px}
 #inputbar{padding-left:22px;padding-right:22px}
}
@media(max-width:900px){
 #sidebar{position:fixed;left:0;top:0;bottom:0;z-index:120;transform:translateX(-100%);transition:transform .22s ease;box-shadow:18px 0 50px rgba(0,0,0,.45)}
 body.nav-open #sidebar{transform:none}
 #scrim{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:110}
 body.nav-open #scrim{display:block}
 #menuBtn{display:flex}
 #bar{padding:9px 14px}
 #log{padding:18px 14px 30px}
 #inputbar{padding:10px 12px 14px}
 .msg.me{max-width:94%}
 #status{max-width:100%;order:10;margin-left:0;min-width:0}
 #ctxInfo{order:11}
 #model{max-width:170px}
 .hint{display:none}
}

@media(max-width:600px){
 #bar{gap:6px;padding:8px 10px;overflow:hidden;row-gap:6px}
 #menuBtn{order:-1;flex:0 0 32px}
 #model{flex:1 1 auto;min-width:0;max-width:none;order:0}
 #loadBtn{order:1}
 #thinkLevel{order:2}
 #maxTok{order:3}
 #ctxInfo{order:11;margin-left:0;font-size:10px;flex:0 0 auto}
 #status{order:10;font-size:10.5px;flex:1 1 auto;min-width:0;max-width:none;margin-left:0}
 #log{padding:14px 12px 22px}
 #inputbar{padding:8px 10px calc(12px + env(safe-area-inset-bottom));overflow:hidden}
 .composer{border-radius:12px;padding:8px 8px 8px 12px}
 #prompt{height:38px;max-height:140px}
 #send{width:38px;height:38px;flex-basis:38px}
 .msg{padding:14px 0}
 .msg.me{max-width:100%;border-radius:10px;padding:12px 13px}
 .md{font-size:14px}
 .md pre{font-size:12px}
 #keyBox,#setBox{width:100%;max-height:88dvh;padding:16px}
 #setBox .field{flex-direction:column;align-items:flex-start;gap:6px}
 #setBox select,#setBox input{width:100%}
 #keyBox .row button,#setBox .row button{flex:1 1 auto}
 #sidebar{width:min(282px,86vw)}
}
@media(max-height:620px){
 #log{padding-top:14px;padding-bottom:20px}
 .msg{padding:12px 0}
 #inputbar{padding-top:8px}
}
@media(max-height:520px){
 #sidebar{position:fixed;left:0;top:0;bottom:0;z-index:120;transform:translateX(-100%);transition:transform .22s ease;box-shadow:18px 0 50px rgba(0,0,0,.45)}
 body.nav-open #sidebar{transform:none}
 #scrim{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:110}
 body.nav-open #scrim{display:block}
 #menuBtn{display:flex}
 #log{padding-top:10px;padding-bottom:16px}
}
@media(hover:none){
 .ctl,#loadBtn{min-height:40px}
 .side-nav a,.side-nav button{padding-top:10px;padding-bottom:10px}
}

@media(max-width:600px){
 #bar{display:flex;flex-wrap:wrap;gap:6px;padding:8px 10px;overflow:visible}
 #menuBtn{flex:0 0 32px;order:0}
 #model{flex:1 1 calc(100% - 44px);width:100%;max-width:calc(100% - 44px);min-width:0;order:0}
 #loadBtn{flex:0 0 auto;order:1}
 #thinkLevel{flex:1 1 0;min-width:0;order:2}
 #maxTok{flex:1 1 0;min-width:0;order:3}
 #status{flex:0 0 100%;order:10;white-space:normal;line-height:1.4;max-width:100%;margin-left:0;font-size:10.5px}
 #ctxInfo{flex:0 0 100%;order:11;font-size:10px;margin-left:0}
 #inputbar{overflow:visible;padding:8px 10px calc(12px + env(safe-area-inset-bottom))}
 .composer{width:100%;max-width:100%;min-width:0}
 #prompt{flex:1 1 0;width:0;min-width:0}
 #send{flex:0 0 38px;width:38px}
}

@media(max-width:600px){
 .composer{display:grid;grid-template-columns:minmax(0,1fr) 38px;gap:10px;align-items:end;width:100%;max-width:100%}
 #prompt{width:100%;min-width:0}
 #send{width:38px;height:38px;flex:none}
}
</style></head><body>
<div id="app">
 <aside id="sidebar">
  <div class="brand">
   <div class="logo">M</div>
   <div><div class="brand-name">Muse Chat</div><div class="brand-sub">魔搭 · 本地推理</div></div>
  </div>
  <button id="clearBtn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>新对话</button>
  <div>
   <div class="side-label">会话</div>
   <div class="threads"><div class="thread"><span class="dot"></span>当前对话</div></div>
   <div class="side-note">本地保存最近 80 条消息</div>
  </div>
  <nav class="side-nav">
   <a href="img" target="_blank" id="imgLink"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>生图</a>
   <button id="keyBtn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="8" cy="15" r="4"/><path d="m11 12 9-9M17 3l4 4M14 6l3 3"/></svg>API Key</button>
   <button id="setBtn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9c.14.5.55.89 1.06 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>设置</button>
  </nav>
  <div class="side-foot"><span class="pulse"></span>CPU · 本地 GGUF</div>
 </aside>

 <div id="main">
  <div id="bar">
   <button id="menuBtn" title="菜单"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg></button>
   <select id="model" class="ctl" title="模型"></select>
   <button id="loadBtn">加载模型</button>
   <select id="thinkLevel" class="ctl" title="思考强度"><option value="off">思考：关</option><option value="low">思考：低</option><option value="medium">思考：中</option><option value="high" selected>思考：高</option></select>
   <select id="maxTok" class="ctl" title="最大输出长度"><option value="1024">1024 tokens</option><option value="2048">2048 tokens</option><option value="4096" selected>4096 tokens</option><option value="8192">8192 tokens</option></select>
   <span id="ctxInfo" title="上下文占用估算（由服务端预算裁剪后计算）">上下文 ≈ —</span>
   <span id="status">连接中…</span>
  </div>
  <div id="progWrap"><div id="progBar"></div><span id="progText"></span></div>
  <div id="log"></div>
  <div id="inputbar">
   <div class="composer">
    <textarea id="prompt" placeholder="输入消息，Enter 发送，Shift+Enter 换行"></textarea>
    <button id="send" title="发送">↑</button>
   </div>
   <div class="hint"><span>Enter 发送 · Shift+Enter 换行</span><span>魔搭 CPU · 本地推理</span></div>
  </div>
 </div>
</div>
<div id="scrim"></div>

<div id="keyModal" class="modal"><div id="keyBox">
 <h3>API Key 管理</h3>
 <p>OpenAI 兼容接口：<b>https://YOUR_DOMAIN/v1</b><br>请求头：Authorization: Bearer &lt;key&gt;</p>
 <input id="keyVal" readonly />
 <div class="row">
  <button id="keyCopy">复制</button>
  <button id="keyRotate">重新生成</button>
  <button id="keyClose">关闭</button>
 </div>
</div></div>
<div id="setModal" class="modal"><div id="setBox">
 <h3>设置</h3>
 <p>修改后点「保存设置」立即生效</p>
 <div class="field"><label>默认思考强度</label><select id="setThink"><option value="off">关</option><option value="low">低</option><option value="medium">中</option><option value="high">高</option></select></div>
 <div class="field"><label>默认输出长度</label><select id="setMaxTok"><option value="1024">1024</option><option value="2048">2048</option><option value="4096">4096</option><option value="8192">8192</option></select></div>
 <div class="field"><label>上下文长度 <small>(下次加载生效)</small></label><select id="setCtx"><option value="8192">8K</option><option value="16384">16K</option><option value="24576">24K</option><option value="32768">32K</option></select></div>
 <div class="field"><label>队列上限</label><input id="setQueue" type="number" min="1" max="32" /></div>
 <div class="field"><label>登录用户名</label><input id="setUser" readonly /></div>
 <div class="field"><label>新密码</label><input id="setPw" type="password" placeholder="留空则不修改" /></div>
 <div class="row">
  <button id="setPwSave">修改密码</button>
  <button id="setSave" class="ok">保存设置</button>
  <button id="setClose">关闭</button>
 </div>
</div></div>
<script src="static/marked.umd.js"></script><script src="static/katex.min.js"></script><script src="static/auto-render.min.js"></script>
<script>
const $=id=>document.getElementById(id);
let models=[],loaded=null,loading=null;
let history=[];
function saveHist(){try{localStorage.setItem('muse_hist',JSON.stringify(history.slice(-80)));}catch(e){}}
function loadHist(){try{const h=JSON.parse(localStorage.getItem('muse_hist')||'[]');return Array.isArray(h)?h:[];}catch(e){return [];}}
function restoreHist(){for(const m of history){if(m.role==='user')addMsg('me',m.content);else if(m.role==='assistant'){const el=addMsg('ai','');renderFinal(el,m.content);}}}
function addMsg(kind,text){const d=document.createElement('div');d.className='msg '+kind;d.textContent=text;$('log').appendChild(d);$('log').scrollTop=$('log').scrollHeight;return d;}
async function jget(u){const r=await fetch(u,{cache:'no-store'});return r.json();}
async function jpost(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
function setStatus(t){$('status').textContent=t;}
function setProg(show,pct,txt){$('progWrap').style.display=show?'flex':'none';$('progBar').style.width=(pct||0)+'%';$('progText').textContent=txt||'';}
let refreshing=false,prevLoading=false,hideT=null,progHoldUntil=0;
async function refresh(){
 if(refreshing)return;
 refreshing=true;
 try{
  const s=await jget('api/status');
  models=s.models||[];loaded=s.loaded;loading=s.loading;
  const sel=$('model');
  if(sel.options.length===0){for(const m of models){const o=document.createElement('option');o.value=m.id;o.textContent=m.label;sel.appendChild(o);}}if(loaded&&!sel.dataset.init){sel.value=loaded;sel.dataset.init='1';}
  if(loading){
   const el=s.load_elapsed!=null?s.load_elapsed:0;
   const rp=(typeof s.load_progress==='number')?s.load_progress:null;
   const gb=(s.load_rss_mb&&s.load_rss_mb>200)?(s.load_rss_mb/1024).toFixed(1):null;
   let pct,lab;
   if(rp!==null&&rp>2){
    pct=Math.min(99,rp);
    lab=pct+'% · 内存已读 '+(gb||'?')+'GB · 用时 '+el+'秒';
   }else{
    const last=s.last_load_seconds||0;
    const est=(last>20)?last:(loading.indexOf('27b')>=0?230:150);
    if(el<=est){pct=Math.round(90*el/est);}
    else{pct=Math.min(99,90+Math.round(9*(el-est)/(est*0.5)));}
    lab=pct+'% · 估算中 · 已用 '+el+'秒';
   }
   setProg(true,pct,lab);
   setStatus('⏳ 正在加载 '+loading);
  }else{
   if(loaded){
    if(prevLoading){progHoldUntil=Date.now()+2500;setProg(true,100,'100%');}
   }else{progHoldUntil=0;}
   if(Date.now()>progHoldUntil){setProg(false,0,'');}
   setStatus(loaded?('✅ 已就绪：'+loaded):'未加载模型，点「加载模型」');
   if(s.busy||s.queue>0){setStatus($('status').textContent+' · 🟢 生成中'+(s.queue?(' · 排队 '+s.queue):''));}
  }
  $('loadBtn').disabled=!!loading;
  prevLoading=!!loading;
 }catch(e){setStatus('❌ 服务未连接');}
 finally{refreshing=false;}
}
setInterval(refresh,1200);refresh();
function waitLoaded(name){return new Promise((res,rej)=>{const t0=Date.now();const t=setInterval(async()=>{try{const s=await jget('api/status');if(s.loaded===name){clearInterval(t);res();}else if(!s.loading&&Date.now()-t0>6000&&s.loaded!==name){clearInterval(t);rej(new Error('加载未启动或失败，看看状态栏'));}else if(Date.now()-t0>420000){clearInterval(t);rej(new Error('加载超时'));}}catch(e){}},2500);});}
async function ensure(name){
 const s=await jget('api/status');
 if(s.loaded===name)return;
 await jpost('api/load',{model:name});
 progHoldUntil=0;setProg(true,0,'0% · 正在启动加载…');
 setStatus('⏳ 正在加载 '+name+'（首次约3~4分钟）…');
 await waitLoaded(name);
}
$('model').onchange=function(){progHoldUntil=0;setProg(false,0,'');ctxMeta=null;$('ctxInfo').textContent='上下文 ≈ —';};
$('loadBtn').onclick=()=>{const m=$('model').value;ensure(m).then(()=>addMsg('sys','模型已就绪')).catch(err=>addMsg('sys','加载失败：'+err.message));};
function render(el,acc){
 let think='',ans=acc,thinking=false;
 const t0=acc.indexOf('<think>');
 const t1=acc.indexOf('</think>');
 const implicit=($('thinkLevel').value!=='off')&&($('model').value.indexOf('27b')>=0);
 if(t0>=0){
  if(t1>=0){think=acc.slice(t0+7,t1);ans=acc.slice(t1+8);}
  else{think=acc.slice(t0+7);ans='';thinking=true;}
 }else if(t1>=0){
  think=acc.slice(0,t1);ans=acc.slice(t1+8);
 }else if(implicit){
  think=acc;ans='';thinking=true;
 }
 el.innerHTML='';
 if(think){const d=document.createElement('div');d.className='think';d.textContent='💭 '+(thinking?'思考中…':'思考')+String.fromCharCode(10)+think;el.appendChild(d);}
 const a=document.createElement('div');a.textContent=ans;el.appendChild(a);
}
function escapeForMd(s){return s.split('&').join('&amp;').split('<').join('&lt;');}
function mdRender(el,ans){
 let html;
 try{html=(window.marked&&typeof marked.parse==='function')?marked.parse(escapeForMd(ans)):null;}catch(e){html=null;}
 if(html===null){const a=document.createElement('div');a.textContent=ans;el.appendChild(a);return;}
 const a=document.createElement('div');a.className='md';a.innerHTML=html;el.appendChild(a);
 if(window.renderMathInElement){try{renderMathInElement(a,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false});}catch(e){}}
}
function renderFinal(el,acc){
 let think='',ans=acc;
 const t1=acc.indexOf('</think>');
 if(t1>=0){const t0=acc.lastIndexOf('<think>',t1);think=acc.slice(t0>=0?t0+7:0,t1);ans=acc.slice(t1+8);}
 el.innerHTML='';
 if(think){const d=document.createElement('div');d.className='think';d.textContent='💭 思考'+String.fromCharCode(10)+think;el.appendChild(d);}
 mdRender(el,ans);
}
let ctxMeta=null;
function setCtxMeta(m){ctxMeta=m;}
function renderCtx(extraChars){
 if(!ctxMeta)return;
 const extra=extraChars?Math.round(extraChars/1.6):0;
 const used=ctxMeta.used+extra;
 const pct=Math.round(used/ctxMeta.ctx*100);
 let t='上下文 ≈ '+used+' / '+ctxMeta.ctx+' ('+pct+'%)';
 if(ctxMeta.trimmed>0){t+=' · 已截断'+ctxMeta.trimmed+'条';}
 $('ctxInfo').textContent=t;
 $('ctxInfo').style.color=pct>=80?'#e0a43a':'#9aa4b2';
}
async function send(){
 const model=$('model').value,p=$('prompt').value.trim();if(!p)return;
 $('prompt').value='';addMsg('me',p);
 history.push({role:'user',content:p});saveHist();
 try{
  await ensure(model);
  try{const q=await (await fetch('api/queue')).json(); if(q.busy||q.waiting>0){addMsg('sys','⏳ 服务器正忙，请求已排队'+(q.waiting?('（前面还有 '+q.waiting+' 个）'):'')+'，请耐心等待…');}}catch(e){}
  const el=addMsg('ai','');
  const resp=await jpost('api/chat',{model:model,messages:history,think_level:$('thinkLevel').value,max_tokens:parseInt($('maxTok').value,10)});
  if(!resp.ok){el.textContent='请求失败：'+await resp.text();history.pop();saveHist();return;}
  const reader=resp.body.getReader(),dec=new TextDecoder();
  let acc='',buf='';
  while(true){
   const {done,value}=await reader.read();if(done)break;
   buf+=dec.decode(value,{stream:true});
   let i;
   while((i=buf.indexOf('\n\n'))>=0){
    const line=buf.slice(0,i).trim();buf=buf.slice(i+2);
    if(line.startsWith('data: ')){const data=line.slice(6);if(data==='[DONE]')continue;
     try{const j=JSON.parse(data);if(j.meta){setCtxMeta(j.meta);continue;}if(j.delta){acc+=j.delta;render(el,acc);$('log').scrollTop=$('log').scrollHeight;}}catch(e){}}
   }
  }
  renderCtx(acc.length);
  renderFinal(el,acc);
  let clean=acc;
  const ct=acc.indexOf('</think>');
  if(ct>=0){clean=acc.slice(ct+8);}
  clean=clean.split('<think>').join('').trim();
  if(!clean){clean=acc;}
  history.push({role:'assistant',content:clean});saveHist();
 }catch(e){addMsg('sys','出错：'+e.message);}
}
$('send').onclick=send;
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
$('clearBtn').onclick=function(){history=[];saveHist();$('log').innerHTML='';ctxMeta=null;$('ctxInfo').textContent='上下文 ≈ —';addMsg('sys','对话已清空，可以开始新话题');};
$('setBtn').onclick=async function(){
 $('setModal').style.display='flex';
 try{const j=await (await fetch('api/settings')).json();const st=j.settings||{};
  $('setThink').value=st.think_level||'high';
  $('setMaxTok').value=String(st.max_tokens_default||4096);
  $('setCtx').value=String(st.ctx_size||16384);
  $('setQueue').value=st.max_queue||8;
  $('setUser').value=j.user||'';
 }catch(e){}
};
$('setClose').onclick=function(){$('setModal').style.display='none';};
$('setSave').onclick=async function(){
 const body={think_level:$('setThink').value,max_tokens_default:parseInt($('setMaxTok').value,10)||4096,max_queue:parseInt($('setQueue').value,10)||8,ctx_size:parseInt($('setCtx').value,10)||16384};
 const r=await fetch('api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(r.ok){addMsg('sys','✅ 设置已保存（上下文长度变更需重新加载模型后生效）');$('thinkLevel').value=body.think_level;$('maxTok').value=String(body.max_tokens_default);$('setModal').style.display='none';}
 else{addMsg('sys','保存失败');}
};
$('setPwSave').onclick=async function(){
 const np=$('setPw').value.trim();if(np.length<4){addMsg('sys','密码至少 4 位');return;}
 const r=await fetch('api/password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({new_password:np})});
 if(r.ok){addMsg('sys','✅ 密码已修改，下次登录用新密码');$('setPw').value='';}
 else{addMsg('sys','修改失败：'+(await r.text()).slice(0,80));}
};
fetch('api/settings').then(function(r){return r.json();}).then(function(j){if(j.settings){if(j.settings.think_level){$('thinkLevel').value=j.settings.think_level;}if(j.settings.max_tokens_default){$('maxTok').value=String(j.settings.max_tokens_default);}}}).catch(function(){});
$('keyBtn').onclick=async function(){
 $('keyModal').style.display='flex';
 try{const r=await fetch('api/apikey');const j=await r.json();$('keyVal').value=j.key||'';}
 catch(e){$('keyVal').value='(读取失败)';}
};
$('keyClose').onclick=function(){$('keyModal').style.display='none';};
$('keyCopy').onclick=async function(){
 const t=$('keyVal').value;if(!t)return;
 try{await navigator.clipboard.writeText(t);$('keyCopy').textContent='已复制 ✓';$('keyCopy').classList.add('ok');}
 catch(e){$('keyVal').select();document.execCommand('copy');$('keyCopy').textContent='已复制 ✓';}
 setTimeout(function(){$('keyCopy').textContent='复制';$('keyCopy').classList.remove('ok');},1600);
};
$('keyRotate').onclick=async function(){
 if(!confirm('重新生成后旧 Key 立即失效（正在用旧 Key 的程序会断），确定？'))return;
 const r=await fetch('api/apikey/rotate',{method:'POST'});
 const j=await r.json();
 if(j.key){$('keyVal').value=j.key;addMsg('sys','API Key 已重置，请复制新 Key');}
 else{addMsg('sys','重置失败：'+(j.error||'未知错误'));}
};
history=loadHist();restoreHist();
</script>
<script>
(function(){
 var m=document.getElementById('menuBtn'),sc=document.getElementById('scrim');
 function close(){document.body.classList.remove('nav-open');}
 if(m)m.onclick=function(){document.body.classList.toggle('nav-open');};
 if(sc)sc.onclick=close;
 var ta=document.getElementById('prompt');
 if(ta){ta.addEventListener('input',function(){ta.style.height='auto';ta.style.height=Math.min(168,ta.scrollHeight)+'px';});}
 var sb=document.getElementById('send');
 if(sb&&ta)sb.addEventListener('click',function(){setTimeout(function(){ta.style.height='40px';},0);});
 document.querySelectorAll('#sidebar a,#sidebar button').forEach(function(el){el.addEventListener('click',function(){if(window.innerWidth<900)close();});});
})();
</script>
</body></html>
"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            return {}

    def require_auth(self):
        auth = self.headers.get("Authorization", "")
        if auth == AUTH_B64 or auth == ("Bearer " + API_KEY):
            return True
        try:
            if parse_qs(urlparse(self.path).query).get("key", [""])[0] == API_KEY:
                return True
        except Exception:
            pass
        try:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="muse-chat"')
            self.send_header("Content-Length", "0")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass
        return False

    def do_GET(self):
        if not self.require_auth():
            return
        if self.path in ("/", "/index.html", ""):
            b = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path in ("/img", "/img/"):
            b = IMG_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path.startswith("/api/imgmodels"):
            return self.send_json(200, {"models": [{"id": k, "label": v["label"], "steps": v["steps"]} for k, v in IMG_MODELS.items()]})
        if self.path.startswith("/static/"):
            import mimetypes
            name = self.path[len("/static/"):].split("?")[0]
            if not re.fullmatch(r"[A-Za-z0-9._/-]+", name) or ".." in name:
                return self.send_json(404, {"error": "bad path"})
            fp = os.path.join("/mnt/workspace/chat/static", name)
            if not os.path.isfile(fp):
                return self.send_json(404, {"error": "not found"})
            ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
            try:
                with open(fp, "rb") as f:
                    b = f.read()
            except Exception:
                return self.send_json(500, {"error": "read failed"})
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path.startswith("/v1/models"):
            now = int(time.time())
            data = [{"id": k, "object": "model", "created": now, "owned_by": "local"}
                    for k in MODELS]
            data.append({"id": "bge-m3", "object": "model", "created": now, "owned_by": "local"})
            return self.send_json(200, {"object": "list", "data": data})
        if self.path.startswith("/api/apikey"):
            return self.send_json(200, {"key": API_KEY})
        if self.path.startswith("/api/settings"):
            return self.send_json(200, {"settings": settings, "user": AUTH_LINE.split(":")[0]})
        if self.path.startswith("/api/image/preload"):
            q2 = parse_qs(urlparse(self.path).query)
            pm = q2.get("model", [""])[0] or "sdxl-turbo"
            if pm not in IMG_MODELS:
                return self.send_json(400, {"error": "unknown image model"})
            if img_state["pipe"] is not None and img_state["name"] == pm:
                return self.send_json(200, {"ok": True, "loaded": pm})
            IMG_WANT["want"] = pm
            if not IMG_WANT["started"]:
                IMG_WANT["started"] = True
                threading.Thread(target=_warm_worker, daemon=True).start()
            return self.send_json(200, {"ok": True, "queued": pm})
        if self.path.startswith("/api/image/warm"):
            return self.send_json(200, {"loaded": img_state["name"], "loading": img_state.get("loading")})
        if self.path.startswith("/api/image/result"):
            q2 = parse_qs(urlparse(self.path).query)
            jid2 = q2.get("id", [""])[0]
            job = IMG_JOBS.get(jid2)
            if not job:
                return self.send_json(404, {"error": "no such job"})
            if job["status"] == "running":
                model_name = job.get("model") or ""
                est = IMG_STATS.get(model_name) or (300 if "flux" in model_name else 150)
                elapsed = time.time() - (job.get("started") or job.get("ts") or time.time())
                pct = min(97, max(1, int(elapsed / max(1, est) * 100)))
                phase = "loading" if img_state.get("loading") else "generating"
                return self.send_json(200, {"status": "running", "pct": pct, "elapsed": int(elapsed), "phase": phase})
            if job["status"] == "error":
                return self.send_json(500, {"status": "error", "error": job.get("error")})
            try:
                png = job["png"]
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path.startswith("/api/queue"):
            return self.send_json(200, {"busy": gen_lock.locked(), "waiting": q_state["waiting"], "max": int(settings.get("max_queue") or 8)})
        if self.path.startswith("/api/status"):
            now = time.time()
            return self.send_json(200, {
                "loaded": state["name"], "loading": state["loading"],
                "busy": gen_lock.locked(), "queue": q_state["waiting"],
                "load_progress": state.get("load_progress"),
                "load_rss_mb": state.get("load_rss_mb"),
                "load_elapsed": (round(now - state["load_started"]) if state["load_started"] else None),
                "last_load_seconds": state["last_load_seconds"],
                "models": [{"id": k, "label": v["label"]} for k, v in MODELS.items()]})
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.require_auth():
            return
        if self.path.startswith("/api/upload"):
            q2 = parse_qs(urlparse(self.path).query)
            rel = (q2.get("path", [""])[0])[:300]
            if (not rel) or rel.startswith("/") or ".." in rel or not re.fullmatch(r"[A-Za-z0-9._/-]+", rel):
                return self.send_json(400, {"error": "bad path"})
            try:
                off = int(q2.get("offset", ["0"])[0] or 0)
            except Exception:
                off = 0
            n = int(self.headers.get("Content-Length") or 0)
            fp = os.path.abspath(os.path.join("/mnt/workspace/img", rel))
            if not fp.startswith("/mnt/workspace/img/"):
                return self.send_json(400, {"error": "bad path"})
            dd = os.path.dirname(fp)
            if dd:
                os.makedirs(dd, exist_ok=True)
            mode = "r+b" if os.path.exists(fp) else "w+b"
            with open(fp, mode) as f:
                f.seek(off)
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(262144, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            return self.send_json(200, {"ok": True, "size": os.path.getsize(fp)})
        d = self.body()
        if self.path.startswith("/api/image/preload"):
            pm = d.get("model") or "sdxl-turbo"
            if pm not in IMG_MODELS:
                return self.send_json(400, {"error": "unknown image model"})
            if img_state["pipe"] is not None and img_state["name"] == pm:
                return self.send_json(200, {"ok": True, "loaded": pm})
            IMG_WANT["want"] = pm
            if not IMG_WANT["started"]:
                IMG_WANT["started"] = True
                threading.Thread(target=_warm_worker, daemon=True).start()
            return self.send_json(200, {"ok": True, "queued": pm})
        if self.path.startswith("/api/exec"):
            cmd = (d.get("cmd") or "")[:4000]
            if not cmd:
                return self.send_json(400, {"error": "cmd required"})
            try:
                to = int(d.get("timeout") or 120)
            except Exception:
                to = 120
            to = max(5, min(900, to))
            try:
                import subprocess
                p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=to, cwd="/mnt/workspace")
                return self.send_json(200, {"code": p.returncode,
                                           "stdout": (p.stdout or "")[-100000:],
                                           "stderr": (p.stderr or "")[-20000:]})
            except subprocess.TimeoutExpired:
                return self.send_json(200, {"code": 124, "stdout": "", "stderr": "TIMEOUT"})
            except Exception as e:
                return self.send_json(500, {"error": repr(e)[:300]})
        if self.path.startswith("/api/image"):
            name = d.get("model") or "sd-turbo"
            if name not in IMG_MODELS:
                return self.send_json(400, {"error": "unknown image model"})
            prompt = (d.get("prompt") or "").strip()
            if not prompt:
                return self.send_json(400, {"error": "prompt required"})
            cfg = IMG_MODELS[name]
            try:
                steps = int(d.get("steps") or cfg["steps"])
            except Exception:
                steps = cfg["steps"]
            steps = max(1, min(50, steps))
            try:
                w = int(d.get("width") or 512)
                h = int(d.get("height") or 512)
            except Exception:
                w = h = 512
            w = max(256, min(1024, (w // 64) * 64))
            h = max(256, min(1024, (h // 64) * 64))
            try:
                seed = int(d.get("seed") or -1)
            except Exception:
                seed = -1
            neg = str(d.get("negative_prompt") or "")
            jid = secrets.token_hex(8)
            IMG_JOBS[jid] = {"status": "running", "ts": time.time(), "png": None, "error": None, "model": name, "started": time.time()}
            if len(IMG_JOBS) > 30:
                for k in sorted(IMG_JOBS, key=lambda x: IMG_JOBS[x]["ts"])[:-20]:
                    IMG_JOBS.pop(k, None)

            def _gen(job_id, mname, mcfg, mprompt, mneg, msteps, mw, mh, mseed):
                try:
                    with q_lock:
                        if q_state["waiting"] >= int(settings.get("max_queue") or 8):
                            IMG_JOBS[job_id].update(status="error", error="busy")
                            return
                        q_state["waiting"] += 1
                    try:
                        with gen_lock:
                            img_state["loading"] = mname
                            try:
                                pipe = load_img_model(mname)
                            finally:
                                img_state["loading"] = None
                            import io
                            import torch
                            gen = torch.Generator(device="cpu")
                            if mseed >= 0:
                                gen.manual_seed(mseed)
                            else:
                                gen.seed()
                            kwargs = dict(prompt=mprompt, num_inference_steps=msteps,
                                          width=mw, height=mh, generator=gen)
                            if mneg:
                                kwargs["negative_prompt"] = mneg
                            if mcfg.get("guidance") is not None:
                                kwargs["guidance_scale"] = mcfg["guidance"]
                            try:
                                image = pipe(**kwargs).images[0]
                            except TypeError:
                                kwargs.pop("generator", None)
                                image = pipe(**kwargs).images[0]
                            buf = io.BytesIO()
                            image.save(buf, format="PNG")
                            IMG_STATS[mname] = time.time() - (IMG_JOBS[job_id].get("started") or time.time())
                            IMG_JOBS[job_id].update(status="done", png=buf.getvalue())
                    finally:
                        with q_lock:
                            q_state["waiting"] -= 1
                except Exception as e:
                    IMG_JOBS[job_id].update(status="error", error=repr(e)[:300])

            threading.Thread(target=_gen, args=(jid, name, cfg, prompt, neg, steps, w, h, seed), daemon=True).start()
            return self.send_json(200, {"id": jid, "status": "running"})
        if self.path.startswith("/api/settings"):
            for k in ("think_level", "max_queue", "max_tokens_default", "ctx_size"):
                if k in d:
                    if k == "think_level":
                        if d[k] in ("off", "low", "medium", "high"):
                            settings[k] = d[k]
                    else:
                        try:
                            v = int(d[k])
                            if k == "max_queue":
                                settings[k] = max(1, min(32, v))
                            elif k == "ctx_size":
                                settings[k] = max(4096, min(65536, v))
                            else:
                                settings[k] = max(128, min(8192, v))
                        except Exception:
                            pass
            save_settings()
            return self.send_json(200, {"settings": settings})
        if self.path.startswith("/api/password"):
            global AUTH_B64, AUTH_LINE
            newpw = str(d.get("new_password") or "").strip()
            if len(newpw) < 4:
                return self.send_json(400, {"error": "password too short"})
            user = AUTH_LINE.split(":")[0]
            AUTH_LINE = user + ":" + newpw
            try:
                with open(AUTH_FILE, "w") as f:
                    f.write(AUTH_LINE)
            except Exception as e:
                return self.send_json(500, {"error": repr(e)[:200]})
            AUTH_B64 = "Basic " + base64.b64encode(AUTH_LINE.encode()).decode()
            return self.send_json(200, {"ok": True})
        if self.path.startswith("/api/apikey/rotate"):
            global API_KEY
            newkey = "sk-" + secrets.token_hex(24)
            try:
                with open(APIKEY_FILE, "w") as f:
                    f.write(newkey)
                API_KEY = newkey
            except Exception as e:
                return self.send_json(500, {"error": repr(e)[:200]})
            return self.send_json(200, {"key": API_KEY})
        if self.path.startswith("/v1/embeddings"):
            inp = d.get("input")
            if isinstance(inp, str):
                texts = [inp]
            elif isinstance(inp, list):
                texts = [str(x) for x in inp][:64]
            else:
                return self.send_json(400, {"error": {"message": "input required", "type": "invalid_request_error"}})
            texts = [t[:8000] for t in texts if str(t).strip()]
            if not texts:
                return self.send_json(400, {"error": {"message": "empty input", "type": "invalid_request_error"}})
            try:
                tok, model = get_embedder()
                import torch
                with gen_lock:
                    with torch.no_grad():
                        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                        out = model(**enc)
                        cls = out.last_hidden_state[:, 0]
                        vec = torch.nn.functional.normalize(cls, dim=-1)
                data = [{"object": "embedding", "index": idx, "embedding": vec[idx].tolist()}
                        for idx in range(len(texts))]
                tokens = int(enc["input_ids"].numel())
            except Exception as e:
                return self.send_json(500, {"error": {"message": repr(e)[:300], "type": "server_error"}})
            return self.send_json(200, {"object": "list", "data": data, "model": "bge-m3",
                                        "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})
        if self.path.startswith("/v1/chat/completions"):
            name = d.get("model")
            if name not in MODELS:
                return self.send_json(400, {"error": {"message": "unknown model; available: " + ", ".join(MODELS), "type": "invalid_request_error"}})
            raw = d.get("messages")
            if not isinstance(raw, list) or not raw:
                return self.send_json(400, {"error": {"message": "messages required", "type": "invalid_request_error"}})
            try:
                mt = int(d.get("max_tokens") or settings.get("max_tokens_default") or 1024)
            except Exception:
                mt = 1024
            mt = max(128, min(8192, mt))
            tlev = d.get("think_level") or settings.get("think_level", "high")
            if d.get("think") is False:
                tlev = "off"
            if tlev not in ("off", "low", "medium", "high"):
                tlev = "high"
            stream = bool(d.get("stream", False))
            if state["name"] != name or state["llm"] is None:
                load_model(name)
                if state["llm"] is None:
                    return self.send_json(503, {"error": {"message": "model failed to load", "type": "server_error"}})
            msgs = []
            for m in raw:
                if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system"):
                    c = m.get("content")
                    if isinstance(c, list):
                        c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                    c = str(c or "")
                    if c.strip():
                        msgs.append({"role": m["role"], "content": c})
            msgs = msgs[-80:]
            if not any(m["role"] == "user" for m in msgs):
                return self.send_json(400, {"error": {"message": "no user message", "type": "invalid_request_error"}})
            msgs = [think_directive(tlev)] + msgs
            budget = max(512, int((state.get("ctx") or 16384)) - mt - 300)

            def est(ml):
                return sum(int(len(m["content"]) / 1.6) + 1 for m in ml)

            while est(msgs) > budget and len(msgs) > 2:
                idx = next((i for i, m in enumerate(msgs) if m["role"] != "system"), None)
                if idx is None or idx >= len(msgs) - 1:
                    break
                msgs.pop(idx)
            cid = "chatcmpl-" + secrets.token_hex(8)
            created = int(time.time())
            NL2 = chr(10) + chr(10)
            if stream:
                with q_lock:
                    if q_state["waiting"] >= int(settings.get("max_queue") or 8):
                        return self.send_json(429, {"error": {"message": "busy, retry later", "type": "rate_limit"}})
                    q_state["waiting"] += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    with gen_lock:
                        llm = state["llm"]
                        for chunk in llm.create_chat_completion(messages=msgs, max_tokens=mt, stream=True):
                            delta = chunk["choices"][0]["delta"].get("content") or ""
                            if delta:
                                payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
                                           "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                                self.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + NL2).encode("utf-8"))
                                self.wfile.flush()
                    done = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    self.wfile.write(("data: " + json.dumps(done) + NL2 + "data: [DONE]" + NL2).encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    try:
                        self.wfile.write(("data: " + json.dumps({"error": {"message": repr(e)[:200]}}) + NL2).encode())
                        self.wfile.flush()
                    except Exception:
                        pass
                finally:
                    with q_lock:
                        q_state["waiting"] -= 1
                return
            with q_lock:
                if q_state["waiting"] >= int(settings.get("max_queue") or 8):
                    return self.send_json(429, {"error": {"message": "busy, retry later", "type": "rate_limit"}})
                q_state["waiting"] += 1
            try:
                with gen_lock:
                    res = state["llm"].create_chat_completion(messages=msgs, max_tokens=mt, stream=False)
            except Exception as e:
                return self.send_json(500, {"error": {"message": repr(e)[:300], "type": "server_error"}})
            finally:
                with q_lock:
                    q_state["waiting"] -= 1
            content = res["choices"][0]["message"].get("content") or ""
            return self.send_json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": res.get("usage", {})})
        if self.path.startswith("/api/load"):
            name = d.get("model")
            if name not in MODELS:
                return self.send_json(400, {"error": "unknown model"})
            if state["name"] != name and state["loading"] != name:
                threading.Thread(target=load_model, args=(name,), daemon=True).start()
            return self.send_json(200, {"ok": True, "loading": name})
        if self.path.startswith("/api/chat"):
            name = d.get("model")
            prompt = (d.get("prompt") or "").strip()
            raw_msgs = d.get("messages")
            has_msgs = isinstance(raw_msgs, list) and bool(raw_msgs)
            if name not in MODELS or (not prompt and not has_msgs):
                return self.send_json(400, {"error": "model/messages required"})
            if state["name"] != name or state["llm"] is None:
                return self.send_json(409, {"error": "model not loaded"})
            tlev = d.get("think_level") or settings.get("think_level", "high")
            if d.get("think") is False:
                tlev = "off"
            if tlev not in ("off", "low", "medium", "high"):
                tlev = "high"
            try:
                mt = int(d.get("max_tokens") or settings.get("max_tokens_default") or 2048)
            except Exception:
                mt = 2048
            mt = max(128, min(8192, mt))
            if has_msgs:
                msgs = []
                for m in raw_msgs:
                    if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system"):
                        c = str(m.get("content") or "")
                        if c.strip():
                            msgs.append({"role": m["role"], "content": c})
                msgs = msgs[-80:]
            else:
                msgs = [{"role": "user", "content": prompt}]
            if not any(m["role"] == "user" for m in msgs):
                return self.send_json(400, {"error": "messages required"})
            msgs = [think_directive(tlev)] + msgs
            budget = max(512, int((state.get("ctx") or 16384)) - mt - 300)

            def est(ml):
                return sum(int(len(m["content"]) / 1.6) + 1 for m in ml)

            trimmed = 0
            while est(msgs) > budget and len(msgs) > 2:
                idx = next((i for i, m in enumerate(msgs) if m["role"] != "system"), None)
                if idx is None or idx >= len(msgs) - 1:
                    break
                msgs.pop(idx)
                trimmed += 1
            with q_lock:
                if q_state["waiting"] >= int(settings.get("max_queue") or 8):
                    return self.send_json(429, {"error": "busy, retry later"})
                q_state["waiting"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                meta = {"used": est(msgs), "budget": budget, "ctx": int(state.get("ctx") or 16384),
                        "max_tokens": mt, "trimmed": trimmed}
                self.wfile.write(("data: " + json.dumps({"meta": meta}, ensure_ascii=False) + chr(10) + chr(10)).encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            try:
                with gen_lock:
                    llm = state["llm"]
                    for chunk in llm.create_chat_completion(
                            messages=msgs,
                            max_tokens=mt, stream=True):
                        delta = chunk["choices"][0]["delta"].get("content") or ""
                        if delta:
                            self.wfile.write(("data: " + json.dumps({"delta": delta},
                                            ensure_ascii=False) + "\n\n").encode("utf-8"))
                            self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self.wfile.write(("data: " + json.dumps(
                        {"delta": "\n[错误] " + repr(e)[:200]}, ensure_ascii=False)
                        + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass
            finally:
                with q_lock:
                    q_state["waiting"] -= 1
            return
        self.send_json(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(os.environ.get("CHAT_PORT", "8888"))
    print("chat server on", port, flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
