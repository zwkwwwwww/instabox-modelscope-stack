#!/usr/bin/env python3
# ModelScope chat: model-switchable chat UI over llama.cpp GGUF models on NAS.
# Runs in the ModelScope container on :8888. One model resident at a time (RAM-safe).
import base64, json, os, re, secrets, sys, time, gc, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

MODELS = {
    "qwen3-30b-a3b": {"path": "/mnt/workspace/gguf/30b/Qwen3-30B-A3B-Q4_K_M.gguf",
                      "label": "Qwen3-30B-A3B · 约7 tok/s",
                      "think_levels": ["off", "high"],
                      "think_note": "关/开（硬开关）"},
    "qwen3.8-27b":   {"path": "/mnt/workspace/gguf/27b/Qwen3.8-27B-Q4_K_M.gguf",
                      "label": "Qwen3.8-27B · 约3.3 tok/s · 最强",
                      "think_levels": ["high"],
                      "think_note": "固定深度思考（无硬开关）"},
    "qwen3-4b":      {"path": "/mnt/workspace/gguf/4b/Qwen3-4B-Q4_K_M.gguf",
                      "label": "Qwen3-4B · 约15-20 tok/s · 快",
                      "think_levels": ["off", "high"],
                      "think_note": "关/开（硬开关）"},
    "qwen3-1.7b":    {"path": "/mnt/workspace/gguf/17b/Qwen3-1.7B-Q4_K_M.gguf",
                      "label": "Qwen3-1.7B · 约30 tok/s · 秒回",
                      "think_levels": ["off", "high"],
                      "think_note": "关/开（硬开关，开档可能不稳定）"},
    "minicpm4.1-8b": {"path": "/mnt/workspace/gguf/minicpm41/MiniCPM4.1-8B-Q4_K_M.gguf",
                      "label": "MiniCPM4.1-8B · 约6-9 tok/s · 高效",
                      "think_levels": ["off", "high"],
                      "think_note": "关/开（关走模板硬关闭）"},
}

DEFAULT_THINK_LEVELS = ("off", "high")


def think_levels_for(name):
    return list((MODELS.get(name) or {}).get("think_levels") or DEFAULT_THINK_LEVELS)


def normalize_think(name, level):
    levels = think_levels_for(name)
    if level in levels:
        return level
    if "high" in levels:
        return "high"
    return levels[-1] if levels else "high"

HARD_OFF_MODELS = {"minicpm4.1-8b"}


def _hard_off_prompt(llm, msgs):
    try:
        tpl = (getattr(llm, "metadata", {}) or {}).get("tokenizer.chat_template")
        if isinstance(tpl, bytes):
            tpl = tpl.decode("utf-8", "replace")
        if not tpl:
            return None
        clean = []
        for m in msgs:
            if m.get("role") == "system" and str(m.get("content") or "").startswith("/no_think"):
                rest = str(m.get("content") or "")[len("/no_think"):].strip()
                if rest:
                    clean.append({"role": "system", "content": rest})
            else:
                clean.append(m)
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        fmt = Jinja2ChatFormatter(tpl, "<|im_end|>", "<|im_start|>", True)
        r = fmt(messages=clean, enable_thinking=False)
        return getattr(r, "prompt", None) or getattr(r, "text", None)
    except Exception:
        return None


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

UI_SYSTEM_PROMPT = """你可以调用前端工具，把回答补充成交互视图。工具只在条件满足时调用，最多 2 个；普通问答不要调用。工具必须放在正文之后，正文仍然完整回答，不要在正文重复 Markdown 表格。
调用格式：
<tool name="render_chart">{"title":"...","labels":["A","B"],"values":[10,20],"unit":"..."}</tool>
<tool name="render_compare">{"title":"...","columns":["项目","A","B"],"rows":[["价格","10","20"]]}</tool>
<tool name="render_calculator">{"mode":"loan","title":"月供估算","fields":[{"key":"amount","label":"贷款金额","value":1000000},{"key":"rate","label":"年利率 %","value":4.2},{"key":"years","label":"年限","value":30}],"unit":"元"}</tool>
<tool name="render_svg"><svg viewBox="0 0 400 240">...</svg></tool>
<tool name="render_flow">{"title":"...","steps":["第一步","第二步"]}</tool>
工具条件：
- render_chart：用户要图表/柱状/折线/趋势/可视化，或回答有明显数值序列；labels 与 values 等长，values 是数字。
- render_compare：用户要对比多个对象/参数，或回答适合用表格比较。
- render_calculator：用户要可调计算/贷款/分摊；贷款用 mode=loan，AA/分摊用 mode=split，其他可用 formula。
- render_svg：用户明确要 SVG/矢量图/示意图/画一个...svg；工具体内直接放完整 <svg>，不要 Markdown 代码围栏。
- render_flow：用户要步骤/流程。
用户说“随便画”时，用 render_svg 给一个简单示意，或用 render_chart 给标题注明“示例”的示例数据；不要反问风格。
JSON 必须严格合法；不要输出 HTML（render_svg 里的 SVG 除外）；不需要工具时不要输出 <tool>。也可以用函数调用形式（参数为 JSON）：render_chart(["A","B"],[10,20])，不要输出其他代码。"""

UI_PLANNER_PROMPT = """你是 UI 规格生成器。根据“问题”和“回答”，只输出一个 JSON 对象，不要输出解释、Markdown 代码围栏或 HTML。
可用类型与严格格式：
{"type":"compare","title":"...","columns":["项目","A","B"],"rows":[["价格","10 元","20 元"]]}
{"type":"chart","chart":"bar 或 line","title":"...","labels":["..."],"values":[1,2],"unit":"..."}
{"type":"calculator","mode":"loan","title":"月供估算","fields":[{"key":"amount","label":"贷款金额","value":1000000},{"key":"rate","label":"年利率 %","value":4.2},{"key":"years","label":"年限","value":30}],"unit":"元"}
{"type":"calculator","mode":"split","title":"账单分摊","fields":[{"key":"total","label":"总金额","value":300},{"key":"people","label":"人数","value":3},{"key":"tip","label":"小费 %","value":10}],"unit":"元"}
{"type":"calculator","title":"其他计算","fields":[{"key":"x","label":"数值","value":100}],"formula":"x*2","unit":"..."}
{"type":"flow","title":"...","steps":["第一步","第二步"]}
规则：
1. 只输出 JSON 对象本身，不要 ```、解释或 HTML。
2. labels 与 values 必须等长，values 必须是数字；数据只能来自回答，不能编造。
3. 对比/表格用 compare；趋势/柱状/折线用 chart；贷款用 mode=loan；AA/分摊用 mode=split；其他计算才用 formula。
4. 如果问题只是“随便画/画一个”且回答没有数据，给一个带示例数据和“示例”标题的 chart；无法生成有效规格时输出 {"type":"none"}。"""

def _ui_extract_json(text):
    t = str(text or "")
    i = t.rfind("</" + "think>")
    if i >= 0:
        t = t[i + 8:]
    t = t.replace("```ui", "").replace("```json", "").replace("```", "").strip()
    a = t.find("{")
    b = t.rfind("}")
    if a < 0 or b <= a:
        return None
    raw = t[a:b + 1]
    raws = [raw, raw.replace('\\"', '"')]
    for base_raw in raws:
        for candidate in (base_raw, re.sub(r",\s*([}\]])", r"\1", base_raw)):
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return None


def _ui_valid(spec):
    if not isinstance(spec, dict):
        return False
    t = spec.get("type")
    if t == "compare":
        cols, rows = spec.get("columns"), spec.get("rows")
        return (isinstance(cols, list) and len(cols) >= 2 and
                isinstance(rows, list) and len(rows) > 0 and
                all(isinstance(r, list) and len(r) >= 2 for r in rows))
    if t == "chart":
        labels, vals = spec.get("labels"), spec.get("values")
        if not (isinstance(labels, list) and isinstance(vals, list) and len(labels) == len(vals) and len(vals) >= 2):
            return False
        if spec.get("chart", "bar") not in ("bar", "line"):
            return False
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
    if t == "calculator":
        fields = spec.get("fields")
        if not isinstance(fields, list) or not fields:
            return False
        keys = {str(f.get("key") or "") for f in fields if isinstance(f, dict)}
        mode = spec.get("mode")
        if mode == "loan":
            return {"amount", "rate", "years"}.issubset(keys)
        if mode == "split":
            return {"total", "people", "tip"}.issubset(keys)
        return bool(str(spec.get("formula") or "").strip())
    if t == "flow":
        steps = spec.get("steps")
        return isinstance(steps, list) and len(steps) > 0
    return False



state = {"name": None, "llm": None, "loading": None,
         "load_started": None, "last_load_seconds": None,
         "load_progress": None, "load_rss_mb": None}
load_lock = threading.Lock()
gen_lock = threading.Lock()
q_lock = threading.Lock()
q_state = {"waiting": 0}
MAX_QUEUE = 8
CONV_FILE = "/mnt/workspace/chat/conversations.json"
conv_lock = threading.Lock()
stop_lock = threading.Lock()
STOP_EVENTS = {}
JOBS = {}
jobs_lock = threading.Lock()


def _load_convs_db():
    try:
        with open(CONV_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("conversations"), list):
            return data["conversations"]
    except Exception:
        pass
    return []


def _save_convs_db(items):
    try:
        os.makedirs(os.path.dirname(CONV_FILE), exist_ok=True)
        tmp = CONV_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"conversations": items}, f, ensure_ascii=False)
        os.replace(tmp, CONV_FILE)
    except Exception:
        pass


def _upsert_conv_snapshot(conv):
    if not isinstance(conv, dict) or not conv.get("id"):
        return
    cid = str(conv.get("id"))
    with conv_lock:
        items = _load_convs_db()
        found = False
        for i, c in enumerate(items):
            if isinstance(c, dict) and str(c.get("id")) == cid:
                items[i] = conv
                found = True
                break
        if not found:
            items.append(conv)
        _save_convs_db(items)


def _update_conv_message(cid, mid, content, pending, extra=None):
    if not cid or not mid:
        return
    with conv_lock:
        items = _load_convs_db()
        for c in items:
            if not isinstance(c, dict) or str(c.get("id")) != str(cid):
                continue
            msgs = c.get("messages")
            if not isinstance(msgs, list):
                msgs = []
                c["messages"] = msgs
            target = None
            for m in msgs:
                if isinstance(m, dict) and str(m.get("mid")) == str(mid):
                    target = m
                    break
            if target is None:
                target = {"role": "assistant", "mid": mid, "content": ""}
                msgs.append(target)
            target["content"] = content
            target["pending"] = bool(pending)
            if extra:
                for k, v in extra.items():
                    if v is None:
                        target.pop(k, None)
                    else:
                        target[k] = v
            c["updatedAt"] = int(time.time() * 1000)
            _save_convs_db(items)
            return


def _clean_reply_text(text):
    t = str(text or "")
    ct = t.find("</" + "think>")
    if ct >= 0:
        t = t[ct + 8:]
    t = t.replace("<" + "think>", "").strip()
    return t




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
.side-scroll{display:flex;flex-direction:column;gap:6px;flex:1;min-height:0}
.threads{display:flex;flex-direction:column;gap:10px;min-height:0;overflow:auto;flex:1}
.t-group{display:flex;flex-direction:column;gap:4px}
.t-group-label{font:600 10px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--dim);padding:0 8px;margin:2px 0 3px}
.thread{display:flex;align-items:center;gap:8px;padding:7px 8px 7px 10px;border-radius:8px;background:transparent;border:1px solid transparent;color:var(--text);font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:pointer}
.thread:hover{background:var(--card);border-color:var(--border)}
.thread .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);flex:0 0 auto}
.thread.active{background:var(--card2);border-color:var(--border2)}
.thread .t-title{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.thread .t-act{display:none;border:0;background:transparent;color:var(--dim);cursor:pointer;font-size:12px;line-height:1;padding:0 1px;flex:0 0 auto}
.thread:hover .t-act{display:block}
.thread.pinned .t-pin{display:block;color:var(--accent)}
.thread .t-act:hover{color:var(--text)}
.thread .t-del:hover{color:var(--danger)}
.thread input.t-rename{flex:1;min-width:0;background:var(--card2);border:1px solid var(--border2);border-radius:5px;color:var(--text);font-family:var(--sans);font-size:12.5px;padding:1px 4px;outline:none}
.t-empty{font-size:11.5px;color:var(--dim);padding:8px 9px}
.side-search{position:relative}
.side-search svg{position:absolute;left:9px;top:50%;transform:translateY(-50%);width:13px;height:13px;color:var(--dim);pointer-events:none}
#convSearch{width:100%;min-width:0;box-sizing:border-box;background:var(--card);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:var(--sans);font-size:12px;padding:7px 9px 7px 28px;outline:none}
#convSearch:focus{border-color:var(--border2);background:var(--card2)}
#convSearch::placeholder{color:var(--dim)}
#convCount{color:var(--dim);font-weight:400;letter-spacing:0;margin-left:4px}
@media(hover:none){.thread .t-act{display:none}.thread.active .t-act{display:block;opacity:.55}.thread.pinned .t-pin{display:block;opacity:.85}}
@media(min-width:901px){#menuBtn{display:flex}body.side-hidden #sidebar{display:none}}
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
#menuBtn{display:flex;align-items:center;justify-content:center;width:32px;height:32px;border:1px solid var(--border);background:var(--card);color:var(--muted);border-radius:8px;cursor:pointer}
.ctl{appearance:none;background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:6px 9px;font-size:12.5px;outline:none;cursor:pointer;transition:border-color .15s,background .15s}
.ctl:hover{border-color:var(--border2);background:var(--card2)}
.ctl:focus{border-color:#4a4f58}
#thinkLevel:disabled{opacity:.55;cursor:not-allowed}
#loadBtn{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:6px 11px;font-size:12.5px;cursor:pointer}
#loadBtn:hover:not(:disabled){background:var(--card2);border-color:var(--border2)}
#loadBtn:disabled{opacity:.45;cursor:default}
#model{max-width:230px}
#ctxInfo{font-size:11.5px;color:var(--dim);white-space:nowrap;margin-left:2px}
#status{margin-left:auto;display:flex;align-items:center;gap:7px;font-size:11.5px;color:var(--muted);max-width:42%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#status::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);flex:0 0 auto}
#progWrap{display:none;align-items:center;gap:10px;padding:7px 22px;border-bottom:1px solid var(--border);background:var(--panel)}
#progTrack{flex:0 0 180px;height:4px;background:#232529;border-radius:3px;overflow:hidden}
#progBar{height:100%;width:0;background:linear-gradient(90deg,#6f9f4a,var(--accent));border-radius:3px;transition:width .6s ease}
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
.think{color:var(--dim);font:12.5px/1.65 var(--mono);border-left:2px solid var(--border2);padding:8px 0 8px 12px;margin:0 0 12px}
.think>summary{cursor:pointer;list-style:none;color:var(--muted);user-select:none;display:flex;align-items:center;gap:6px;white-space:nowrap}
.think>summary::-webkit-details-marker{display:none}
.think>summary::before{content:'▸';display:inline-block;color:var(--dim);transition:transform .15s}
.think[open]>summary::before{transform:rotate(90deg)}
.think-body{white-space:pre-wrap;max-height:260px;overflow:auto;margin-top:8px;padding-right:4px}
.iui{margin:14px 0 2px;border:1px solid var(--border);border-radius:12px;background:var(--card);overflow:hidden;max-width:760px}
.iui-head{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--border);min-width:0}
.iui-badge{font:600 9px/1 var(--mono);letter-spacing:.08em;color:var(--accent);background:var(--accent-soft);border:1px solid rgba(158,206,106,.25);padding:3px 5px;border-radius:4px;white-space:nowrap}
.iui-title{font-weight:600;color:var(--text);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.iui-body{padding:12px}
.iui-tablewrap{overflow:auto}
.iui table{width:100%;border-collapse:collapse;font-size:12.5px}
.iui th,.iui td{padding:8px 9px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top;white-space:nowrap}
.iui th{font:600 10px/1.4 var(--mono);color:var(--dim);letter-spacing:.04em;background:var(--bg)}
.iui tbody tr:last-child td{border-bottom:0}
.iui tbody tr:hover td{background:rgba(255,255,255,.025)}
.iui td.iui-k{color:var(--muted)}
.iui-svg{display:block;width:100%;height:auto;overflow:visible}
.iui-svg text{fill:var(--dim);font:10px var(--mono)}
.iui-svg .iui-val{fill:var(--muted);font-size:10px}
.iui-svg .iui-bar{fill:rgba(158,206,106,.72)}
.iui-svg .iui-line{fill:none;stroke:var(--accent);stroke-width:2}
.iui-svg .iui-dot{fill:var(--accent);stroke:var(--card);stroke-width:2}
.iui-field{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:8px 0}
.iui-field label{font-size:12px;color:var(--muted)}
.iui-field input{width:132px;background:var(--bg);border:1px solid var(--border);border-radius:7px;color:var(--text);padding:6px 8px;font:12px var(--mono);outline:none}
.iui-field input:focus{border-color:var(--border2)}
.iui-result{margin-top:10px;padding:10px 12px;border-radius:8px;background:var(--bg);font:600 17px/1.3 var(--mono);color:var(--accent)}
.iui-note{font-size:10.5px;color:var(--dim);margin-top:6px}
.iui-flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.iui-step{padding:7px 10px;border:1px solid var(--border);border-radius:8px;background:var(--bg);font-size:12px;color:var(--text)}
.iui-arrow{color:var(--dim);font-size:12px}
.iui-pending{display:inline-flex;align-items:center;gap:6px;margin-top:10px;padding:5px 8px;border:1px solid var(--border);border-radius:6px;color:var(--dim);font:10.5px var(--mono)}
.iui-pending::before{content:'✦';color:var(--accent)}
@media(max-width:600px){.iui-body{padding:10px}.iui th,.iui td{padding:7px}.iui-field input{width:112px}}

.md{white-space:normal}

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
.svg-card{margin:14px 0;border:1px solid var(--border);border-radius:12px;background:var(--card);overflow:hidden;max-width:760px}
.svg-bar{padding:8px 12px;border-bottom:1px solid var(--border);font:600 9px/1.2 var(--mono);letter-spacing:.08em;color:var(--accent)}
.svg-wrap{padding:12px;overflow:auto;background:#fff}
.svg-wrap svg{display:block;max-width:100%;height:auto;margin:0 auto}
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
#send:active{transform:scale(.96)}#send:disabled{opacity:.45;cursor:default;transform:none}#send.stop{background:var(--danger);color:#0b0c0e}
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
  <div class="side-search">
   <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
   <input id="convSearch" type="search" placeholder="搜索会话" autocomplete="off" aria-label="搜索会话" />
  </div>
  <div class="side-scroll">
   <div class="side-label">会话<span id="convCount"></span></div>
   <div class="threads" id="threads"></div>
   <div class="side-note">每个会话独立上下文 · 本地保存</div>
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
   <select id="thinkLevel" class="ctl" title="思考开关"><option value="off">思考：关</option><option value="high" selected>思考：开</option></select>
   <select id="maxTok" class="ctl" title="最大输出长度"><option value="1024">1024 tokens</option><option value="2048">2048 tokens</option><option value="4096" selected>4096 tokens</option><option value="8192">8192 tokens</option></select>
   <span id="ctxInfo" title="上下文占用估算（由服务端预算裁剪后计算）">上下文 ≈ —</span>
   <span id="status">连接中…</span>
  </div>
  <div id="progWrap"><div id="progTrack"><div id="progBar"></div></div><span id="progText"></span></div>
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
 <div class="field"><label>默认思考强度</label><select id="setThink"><option value="off">关</option><option value="high">开</option></select></div>
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
const CONV_KEY='muse_convs_v1';
let convs=[],currentId=null,lastUserPrompt='',pendingStreams={},sending=false,currentAbort=null,convSyncTimer=null,syncingPaused=false,currentJobId=null;
function uid(){return 'c'+Date.now().toString(36)+Math.random().toString(36).slice(2,7);}
function makeTitle(s){
 s=String(s||'');
 s=s.split(String.fromCharCode(10)).join(' ').split(String.fromCharCode(13)).join(' ').split(String.fromCharCode(9)).join(' ');
 s=s.replace(/ +/g,' ').trim();
 return s?s.slice(0,26):'新对话';
}
function currentConv(){for(var i=0;i<convs.length;i++){if(convs[i].id===currentId)return convs[i];}return null;}
function saveConvs(){
 var data={conversations:convs,currentId:currentId};
 try{localStorage.setItem(CONV_KEY,JSON.stringify(data));}catch(e){
  var cur=currentConv();
  var rest=convs.filter(function(c){return c.id!==currentId;}).slice(0,9);
  convs=(cur?[cur]:[]).concat(rest);
  try{localStorage.setItem(CONV_KEY,JSON.stringify({conversations:convs,currentId:currentId}));}catch(e2){}
 }
 if(!syncingPaused)scheduleConvSync();
}
function scheduleConvSync(){
 if(syncingPaused)return;
 clearTimeout(convSyncTimer);
 convSyncTimer=setTimeout(function(){
  fetch('api/convs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversations:convs})}).catch(function(){});
 },400);
}
async function syncConvsNow(){
 try{await fetch('api/convs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversations:convs})});}catch(e){}
}
function saveHist(){var c=currentConv();if(!c)return;c.messages=history.slice(-80);c.updatedAt=Date.now();saveConvs();renderThreads();}
function findConv(id){for(var i=0;i<convs.length;i++){if(convs[i].id===id)return convs[i];}return null;}
function sortedConvs(){return convs.slice().sort(function(a,b){var pa=a.pinned?1:0,pb=b.pinned?1:0;if(pa!==pb)return pb-pa;return (b.updatedAt||0)-(a.updatedAt||0);});}
function convBucket(c){
 var ts=c.updatedAt||c.createdAt||0;
 var n=new Date();var start=new Date(n.getFullYear(),n.getMonth(),n.getDate()).getTime();
 if(ts>=start)return 0;
 if(ts>=start-86400000)return 1;
 if(ts>=start-6*86400000)return 2;
 return 3;
}
function convMatches(c,q){
 if(!q)return true;
 if(String(c.title||'').toLowerCase().indexOf(q)>=0)return true;
 var ms=c.messages||[];
 for(var i=0;i<ms.length;i++){if(String(ms[i].content||'').toLowerCase().indexOf(q)>=0)return true;}
 return false;
}
function togglePin(id){var c=findConv(id);if(!c)return;c.pinned=!c.pinned;saveConvs();renderThreads();}
function startRename(id,titleEl){
 var c=findConv(id);if(!c||!titleEl)return;
 var old=String(c.title||'新对话');
 var inp=document.createElement('input');inp.className='t-rename';inp.value=old;inp.maxLength=60;
 titleEl.replaceWith(inp);inp.focus();inp.select();
 var done=false;
 function finish(save){
  if(done)return;done=true;
  var v=inp.value.trim();
  if(save&&v){c.title=v.slice(0,60);saveConvs();}
  renderThreads();
 }
 inp.addEventListener('click',function(ev){ev.stopPropagation();});
 inp.addEventListener('dblclick',function(ev){ev.stopPropagation();});
 inp.addEventListener('keydown',function(ev){
  if(ev.key==='Enter'){ev.preventDefault();finish(true);}
  else if(ev.key==='Escape'){ev.preventDefault();finish(false);}
 });
 inp.addEventListener('blur',function(){finish(true);});
}
function threadRow(c){
 var d=document.createElement('div');
 d.className='thread'+(c.id===currentId?' active':'')+(c.pinned?' pinned':'');
 d.setAttribute('data-id',c.id);
 d.title=(c.updatedAt||c.createdAt)?new Date(c.updatedAt||c.createdAt).toLocaleString():'';
 var dot=document.createElement('span');dot.className='dot';
 var t=document.createElement('span');t.className='t-title';t.textContent=c.title||'新对话';
 function act(cls,txt,tip,fn){var b=document.createElement('button');b.className='t-act '+cls;b.textContent=txt;b.title=tip;b.onclick=function(ev){ev.stopPropagation();fn();};return b;}
 var pin=act('t-pin',c.pinned?'★':'☆',c.pinned?'取消置顶':'置顶',function(){togglePin(c.id);});
 var ren=act('t-ren','✎','重命名',function(){startRename(c.id,d.querySelector('.t-title'));});
 var del=act('t-del','×','删除会话',function(){deleteConv(c.id);});
 t.ondblclick=function(ev){ev.stopPropagation();startRename(c.id,t);};
 d.appendChild(dot);d.appendChild(t);d.appendChild(pin);d.appendChild(ren);d.appendChild(del);
 d.onclick=function(){switchConv(c.id);};
 return d;
}
function addThreadGroup(box,label,items){
 if(!items.length)return;
 var g=document.createElement('div');g.className='t-group';
 var l=document.createElement('div');l.className='t-group-label';l.textContent=label;
 g.appendChild(l);
 for(var i=0;i<items.length;i++)g.appendChild(threadRow(items[i]));
 box.appendChild(g);
}
function renderThreads(){
 var box=$('threads');if(!box)return;
 box.innerHTML='';
 var cnt=$('convCount');if(cnt)cnt.textContent=convs.length?String(convs.length):'';
 var input=$('convSearch');var q=input?input.value.trim().toLowerCase():'';
 var list=q?sortedConvs().filter(function(c){return convMatches(c,q);}):sortedConvs();
 if(!list.length){var e=document.createElement('div');e.className='t-empty';e.textContent=q?'没有匹配的会话':'还没有会话';box.appendChild(e);return;}
 if(q){addThreadGroup(box,'搜索结果 · '+list.length,list);return;}
 addThreadGroup(box,'置顶',list.filter(function(c){return c.pinned;}));
 var gs=[[],[],[],[]];
 for(var i=0;i<list.length;i++){if(list[i].pinned)continue;gs[convBucket(list[i])].push(list[i]);}
 var labels=['今天','昨天','最近 7 天','更早'];
 for(var j=0;j<4;j++)addThreadGroup(box,labels[j],gs[j]);
}
function appendToConv(id,msg){
 var c=null;for(var i=0;i<convs.length;i++){if(convs[i].id===id){c=convs[i];break;}}
 if(!c)return;
 c.messages=c.messages||[];
 c.messages.push(msg);
 c.messages=c.messages.slice(-80);
 c.updatedAt=Date.now();
 if((!c.title||c.title==='新对话')&&msg.role==='user'){c.title=makeTitle(msg.content);}
 saveConvs();
 if(currentId===id){history=c.messages.slice();}
 renderThreads();
}
function removeLastUser(id){
 var c=null;for(var i=0;i<convs.length;i++){if(convs[i].id===id){c=convs[i];break;}}
 if(!c)return;
 if(c.messages&&c.messages.length)c.messages.pop();
 c.updatedAt=Date.now();saveConvs();
 if(currentId===id){history=c.messages.slice();}
 renderThreads();
}
function startAssistant(convId){
 var c=findConv(convId);if(!c)return null;
 var mid='m'+Date.now().toString(36)+Math.random().toString(36).slice(2,6);
 c.messages=c.messages||[];
 c.messages.push({role:'assistant',content:'',mid:mid,pending:true});
 c.messages=c.messages.slice(-80);c.updatedAt=Date.now();
 saveConvs();if(currentId===convId)history=c.messages.slice();
 renderThreads();return mid;
}
function updateAssistant(convId,mid,content,extra){
 var c=findConv(convId);if(!c)return;
 var ms=c.messages||[];
 for(var i=0;i<ms.length;i++){
  if(ms[i].mid===mid){
   ms[i].content=String(content||'');
   if(extra)for(var k in extra)ms[i][k]=extra[k];
   c.updatedAt=Date.now();saveConvs();
   if(currentId===convId)history=c.messages.slice();
   return;
  }
 }
}
function removeAssistant(convId,mid){
 var c=findConv(convId);if(!c)return;
 c.messages=(c.messages||[]).filter(function(m){return m.mid!==mid;});
 c.updatedAt=Date.now();saveConvs();
 if(currentId===convId)history=c.messages.slice();
 renderThreads();
}
function newConv(){
 var c={id:uid(),title:'新对话',createdAt:Date.now(),updatedAt:Date.now(),messages:[],pinned:false};
 convs.unshift(c);currentId=c.id;history=[];
 var cs=$('convSearch');if(cs)cs.value='';
 saveConvs();renderThreads();
 $('log').innerHTML='';ctxMeta=null;$('ctxInfo').textContent='上下文 ≈ —';
 var p=$('prompt');if(p)p.focus();
}
function switchConv(id){
 if(id===currentId)return;
 var c=null;for(var i=0;i<convs.length;i++){if(convs[i].id===id){c=convs[i];break;}}
 if(!c)return;
 currentId=id;history=(c.messages||[]).slice();saveConvs();
 $('log').innerHTML='';ctxMeta=null;$('ctxInfo').textContent='上下文 ≈ —';
 renderThreads();restoreHist();
}
function deleteConv(id){
 if(!confirm('删除这个会话？'))return;
 if(currentId!==id){
  convs=convs.filter(function(c){return c.id!==id;});
  saveConvs();renderThreads();return;
 }
 convs=convs.filter(function(c){return c.id!==id;});
 if(!convs.length){newConv();return;}
 var next=sortedConvs()[0];currentId=next.id;history=(next.messages||[]).slice();
 saveConvs();renderThreads();
 $('log').innerHTML='';ctxMeta=null;$('ctxInfo').textContent='上下文 ≈ —';
 restoreHist();
}
async function initConvs(){
 syncingPaused=true;
 var server=null,localCurrent=null,localConvs=[];
 try{
  var raw=localStorage.getItem(CONV_KEY);
  if(raw){var d=JSON.parse(raw);localCurrent=d.currentId||null;localConvs=Array.isArray(d.conversations)?d.conversations:[];}
 }catch(e){}
 try{
  var r=await fetch('api/convs',{cache:'no-store'});
  if(r.ok){var j=await r.json();if(Array.isArray(j.conversations)&&j.conversations.length)server=j.conversations;}
 }catch(e){}
 if(server)convs=server;
 else{
  convs=localConvs;
  if(!convs.length){
   var old=[];try{old=JSON.parse(localStorage.getItem('muse_hist')||'[]');}catch(e){}
   if(Array.isArray(old)&&old.length){
    var u=null;for(var i=0;i<old.length;i++){if(old[i]&&old[i].role==='user'){u=old[i];break;}}
    convs=[{id:uid(),title:makeTitle(u?u.content:'历史对话'),createdAt:Date.now(),updatedAt:Date.now(),messages:old.slice(-80)}];
    localCurrent=convs[0].id;
   }
  }
 }
 for(var ci=0;ci<convs.length;ci++){if(!Array.isArray(convs[ci].messages))convs[ci].messages=[];convs[ci].pinned=!!convs[ci].pinned;}
 if(!convs.length){syncingPaused=false;newConv();return;}
 currentId=localCurrent;
 var found=false;for(var i=0;i<convs.length;i++){if(convs[i].id===currentId){found=true;break;}}
 if(!found)currentId=convs[0].id;
 var c=currentConv();history=(c.messages||[]).slice();
 renderThreads();restoreHist();
 syncingPaused=false;
 if(!server)await syncConvsNow();
}
function restoreHist(){for(const m of history){
 if(m.role==='user')addMsg('me',m.content);
 else if(m.role==='assistant'){
  if((m.pending||m.interrupted)&&!String(m.content||'').trim())continue;
  const el=addMsg('ai','');el.setAttribute('data-mid',m.mid||'');renderFinal(el,m.content,m.ui||[],m.uiInline||[]);
  if(m.pending||m.interrupted){
   const n=document.createElement('div');n.className='iui-pending';
   n.textContent=m.pending?(m.job_id?'后台生成中 · 稍后刷新':'生成中断 · 刷新前已保存部分内容'):'生成中断';
   el.appendChild(n);
  }
 }
}
scrollBottom();
setTimeout(function(){if(!sending)resumePending();},80);
}
function isAtBottom(){var el=$('log');if(!el)return true;return el.scrollHeight-el.scrollTop-el.clientHeight<48;}
function scrollBottom(){var el=$('log');if(!el)return;var old=el.style.scrollBehavior;el.style.scrollBehavior='auto';el.scrollTop=el.scrollHeight;el.style.scrollBehavior=old;}
function addMsg(kind,text){var stick=isAtBottom();const d=document.createElement('div');d.className='msg '+kind;d.textContent=text;$('log').appendChild(d);if(stick)scrollBottom();return d;}
async function jget(u){const r=await fetch(u,{cache:'no-store'});return r.json();}
async function jpost(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
function setStatus(t){$('status').textContent=t;}
function setProg(show,pct,txt){$('progWrap').style.display=show?'flex':'none';$('progBar').style.width=(pct||0)+'%';$('progText').textContent=txt||'';}
let refreshing=false,prevLoading=false,hideT=null,progHoldUntil=0;
function applyThinkLevels(){
 const m=models.find(function(x){return x.id===$('model').value;});
 const levels=(m&&m.think_levels)||['off','low','medium','high'];
 const note=(m&&m.think_note)||'';
 const sel=$('thinkLevel');
 sel.disabled=(levels.length<=1);
 for(const o of sel.options){
  if(!o.dataset.base)o.dataset.base=o.textContent;
  if(levels.indexOf(o.value)<0){o.disabled=true;o.textContent=o.dataset.base+'（不可用）';}
  else{o.disabled=false;o.textContent=o.dataset.base+((o.value==='low'||o.value==='medium')?'（软引导）':'');}
 }
 sel.title=note?('思考强度 · '+note):'思考强度';
 if(levels.indexOf(sel.value)<0){sel.value=levels.indexOf('high')>=0?'high':levels[0];}
}
async function refresh(){
 if(refreshing)return;
 refreshing=true;
 try{
  const s=await jget('api/status');
  models=s.models||[];loaded=s.loaded;loading=s.loading;
  const sel=$('model');
  if(sel.options.length===0){for(const m of models){const o=document.createElement('option');o.value=m.id;o.textContent=m.label;sel.appendChild(o);}}if(loaded&&!sel.dataset.init){sel.value=loaded;sel.dataset.init='1';}applyThinkLevels();
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
$('model').onchange=function(){progHoldUntil=0;setProg(false,0,'');ctxMeta=null;$('ctxInfo').textContent='上下文 ≈ —';applyThinkLevels();};
$('loadBtn').onclick=()=>{const m=$('model').value;ensure(m).then(()=>addMsg('sys','模型已就绪')).catch(err=>addMsg('sys','加载失败：'+err.message));};
function thinkBlock(text,label,open,final){
 var d=document.createElement('details');d.className='think';if(open)d.open=true;
 var s=document.createElement('summary');s.textContent=label;
 var b=document.createElement('div');b.className='think-body';b.textContent=text;
 d.appendChild(s);d.appendChild(b);
 if(final){d.addEventListener('toggle',function(){s.textContent='💭 思考 · '+text.length+' 字'+(d.open?'（点击收起）':'（点击展开）');});}
 return d;
}
function uiPlainPrompt(s){
 return String(s||'')
  .replace(/[，,。.]?\s*(请)?(用|使用)\s*智能(界面|视图|UI)\s*(来)?\s*(展示|显示|呈现|给出)?/gi,'')
  .replace(/[，,。.]?\s*智能(界面|视图)\s*(展示|显示|呈现|给出)/gi,'')
  .replace(/\s{2,}/g,' ')
  .trim();
}
function plainForModel(m){
 var c=(m.role==='user')?(uiPlainPrompt(m.content)||m.content):String(m.content||'');
 return c.replace(/@@IUI[0-9]+@@/g,'').trim();
}
function uiNode(tag,cls,text){
 var e=document.createElement(tag);
 if(cls)e.className=cls;
 if(text!==undefined&&text!==null)e.textContent=String(text);
 return e;
}
function uiFmt(n){
 var v=Number(n);
 if(!isFinite(v))return '—';
 var a=Math.abs(v);
 if(a>=1000)return v.toLocaleString('zh-CN',{maximumFractionDigits:2});
 return String(Math.round(v*100)/100);
}
function splitUI(s){
 var text=String(s||''),specs=[];
 text=text.replace(/```ui[ \t]*\r?\n?([\s\S]*?)```/gi,function(_,body){
  var t=String(body).trim();
  try{var j=JSON.parse(t);if(j&&typeof j==='object')specs.push(j);}catch(e){
   try{var j2=JSON.parse(t.replace(/,\s*([}\]])/g,'$1'));if(j2&&typeof j2==='object')specs.push(j2);}catch(e2){}
  }
  return '';
 });
 var pending=/```ui[\s\S]*$/i.test(text);
 if(pending)text=text.replace(/```ui[\s\S]*$/i,'');
 return {text:text.trim(),specs:specs,pending:pending};
}
function mdCell(s){return String(s||'').replace(/\*\*/g,'').replace(/`/g,'').trim();}
function mdCells(line){
 var s=String(line).trim();
 if(s.charAt(0)==='|')s=s.slice(1);
 if(s.charAt(s.length-1)==='|')s=s.slice(0,-1);
 return s.split('|').map(mdCell);
}
function numCell(s){
 var m=String(s||'').replace(/[,\s]/g,'').match(/-?\d+(?:\.\d+)?/);
 return m?parseFloat(m[0]):null;
}
function wantsChart(){
 var p=(typeof lastUserPrompt!=='undefined')?lastUserPrompt:'';
 return /图表|柱状|折线|趋势|曲线|可视化|chart|graph/i.test(p);
}
function extractMdTables(text){
 var lines=String(text||'').split('\n'),out=[],inline=[],seen={},i=0;
 function isSep(l){return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(l);}
 while(i<lines.length){
  if(lines[i].indexOf('|')>=0&&i+1<lines.length&&isSep(lines[i+1])){
   var cols=mdCells(lines[i]),rows=[],j=i+2;
   while(j<lines.length&&lines[j].trim()&&lines[j].indexOf('|')>=0){rows.push(mdCells(lines[j]));j++;}
   if(cols.length>=2&&rows.length){
    var title='数据对比',titleIdx=-1;
    for(var k=out.length-1;k>=0;k--){var l=out[k].trim();if(!l)continue;if(/^#{1,6}\s+/.test(l)){title=mdCell(l.replace(/^#{1,6}\s+/,''));titleIdx=k;}break;}
    if(titleIdx>=0)out[titleIdx]='';
    var spec={type:'compare',title:title,columns:cols,rows:rows.slice(0,30)};
    if(wantsChart()&&cols.length===2&&rows.length>=2){
     var labels=[],vals=[];
     for(var rr=0;rr<rows.length;rr++){var vv=numCell(rows[rr][1]);if(vv!==null){labels.push(mdCell(rows[rr][0]));vals.push(vv);}}
     if(vals.length>=2)spec={type:'chart',chart:'bar',title:title,labels:labels,values:vals};
    }
    var sk=JSON.stringify(spec);
    if(!seen[sk]){seen[sk]=1;out.push('@@IUI'+inline.length+'@@');inline.push(spec);}
    i=j;continue;
   }
  }
  out.push(lines[i]);i++;
 }
 return {text:out.join('\n').replace(/\n{3,}/g,'\n\n').trim(),inline:inline};
}
function extractChartFromText(text){
 var s=String(text||'');
 var parts=s.split("```"),sources=[];
 for(var pi=1;pi<parts.length;pi+=2)sources.push(parts[pi]);
 if(!sources.length)return null;
 for(var bi=0;bi<sources.length;bi++){
  var lines=sources[bi].split(String.fromCharCode(10)),labels=[],vals=[];
  for(var li=0;li<lines.length;li++){
   var line=lines[li].split("**").join("").split("`").join("").trim();
   if(!line)continue;
   var cp=line.indexOf(":");if(cp<0)cp=line.indexOf("：");
   if(cp<0)continue;
   var label=line.slice(0,cp).trim().replace(/^[-*•]/,"").trim();
   var rest=line.slice(cp+1).trim();
   var val=parseFloat(rest);
   if(!label||label.length>40||!isFinite(val))continue;
   labels.push(label);vals.push(val);
  }
  if(labels.length>=2)return {type:"chart",chart:"bar",title:"数据图表",labels:labels,values:vals};
 }
 return null;
}
function demoChartSpec(){
 return {type:"chart",chart:"bar",title:"示例图表（演示数据）",labels:["A","B","C","D"],values:[12,18,15,24]};
}
function prepareUI(text){
 var p=splitUI(text);
 if(!p.specs.length){var a=extractMdTables(p.text);p.text=a.text;p.inline=a.inline;}
 return p;
}
function replaceUIToken(el,token,node){
 if(!el)return false;
 var walker=document.createTreeWalker(el,NodeFilter.SHOW_TEXT,null,false);
 while(walker.nextNode()){
  var n=walker.currentNode,pos=n.nodeValue.indexOf(token);
  if(pos<0)continue;
  var parent=n.parentNode;
  if(parent&&parent.tagName==='P'&&parent.textContent.trim()===token){parent.parentNode.replaceChild(node,parent);}
  else{
   var frag=document.createDocumentFragment();
   var before=n.nodeValue.slice(0,pos),after=n.nodeValue.slice(pos+token.length);
   if(before)frag.appendChild(document.createTextNode(before));
   frag.appendChild(node);
   if(after)frag.appendChild(document.createTextNode(after));
   parent.replaceChild(frag,n);
  }
  return true;
 }
 return false;
}
function mountInlineUI(el,specs){
 if(!Array.isArray(specs))return;
 for(var i=0;i<specs.length;i++){
  var n=renderUI(specs[i]);if(!n)continue;
  if(!replaceUIToken(el,'@@IUI'+i+'@@',n))el.appendChild(n);
 }
}
function uiCalc(formula,vars){
 var s=String(formula||'').replace(/\s+/g,'');
 var i=0;
 function peek(){return s.charAt(i);}
 function parseExpr(){
  var v=parseTerm();
  while(peek()==='+'||peek()==='-'){
   var op=s.charAt(i++);
   var r=parseTerm();
   v=op==='+'?v+r:v-r;
  }
  return v;
 }
 function parseTerm(){
  var v=parsePower();
  while(peek()==='*'||peek()==='/'||peek()==='%'){
   var op=s.charAt(i++);
   var r=parsePower();
   if(op==='*')v*=r;
   else if(op==='/')v/=r;
   else v%=r;
  }
  return v;
 }
 function parsePower(){
  var v=parseUnary();
  if(peek()==='^'){i++;var r=parsePower();v=Math.pow(v,r);}
  return v;
 }
 function parseUnary(){
  if(peek()==='+'){i++;return parseUnary();}
  if(peek()==='-'){i++;return -parseUnary();}
  return parsePrimary();
 }
 function parsePrimary(){
  if(peek()==='('){
   i++;
   var v=parseExpr();
   if(peek()!==')')throw new Error('paren');
   i++;
   return v;
  }
  var m=/^\d+(?:\.\d+)?/.exec(s.slice(i));
  if(m){i+=m[0].length;return parseFloat(m[0]);}
  var n=/^[A-Za-z_][A-Za-z0-9_]*/.exec(s.slice(i));
  if(n){
   i+=n[0].length;
   if(Object.prototype.hasOwnProperty.call(vars,n[0]))return Number(vars[n[0]])||0;
   throw new Error('var');
  }
  throw new Error('token');
 }
 var v=parseExpr();
 if(i!==s.length||!isFinite(v))throw new Error('parseExpr');
 return v;
}
function uiShell(spec,body){
 var wrap=uiNode('div','iui');
 var head=uiNode('div','iui-head');
 head.appendChild(uiNode('span','iui-badge','✦ 智能视图'));
 if(spec.title)head.appendChild(uiNode('span','iui-title',spec.title));
 wrap.appendChild(head);
 var b=uiNode('div','iui-body');
 b.appendChild(body);
 wrap.appendChild(b);
 return wrap;
}
function uiCompare(spec){
 var cols=Array.isArray(spec.columns)?spec.columns.slice(0,8):[];
 var rows=Array.isArray(spec.rows)?spec.rows.slice(0,30):[];
 if(!cols.length||!rows.length)return null;
 var table=document.createElement('table');
 var thead=document.createElement('thead');
 var htr=document.createElement('tr');
 for(var i=0;i<cols.length;i++)htr.appendChild(uiNode('th','',cols[i]));
 thead.appendChild(htr);table.appendChild(thead);
 var tb=document.createElement('tbody');
 for(var r=0;r<rows.length;r++){
  var tr=document.createElement('tr');
  var cells=Array.isArray(rows[r])?rows[r]:[];
  for(var c=0;c<cells.length;c++){
   var td=uiNode('td',c===0?'iui-k':'',cells[c]);
   tr.appendChild(td);
  }
  tb.appendChild(tr);
 }
 table.appendChild(tb);
 var wrap=uiNode('div','iui-tablewrap');
 wrap.appendChild(table);
 return uiShell(spec,wrap);
}
function uiSvg(tag,attrs){
 var e=document.createElementNS('http://www.w3.org/2000/svg',tag);
 for(var k in attrs)if(Object.prototype.hasOwnProperty.call(attrs,k))e.setAttribute(k,attrs[k]);
 return e;
}
function uiChart(spec){
 var labels=Array.isArray(spec.labels)?spec.labels.map(String):[];
 var vals=Array.isArray(spec.values)?spec.values.map(Number):[];
 if(!vals.length||labels.length!==vals.length||vals.length>40)return null;
 for(var q=0;q<vals.length;q++)if(!isFinite(vals[q]))return null;
 var unit=spec.unit?String(spec.unit):'';
 var kind=String(spec.chart||'bar').toLowerCase()==='line'?'line':'bar';
 var W=640,H=260,pl=46,pr=18,pt=18,pb=38;
 var plotW=W-pl-pr,plotH=H-pt-pb;
 var max=Math.max.apply(null,vals.concat([0]));
 var min=Math.min.apply(null,vals.concat([0]));
 if(max===min)max=min+1;
 var x=function(idx){return vals.length===1?pl+plotW/2:pl+plotW*idx/(vals.length-1);};
 var y=function(v){return pt+plotH*(1-(v-min)/(max-min));};
 var svg=uiSvg('svg',{viewBox:'0 0 '+W+' '+H,class:'iui-svg'});
 for(var g=0;g<=4;g++){
  var gy=pt+plotH*g/4;
  svg.appendChild(uiSvg('line',{x1:pl,y1:gy,x2:W-pr,y2:gy,stroke:'#232529','stroke-width':1}));
  var txt=uiSvg('text',{x:pl-8,y:gy+3,'text-anchor':'end'});
  txt.textContent=uiFmt(max-(max-min)*g/4);
  svg.appendChild(txt);
 }
 for(var i=0;i<labels.length;i++){
  var lab=labels[i]||'';
  if(lab.length>10)lab=lab.slice(0,9)+'…';
  var lx=kind==='bar'?(pl+plotW*(i+0.5)/vals.length):x(i);
  var ltxt=uiSvg('text',{x:lx,y:H-14,'text-anchor':'middle'});
  ltxt.textContent=lab;
  svg.appendChild(ltxt);
 }
 if(kind==='bar'){
  var bw=Math.min(54,plotW/vals.length*0.58);
  for(var b=0;b<vals.length;b++){
   var bx=pl+plotW*(b+0.5)/vals.length-bw/2;
   var y0=y(0),yv=y(vals[b]);
   var top=Math.min(y0,yv),bh=Math.max(1,Math.abs(y0-yv));
   var rect=uiSvg('rect',{x:bx,y:top,width:bw,height:bh,rx:4,class:'iui-bar'});
   var tip=uiSvg('title',{});
   tip.textContent=(labels[b]||'')+'：'+uiFmt(vals[b])+unit;
   rect.appendChild(tip);svg.appendChild(rect);
   if(vals.length<=8){
    var vt=uiSvg('text',{x:bx+bw/2,y:top-5,'text-anchor':'middle',class:'iui-val'});
    vt.textContent=uiFmt(vals[b]);
    svg.appendChild(vt);
   }
  }
 }else{
  var pts=[];
  for(var p=0;p<vals.length;p++)pts.push([x(p),y(vals[p])]);
  var d=pts.map(function(pt,idx){return (idx?'L':'M')+pt[0].toFixed(1)+' '+pt[1].toFixed(1);}).join(' ');
  svg.appendChild(uiSvg('path',{d:d,class:'iui-line'}));
  for(var c=0;c<pts.length;c++){
   var dot=uiSvg('circle',{cx:pts[c][0],cy:pts[c][1],r:4,class:'iui-dot'});
   var dt=uiSvg('title',{});
   dt.textContent=(labels[c]||'')+'：'+uiFmt(vals[c])+unit;
   dot.appendChild(dt);svg.appendChild(dot);
  }
 }
 return uiShell(spec,svg);
}
function uiCalcWidget(spec){
 var fields=Array.isArray(spec.fields)?spec.fields.slice(0,8):[];
 var mode=String(spec.mode||'');
 var formula=String(spec.formula||'');
 if(mode==='loan')formula='amount*(rate/1200)/(1-(1+rate/1200)^(-years*12))';
 else if(mode==='split')formula='total*(1+tip/100)/people';
 var unit=spec.unit?String(spec.unit):'';
 var wrap=uiNode('div','');
 var inputs={};
 function update(){
  var vars={};
  for(var k in inputs)if(Object.prototype.hasOwnProperty.call(inputs,k))vars[k]=parseFloat(inputs[k].value)||0;
  try{out.textContent=uiFmt(uiCalc(formula,vars))+(unit?(' '+unit):'');}
  catch(e){out.textContent='无法计算';}
 }
 for(var i=0;i<fields.length;i++){
  var f=fields[i];
  if(!f||typeof f!=='object')continue;
  var key=String(f.key||'');
  if(!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key))continue;
  var row=uiNode('div','iui-field');
  row.appendChild(uiNode('label','',f.label||key));
  var inp=document.createElement('input');
  inp.type='number';inp.step='any';inp.value=(f.value!==undefined?f.value:0);
  inp.oninput=update;
  row.appendChild(inp);wrap.appendChild(row);inputs[key]=inp;
 }
 var out=uiNode('div','iui-result','—');
 wrap.appendChild(out);
 wrap.appendChild(uiNode('div','iui-note',mode==='loan'?'等额本息 · 本地计算':(mode==='split'?'账单分摊 · 本地计算':'本地计算 · 不调用模型')));
 update();
 return uiShell(spec,wrap);
}
function uiFlow(spec){
 var steps=Array.isArray(spec.steps)?spec.steps.slice(0,20):[];
 if(!steps.length)return null;
 var wrap=uiNode('div','iui-flow');
 for(var i=0;i<steps.length;i++){
  if(i)wrap.appendChild(uiNode('span','iui-arrow','→'));
  var s=steps[i];
  var label=(s&&typeof s==='object')?(s.title||s.text||''):s;
  wrap.appendChild(uiNode('div','iui-step',label));
 }
 return uiShell(spec,wrap);
}
function parseUISpecText(raw){
 var t=String(raw||'');
 var ti=t.lastIndexOf('</think>');
 if(ti>=0)t=t.slice(ti+8);
 t=t.replace(/```(?:ui|json)?/gi,'').trim();
 var a=t.indexOf('{'),b=t.lastIndexOf('}');
 if(a<0||b<=a)return null;
 var body=t.slice(a,b+1);
 var bodies=[body,body.replace(/\\"/g,'"')];
 var j=null;
 for(var bi=0;bi<bodies.length&&!j;bi++){
  try{j=JSON.parse(bodies[bi]);}catch(e){
   try{j=JSON.parse(bodies[bi].replace(/,\s*([}\]])/g,'$1'));}catch(e2){}
  }
 }
 if(!j||['compare','chart','calculator','flow'].indexOf(String(j.type))<0)return null;
 return j;
}
function wantsUIExplicit(s){return /智能(界面|视图|UI)|smart\s*ui/i.test(String(s||''));}
function uiIntent(s){return /对比|比较|vs\b|图表|柱状|折线|趋势|曲线|可视化|计算|算一下|算算|月供|利息|分摊|贷款|预算|画图|画个图|画一个图|绘图|作图|流程|步骤/i.test(String(s||''));}
async function planUI(model,question,answer){
 var r=await jpost('api/ui',{model:model,question:question,answer:answer});
 if(!r.ok)return null;
 var j=await r.json();
 return j.spec || parseUISpecText(j.text||'');
}
function renderUI(spec){
 if(!spec||typeof spec!=='object')return null;
 var t=String(spec.type||'');
 if(t==='compare')return uiCompare(spec);
 if(t==='chart')return uiChart(spec);
 if(t==='calculator')return uiCalcWidget(spec);
 if(t==='flow')return uiFlow(spec);
 if(t==='svg'){var svg=sanitizeSvg(spec.svg);return svg?makeSvgCard(svg):null;}
 return null;
}

function splitToolCalls(text){
 var s=String(text||""),out="",calls=[],i=0,pending=false,low=s.toLowerCase();
 while(i<s.length){
  var a=low.indexOf("<tool",i);
  if(a<0){out+=s.slice(i);break;}
  out+=s.slice(i,a);
  var headEnd=s.indexOf(">",a);
  if(headEnd<0){pending=true;break;}
  var head=s.slice(a,headEnd+1);
  var nm=head.match(/name\s*=\s*["']([^"']+)["']/i);
  var close=low.indexOf("</tool>",headEnd+1);
  if(!nm||close<0){pending=true;break;}
  calls.push({name:nm[1].toLowerCase(),body:s.slice(headEnd+1,close)});
  i=close+7;
 }
 var rr=extractRenderCalls(out);
 return {text:rr.text.trim(),calls:calls.concat(rr.calls),pending:pending};
}
function splitArgs(s){
 var out=[],start=0,depth=0,q=null;
 for(var i=0;i<s.length;i++){
  var ch=s.charAt(i);
  if(q){if(ch===q&&s.charAt(i-1)!=="\\")q=null;continue;}
  if(ch==="'"||ch==='"'){q=ch;continue;}
  if(ch==="["||ch==="{"||ch==="(")depth++;
  else if(ch==="]"||ch==="}"||ch===")")depth--;
  else if(ch===","&&depth===0){out.push(s.slice(start,i));start=i+1;}
 }
 if(s.slice(start).trim())out.push(s.slice(start));
 return out;
}
function parseLiteral(v){
 var s=String(v||"").trim();
 if(!s)return "";
 if(s.charAt(0)==="["||s.charAt(0)==="{"){
  var j=s.replace(/'/g,'"');
  try{return JSON.parse(j);}catch(e){}
 }
 if(s.charAt(0)==="'"||s.charAt(0)==='"')return s.slice(1,-1);
 var n=parseFloat(s);return isFinite(n)?n:s;
}
function scanRenderCalls(text){
 var s=String(text||""),out="",calls=[],i=0;
 while(i<s.length){
  var idx=s.indexOf("render_",i);
  if(idx<0){out+=s.slice(i);break;}
  out+=s.slice(i,idx);
  var p=s.indexOf("(",idx);
  if(p<0){out+=s.slice(idx);break;}
  var name=s.slice(idx,p).trim();
  if(!/^render_[a-z_]+$/i.test(name)){out+=s.slice(idx,idx+7);i=idx+7;continue;}
  var depth=0,q=null,end=-1;
  for(var j=p;j<s.length;j++){
   var ch=s.charAt(j);
   if(q){if(ch===q&&s.charAt(j-1)!=="\\")q=null;}
   else if(ch==="'"||ch==='"')q=ch;
   else if(ch==="(")depth++;
   else if(ch===")"){depth--;if(depth===0){end=j;break;}}
  }
  if(end<0){out+=s.slice(idx);break;}
  calls.push({name:name.toLowerCase(),args:splitArgs(s.slice(p+1,end)).map(parseLiteral)});
  i=end+1;
 }
 return {text:out,calls:calls};
}
function extractRenderCalls(text){
 var parts=String(text||"").split("```"),calls=[],out="";
 for(var i=0;i<parts.length;i++){
  if(i%2===1){
   var block=parts[i],nl=block.indexOf(String.fromCharCode(10)),lang=nl>=0?block.slice(0,nl):"",content=nl>=0?block.slice(nl+1):block;
   if(content.indexOf("render_")>=0){
    var r=scanRenderCalls(content);
    if(r.calls.length){calls=calls.concat(r.calls);if(r.text.trim())out+="```"+(lang?lang+String.fromCharCode(10):"")+r.text+"```";}
    else out+="```"+block+"```";
   }else out+="```"+block+"```";
  }else out+=parts[i];
 }
 var rest=scanRenderCalls(out);
 return {text:rest.text,calls:calls.concat(rest.calls)};
}
function toolSpec(call){
 if(!call)return null;
 var name=call.name,body=String(call.body||"").trim();
 if(call.args){
  var args=call.args;
  if(name==="render_svg"){var sv=String(args[0]||"");return sv.toLowerCase().indexOf("<svg")>=0?{type:"svg",svg:sv}:null;}
  if(name==="render_chart"){
   var oc=(args[0]&&typeof args[0]==="object"&&!Array.isArray(args[0]))?args[0]:{labels:args[0]||[],values:args[1]||[],title:args[2]||"数据图表",unit:args[3]||""};
   oc.type="chart";return oc;
  }
  if(name==="render_compare"){
   var om=(args[0]&&typeof args[0]==="object"&&!Array.isArray(args[0]))?args[0]:{title:args[0]||"数据对比",columns:args[1]||[],rows:args[2]||[]};
   om.type="compare";return om;
  }
  if(name==="render_calculator"){var oa=args[0];if(!oa||typeof oa!=="object")return null;oa.type="calculator";return oa;}
  if(name==="render_flow"){var of=args[0];if(Array.isArray(of))of={title:"流程",steps:of};if(!of||typeof of!=="object")return null;of.type="flow";return of;}
  return null;
 }
 if(name==="render_svg"){
  var si=body.toLowerCase().indexOf("<svg");
  return si>=0?{type:"svg",svg:body.slice(si)}:null;
 }
 body=body.replace(/^```[a-z]*/i,"").replace(/```$/,"").trim();
 var a=body.indexOf("{"),b=body.lastIndexOf("}");
 if(a<0||b<=a)return null;
 try{var o=JSON.parse(body.slice(a,b+1));}catch(e){return null;}
 if(!o||typeof o!=="object")return null;
 var map={render_chart:"chart",render_compare:"compare",render_calculator:"calculator",render_flow:"flow"};
 var t=map[name];if(!t)return null;
 o.type=t;return o;
}
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
 var oldThink=el.querySelector('details.think');
 var oldOpen=oldThink?oldThink.open:null;
 var oldBody=oldThink?oldThink.querySelector('.think-body'):null;
 var oldScroll=oldBody?oldBody.scrollTop:0;
 var oldAtBottom=oldBody?(oldBody.scrollHeight-oldBody.scrollTop-oldBody.clientHeight<24):false;
 el.innerHTML='';
 if(think){
  var tb=thinkBlock(think,'💭 '+(thinking?'思考中…':'思考'),true,false);
  if(oldOpen===false)tb.open=false;
  el.appendChild(tb);
  var nb=tb.querySelector('.think-body');
  if(nb){if(oldBody){if(oldAtBottom)nb.scrollTop=nb.scrollHeight;else nb.scrollTop=oldScroll;}else nb.scrollTop=nb.scrollHeight;}
 }
 var parts=splitToolCalls(ans);
 const a=document.createElement('div');a.textContent=parts.text;el.appendChild(a);
 if(parts.pending||parts.calls.length){const p=document.createElement('div');p.className='iui-pending';p.textContent='工具调用生成中…';el.appendChild(p);}
}
function normalizeLatex(s){
 var bs=String.fromCharCode(92);
 return String(s||'').split(bs+'[').join('$$').split(bs+']').join('$$').split(bs+'(').join('$').split(bs+')').join('$');
}
function escapeForMd(s){return s.split('&').join('&amp;').split('<').join('&lt;');}
function safeCssValue(v){
 var x=String(v||"").trim();
 if(!x)return "";
 var low=x.toLowerCase();
 if(low.indexOf("javascript:")>=0||low.indexOf("expression")>=0||low.indexOf("behavior")>=0||low.indexOf("@import")>=0)return "";
 if(low.indexOf("url(")>=0&&low.indexOf("url(#")<0)return "";
 return x.replace(/!important/g,"").trim();
}
function safeDecls(css){
 var props={fill:1,stroke:1,"stroke-width":1,"stroke-opacity":1,"fill-opacity":1,opacity:1,color:1,"font-size":1,"font-family":1,"font-weight":1,"text-anchor":1,"dominant-baseline":1,"stroke-linecap":1,"stroke-linejoin":1,"stroke-dasharray":1,"stop-color":1,"stop-opacity":1};
 var out=[],parts=String(css||"").split(";");
 for(var i=0;i<parts.length;i++){
  var d=parts[i],p=d.indexOf(":");
  if(p<0)continue;
  var k=d.slice(0,p).trim().toLowerCase(),v=safeCssValue(d.slice(p+1));
  if(!props[k]||!v)continue;
  out.push(k+":"+v);
 }
 return out.join(";");
}
function safeStyleSheet(css){
 var out=[],rules=String(css||"").split("}");
 for(var ri=0;ri<rules.length;ri++){
  var rule=rules[ri],p=rule.indexOf("{");
  if(p<0)continue;
  var sel=rule.slice(0,p).trim(),decl=safeDecls(rule.slice(p+1));
  if(!sel||!decl||sel.indexOf("@")>=0||sel.toLowerCase().indexOf("javascript")>=0)continue;
  var sels=sel.split(","),scoped=[];
  for(var si=0;si<sels.length;si++){
   var one=sels[si].trim();
   if(!one)continue;
   scoped.push(".svg-wrap "+one);
  }
  if(scoped.length)out.push(scoped.join(",")+"{"+decl+"}");
 }
 return out.join("");
}
function sanitizeSvg(text){
 var doc;
 try{doc=new DOMParser().parseFromString(String(text||""),"image/svg+xml");}catch(e){return null;}
 if(!doc||doc.getElementsByTagName("parsererror").length)return null;
 var root=doc.documentElement;
 if(!root||root.localName!=="svg")return null;
 var allowed={svg:1,g:1,path:1,circle:1,ellipse:1,rect:1,line:1,polyline:1,polygon:1,text:1,tspan:1,defs:1,linearGradient:1,radialGradient:1,stop:1,clipPath:1,mask:1,use:1,title:1,desc:1,marker:1,pattern:1,style:1};
 var attrs={viewbox:1,width:1,height:1,x:1,y:1,x1:1,y1:1,x2:1,y2:1,cx:1,cy:1,r:1,rx:1,ry:1,d:1,points:1,fill:1,"fill-opacity":1,stroke:1,"stroke-width":1,"stroke-opacity":1,"stroke-linecap":1,"stroke-linejoin":1,"stroke-dasharray":1,transform:1,"font-size":1,"font-family":1,"font-weight":1,"text-anchor":1,"dominant-baseline":1,opacity:1,id:1,class:1,style:1,dx:1,dy:1,offset:1,"stop-color":1,"stop-opacity":1,"clip-path":1,mask:1};
 var NS="http://www.w3.org/2000/svg",count=0;
 function build(src){
  count++;if(count>400)return null;
  var tag=src.localName;if(!allowed[tag])return null;
  var e=document.createElementNS(NS,tag);
  if(tag==="style"){e.textContent=safeStyleSheet(src.textContent||"");return e;}
  for(var i=0;i<src.attributes.length;i++){
   var a=src.attributes[i],n=a.name.toLowerCase(),v=a.value;
   if(n==="href"||n==="xlink:href"){if(v.charAt(0)==="#")e.setAttribute("href",v);continue;}
   if(n==="style"){var st=safeDecls(v);if(st)e.setAttribute("style",st);continue;}
   if(n==="class"){e.setAttribute("class",String(v).replace(/[^a-zA-Z0-9 _-]/g,""));continue;}
   if(!attrs[n])continue;
   if(v.indexOf("javascript:")===0||v.indexOf("data:text/html")===0)continue;
   e.setAttribute(a.name,v);
  }
  for(var ch=src.firstChild;ch;ch=ch.nextSibling){
   if(ch.nodeType===3)e.appendChild(document.createTextNode(ch.nodeValue));
   else if(ch.nodeType===1){var c=build(ch);if(c)e.appendChild(c);}
  }
  return e;
 }
 return build(root);
}
function makeSvgCard(svg){
 var card=document.createElement("div");card.className="svg-card";
 var bar=document.createElement("div");bar.className="svg-bar";bar.textContent="✦ 矢量图";
 var wrap=document.createElement("div");wrap.className="svg-wrap";wrap.appendChild(svg);
 card.appendChild(bar);card.appendChild(wrap);return card;
}
function renderSvgBlocks(root){
 var codes=Array.from(root.querySelectorAll("pre code"));
 for(var i=0;i<codes.length;i++){
  var txt=(codes[i].textContent||"").replace(/&lt;/g,"<").replace(/&gt;/g,">").replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,"&").trim();
  if(txt.slice(0,4).toLowerCase()!=="<svg")continue;
  var svg=sanitizeSvg(txt);if(!svg)continue;
  var pre=codes[i].closest("pre");if(pre)pre.replaceWith(makeSvgCard(svg));
 }
}
function protectRawSvg(s){
 var parts=String(s||"").split("```"),svgs=[];
 for(var p=0;p<parts.length;p+=2){
  var seg=parts[p],lower=seg.toLowerCase(),out="",pos=0;
  while(true){
   var a=lower.indexOf("<svg",pos);
   if(a<0){out+=seg.slice(pos);break;}
   var b=lower.indexOf("</svg>",a);
   if(b<0){out+=seg.slice(pos);break;}
   var end=b+6;
   out+=seg.slice(pos,a)+"@@RAWSVG"+svgs.length+"@@";
   svgs.push(seg.slice(a,end));pos=end;
  }
  parts[p]=out;
 }
 return {text:parts.join("```"),svgs:svgs};
}
function mountRawSvgs(root,list){
 for(var i=0;i<list.length;i++){
  var svg=sanitizeSvg(list[i]);if(!svg)continue;
  var card=makeSvgCard(svg);
  if(!replaceUIToken(root,"@@RAWSVG"+i+"@@",card))root.appendChild(card);
 }
}
function mdRender(el,ans){
 ans=normalizeLatex(ans);
 var rawSvg=protectRawSvg(ans);ans=rawSvg.text;
 let html;
 try{html=(window.marked&&typeof marked.parse==='function')?marked.parse(escapeForMd(ans)):null;}catch(e){html=null;}
 if(html===null){const a=document.createElement('div');a.textContent=ans;el.appendChild(a);return;}
 const a=document.createElement('div');a.className='md';a.innerHTML=html;el.appendChild(a);
 if(window.renderMathInElement){try{renderMathInElement(a,{delimiters:[{left:'$$',right:'$$',display:true},{left:String.fromCharCode(92)+'[',right:String.fromCharCode(92)+']',display:true},{left:String.fromCharCode(92)+'(',right:String.fromCharCode(92)+')',display:false},{left:'$',right:'$',display:false}],throwOnError:false,strict:false});}catch(e){}}
 renderSvgBlocks(a);
 mountRawSvgs(a,rawSvg.svgs);
}
function renderFinal(el,acc,knownBlock,knownInline){
 let think='',ans=acc;
 const t1=acc.indexOf('</think>');
 if(t1>=0){const t0=acc.lastIndexOf('<think>',t1);think=acc.slice(t0>=0?t0+7:0,t1);ans=acc.slice(t1+8);}
 el.innerHTML='';
 if(think){el.appendChild(thinkBlock(think,'💭 思考 · '+think.length+' 字（点击展开）',false,true));}
 var parts;
 if(Array.isArray(knownBlock)||Array.isArray(knownInline))parts={text:ans,specs:knownBlock||[],inline:knownInline||[]};
 else{
  var tp=splitToolCalls(ans),sp=[];
  for(var ci=0;ci<tp.calls.length;ci++){var ts=toolSpec(tp.calls[ci]);if(ts)sp.push(ts);}
  parts={text:tp.text,specs:sp,inline:[]};
 }
 if(parts.text)mdRender(el,parts.text);
 mountInlineUI(el,parts.inline);
 for(var i=0;i<parts.specs.length;i++){var n=renderUI(parts.specs[i]);if(n)el.appendChild(n);}
 return parts;
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
async function resumeStream(convId,mid,jobId){
 if(sending)return;
 var el=document.querySelector('#log .msg.ai[data-mid="'+mid+'"]');
 if(!el)return;
 sending=true;currentAbort=new AbortController();currentJobId=jobId;syncingPaused=true;
 var sb=$('send');if(sb){sb.textContent='■';sb.title='停止生成';sb.classList.add('stop');}
 pendingStreams[mid]={convId:convId,acc:'',last:0};
 try{
  var resp=await fetch('api/chat/stream?id='+encodeURIComponent(jobId),{signal:currentAbort.signal});
  if(!resp.ok)throw new Error('job ended');
  var reader=resp.body.getReader(),dec=new TextDecoder(),buf='',acc='';
  while(true){
   var rd=await reader.read();if(rd.done)break;
   buf+=dec.decode(rd.value,{stream:true});
   var i;
   while((i=buf.indexOf(String.fromCharCode(10)+String.fromCharCode(10)))>=0){
    var line=buf.slice(0,i).trim();buf=buf.slice(i+2);
    if(!line.startsWith('data: '))continue;
    var data=line.slice(6);if(data==='[DONE]')continue;
    try{var j=JSON.parse(data);}catch(e){continue;}
    if(j.snapshot!==undefined)acc=String(j.snapshot||'');
    if(j.delta)acc+=j.delta;
    if(j.snapshot!==undefined||j.delta){
     render(el,acc);
     var ps=pendingStreams[mid];if(ps){ps.acc=acc;if(Date.now()-ps.last>700){ps.last=Date.now();updateAssistant(convId,mid,acc,{pending:true,job_id:jobId});}}
    }
   }
  }
  finishResume(convId,mid,el,acc,false);
 }catch(e){
  var ps2=pendingStreams[mid];var acc2=ps2?ps2.acc:'';
  var stopped=(e&&(e.name==='AbortError'||String(e.message||'').indexOf('已停止')>=0));
  finishResume(convId,mid,el,acc2,true);
  if(currentId===convId&&!stopped)addMsg('sys','后台任务已结束');
 }finally{
  delete pendingStreams[mid];
  sending=false;currentAbort=null;currentJobId=null;syncingPaused=false;saveConvs();
  var sb2=$('send');if(sb2){sb2.textContent='↑';sb2.title='发送';sb2.disabled=false;sb2.classList.remove('stop');}
 }
}
function finishResume(convId,mid,el,acc,interrupted){
 var clean=String(acc||'');
 var ct=clean.indexOf('</'+'think>');
 if(ct>=0)clean=clean.slice(ct+8);
 clean=clean.split('<'+'think>').join('').trim();
 if(!clean)clean=String(acc||'');
 var tp=splitToolCalls(clean),ui=[];
 for(var ci=0;ci<tp.calls.length;ci++){var ts=toolSpec(tp.calls[ci]);if(ts)ui.push(ts);}
 var finalText=tp.text||(ui.length?'（智能视图）':clean);
 if(interrupted&&!finalText&&!ui.length){removeAssistant(convId,mid);}
 else updateAssistant(convId,mid,finalText,{pending:false,interrupted:!!interrupted,job_id:null,ui:ui,uiInline:[]});
 if(el&&el.isConnected){if(finalText||ui.length)renderFinal(el,finalText,ui,[]);}
}
function resumePending(){
 if(sending)return;
 for(var i=history.length-1;i>=0;i--){
  var m=history[i];
  if(m.role==='assistant'&&m.pending&&m.job_id){resumeStream(currentId,m.mid,m.job_id);return;}
 }
}
async function send(){
 if(sending){setStatus('⏳ 当前回答还在生成，请稍候再发送');return;}
 const model=$('model').value,p=$('prompt').value.trim();if(!p)return;
 sending=true;currentAbort=new AbortController();
 var sendBtn=$('send');if(sendBtn){sendBtn.textContent='■';sendBtn.title='停止生成';sendBtn.classList.add('stop');}
 lastUserPrompt=p;const convId=currentId;
 $('prompt').value='';addMsg('me',p);
 appendToConv(convId,{role:'user',content:p});
 const aid=startAssistant(convId);
 pendingStreams[aid]={convId:convId,acc:'',last:0};
 const reqMsgs=history.filter(function(m){return !(m.role==='assistant'&&m.pending);}).map(function(m){return {role:m.role,content:plainForModel(m)};});
 syncingPaused=true;currentJobId=null;
 try{
  try{const q=await (await fetch('api/queue')).json(); if(q.busy||q.waiting>0){addMsg('sys','⏳ 服务器正忙，请求已排队'+(q.waiting?('（前面还有 '+q.waiting+' 个）'):'')+'，请耐心等待…');}}catch(e){}
  const el=addMsg('ai','');el.setAttribute('data-mid',aid);
  const resp=await fetch('api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:model,messages:reqMsgs,think_level:$('thinkLevel').value,max_tokens:parseInt($('maxTok').value,10),intelligent_ui:true,conversation_id:convId,assistant_mid:aid,conversation:currentConv()}),signal:currentAbort.signal});
  if(!resp.ok){const err='请求失败：'+await resp.text();el.textContent=err;removeAssistant(convId,aid);removeLastUser(convId);delete pendingStreams[aid];return;}
  const reader=resp.body.getReader(),dec=new TextDecoder();
  let acc='',buf='';
  while(true){
   const {done,value}=await reader.read();if(done)break;
   buf+=dec.decode(value,{stream:true});
   let i;
   while((i=buf.indexOf('\n\n'))>=0){
    const line=buf.slice(0,i).trim();buf=buf.slice(i+2);
    if(line.startsWith('data: ')){const data=line.slice(6);if(data==='[DONE]')continue;
     try{const j=JSON.parse(data);if(j.meta){if(currentId===convId)setCtxMeta(j.meta);if(j.meta.job_id)currentJobId=j.meta.job_id;continue;}if(j.delta){acc+=j.delta;var stick=isAtBottom();render(el,acc);if(stick)scrollBottom();var ps=pendingStreams[aid];if(ps){ps.acc=acc;if(Date.now()-ps.last>700){ps.last=Date.now();updateAssistant(convId,aid,acc,{pending:true});}}}}catch(e){}}
   }
  }
  if(currentId===convId)renderCtx(acc.length);
  var stickFinal=isAtBottom();
  renderFinal(el,acc);
  let clean=acc;
  const ct=acc.indexOf('</think>');
  if(ct>=0){clean=acc.slice(ct+8);}
 clean=clean.split('<'+'think>').join('').trim();
  if(!clean){clean=acc;}
  const toolParts=splitToolCalls(clean);
  const uiBlock=[];
  for(var ti=0;ti<toolParts.calls.length;ti++){var ts=toolSpec(toolParts.calls[ti]);if(ts)uiBlock.push(ts);}
  const finalText=toolParts.text||(uiBlock.length?'（智能视图）':clean);
  updateAssistant(convId,aid,finalText,{pending:false,ui:uiBlock,uiInline:[]});
  delete pendingStreams[aid];
  if(currentId===convId&&!el.isConnected){const e=addMsg('ai','');renderFinal(e,finalText,uiBlock,uiInline);}
  if(stickFinal)scrollBottom();
 }catch(e){
  var stopped=(e&&(e.name==='AbortError'||String(e.message||'').indexOf('已停止')>=0));
  var ps2=pendingStreams[aid];
  if(ps2){
   if(stopped){
    if(!String(ps2.acc||'').trim())removeAssistant(convId,aid);
    else updateAssistant(convId,aid,ps2.acc||'',{pending:false,interrupted:true});
   }else if(!String(ps2.acc||'').trim()){
    removeAssistant(convId,aid);removeLastUser(convId);
   }else{
    updateAssistant(convId,aid,ps2.acc||'',{pending:false,interrupted:true});
   }
   delete pendingStreams[aid];
  }
  if(currentId===convId)addMsg('sys',stopped?'已停止生成':'出错：'+e.message);
 } finally {
  sending=false;currentAbort=null;currentJobId=null;syncingPaused=false;
  saveConvs();
  var sb2=$('send');if(sb2){sb2.textContent='↑';sb2.title='发送';sb2.disabled=false;sb2.classList.remove('stop');}
 }
}
window.addEventListener('beforeunload',function(){
 for(var mid in pendingStreams){
  var ps=pendingStreams[mid];
  if(ps)updateAssistant(ps.convId,mid,ps.acc||'',{pending:true});
 }
});
$('send').onclick=function(){if(sending){if(currentJobId)fetch('api/chat/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:currentJobId}),keepalive:true}).catch(function(){});if(currentAbort)currentAbort.abort();return;}send();};
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
$('clearBtn').onclick=function(){newConv();};
(function(){
 var cs=$('convSearch');
 if(cs){
  cs.addEventListener('input',renderThreads);
  cs.addEventListener('keydown',function(e){if(e.key==='Escape'){cs.value='';renderThreads();cs.blur();}});
 }
 document.addEventListener('keydown',function(e){
  if((e.ctrlKey||e.metaKey)&&String(e.key).toLowerCase()==='k'){
   e.preventDefault();
   if(window.innerWidth>=900&&document.body.classList.contains('side-hidden')){document.body.classList.remove('side-hidden');try{localStorage.setItem('muse_side_hidden','0');}catch(x){}}
   var s=$('convSearch');if(s){s.focus();s.select();}
  }
 });
})();
$('setBtn').onclick=async function(){
 $('setModal').style.display='flex';
 try{const j=await (await fetch('api/settings')).json();const st=j.settings||{};
  $('setThink').value=(st.think_level==='off')?'off':'high';
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
 if(r.ok){addMsg('sys','✅ 设置已保存（上下文长度变更需重新加载模型后生效）');$('thinkLevel').value=body.think_level;applyThinkLevels();$('maxTok').value=String(body.max_tokens_default);$('setModal').style.display='none';}
 else{addMsg('sys','保存失败');}
};
$('setPwSave').onclick=async function(){
 const np=$('setPw').value.trim();if(np.length<4){addMsg('sys','密码至少 4 位');return;}
 const r=await fetch('api/password',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({new_password:np})});
 if(r.ok){addMsg('sys','✅ 密码已修改，下次登录用新密码');$('setPw').value='';}
 else{addMsg('sys','修改失败：'+(await r.text()).slice(0,80));}
};
fetch('api/settings').then(function(r){return r.json();}).then(function(j){if(j.settings){$('thinkLevel').value=(j.settings.think_level==='off')?'off':'high';if(j.settings.max_tokens_default){$('maxTok').value=String(j.settings.max_tokens_default);}}}).catch(function(){});
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
initConvs();
</script>
<script>
(function(){
 var m=document.getElementById('menuBtn'),sc=document.getElementById('scrim');
 function close(){document.body.classList.remove('nav-open');}
 try{if(localStorage.getItem('muse_side_hidden')==='1')document.body.classList.add('side-hidden');}catch(e){}
 if(m)m.onclick=function(){
  if(window.innerWidth>=900){var h=document.body.classList.toggle('side-hidden');try{localStorage.setItem('muse_side_hidden',h?'1':'0');}catch(e){}}
  else document.body.classList.toggle('nav-open');
 };
 if(sc)sc.onclick=close;
 var ta=document.getElementById('prompt');
 if(ta){ta.addEventListener('input',function(){ta.style.height='auto';ta.style.height=Math.min(168,ta.scrollHeight)+'px';});}
 var sb=document.getElementById('send');
 if(sb&&ta)sb.addEventListener('click',function(){setTimeout(function(){ta.style.height='40px';},0);});
 document.querySelectorAll('#sidebar a,#sidebar button').forEach(function(el){el.addEventListener('click',function(){if(window.innerWidth<900)close();});});
 var th=document.getElementById('threads');
 if(th)th.addEventListener('click',function(ev){
  if(window.innerWidth>=900)return;
  if(ev.target&&(ev.target.closest('.t-act')||ev.target.closest('input')))return;
  close();
 });
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
        if self.path.startswith("/api/chat/stream"):
            q2 = parse_qs(urlparse(self.path).query)
            jid = (q2.get("id") or [""])[0]
            with jobs_lock:
                job = JOBS.get(jid)
            if not job:
                return self.send_json(404, {"error": "job not found"})
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            except Exception:
                return
            try:
                with job["cond"]:
                    snap = job["text"]
                    status = job["status"]
                self.wfile.write(("data: " + json.dumps({"snapshot": snap}, ensure_ascii=False) + chr(10) + chr(10)).encode("utf-8"))
                self.wfile.flush()
                off = len(snap)
                while True:
                    chunk = ""
                    with job["cond"]:
                        while len(job["text"]) <= off and job["status"] == "running":
                            job["cond"].wait(15)
                        chunk = job["text"][off:]
                        status = job["status"]
                    if chunk:
                        off += len(chunk)
                        self.wfile.write(("data: " + json.dumps({"delta": chunk}, ensure_ascii=False) + chr(10) + chr(10)).encode("utf-8"))
                        self.wfile.flush()
                    if status != "running":
                        self.wfile.write(b"data: [DONE]" + bytes([10, 10]))
                        self.wfile.flush()
                        break
                    if not chunk:
                        self.wfile.write(b": keepalive" + bytes([10, 10]))
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                pass
            return
        if self.path.startswith("/api/convs"):
            with conv_lock:
                items = _load_convs_db()
            return self.send_json(200, {"conversations": items})
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
                "models": [{"id": k, "label": v["label"], "think_levels": think_levels_for(k), "think_note": v.get("think_note", "")} for k, v in MODELS.items()]})
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
        if self.path.startswith("/api/convs"):
            items = d.get("conversations")
            if not isinstance(items, list):
                return self.send_json(400, {"error": "conversations required"})
            with conv_lock:
                _save_convs_db(items)
            return self.send_json(200, {"ok": True})
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
            tlev = normalize_think(name, tlev)
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
            if d.get("intelligent_ui"):
                base = str(msgs[0].get("content") or "")
                msgs[0] = {"role": "system", "content": base + (chr(10) + chr(10) if base else "") + UI_SYSTEM_PROMPT}
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
                    hello = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
                             "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]}
                    self.wfile.write(("data: " + json.dumps(hello, ensure_ascii=False) + NL2).encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                try:
                    with gen_lock:
                        llm = state["llm"]
                        hard_prompt = _hard_off_prompt(llm, msgs) if (tlev == "off" and name in HARD_OFF_MODELS) else None
                        if hard_prompt:
                            gen = llm.create_completion(prompt=hard_prompt, max_tokens=mt, stream=True,
                                                        stop=["<|im_end|>", "<|eot_id|>"])
                            text_mode = True
                        else:
                            gen = llm.create_chat_completion(messages=msgs, max_tokens=mt, stream=True)
                            text_mode = False
                        for chunk in gen:
                            if text_mode:
                                delta = chunk["choices"][0].get("text") or ""
                            else:
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
                    llm = state["llm"]
                    hard_prompt = _hard_off_prompt(llm, msgs) if (tlev == "off" and name in HARD_OFF_MODELS) else None
                    if hard_prompt:
                        res = llm.create_completion(prompt=hard_prompt, max_tokens=mt, stream=False,
                                                    stop=["<|im_end|>", "<|eot_id|>"])
                        content = res["choices"][0].get("text") or ""
                        usage = res.get("usage", {})
                    else:
                        res = llm.create_chat_completion(messages=msgs, max_tokens=mt, stream=False)
                        content = res["choices"][0]["message"].get("content") or ""
                        usage = res.get("usage", {})
            except Exception as e:
                return self.send_json(500, {"error": {"message": repr(e)[:300], "type": "server_error"}})
            finally:
                with q_lock:
                    q_state["waiting"] -= 1
            return self.send_json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": usage})
        if self.path.startswith("/api/load"):
            name = d.get("model")
            if name not in MODELS:
                return self.send_json(400, {"error": "unknown model"})
            if state["name"] != name and state["loading"] != name:
                threading.Thread(target=load_model, args=(name,), daemon=True).start()
            return self.send_json(200, {"ok": True, "loading": name})
        if self.path.startswith("/api/chat/stop"):
            jid = str(d.get("id") or "")
            with stop_lock:
                ev = STOP_EVENTS.get(jid)
            if ev:
                ev.set()
            return self.send_json(200, {"ok": bool(ev)})
        if self.path.startswith("/api/chat"):
            name = d.get("model")
            prompt = (d.get("prompt") or "").strip()
            raw_msgs = d.get("messages")
            has_msgs = isinstance(raw_msgs, list) and bool(raw_msgs)
            if name not in MODELS or (not prompt and not has_msgs):
                return self.send_json(400, {"error": "model/messages required"})
            tlev = d.get("think_level") or settings.get("think_level", "high")
            if d.get("think") is False:
                tlev = "off"
            if tlev not in ("off", "low", "medium", "high"):
                tlev = "high"
            tlev = normalize_think(name, tlev)
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
            if d.get("intelligent_ui"):
                base = str(msgs[0].get("content") or "")
                msgs[0] = {"role": "system", "content": base + (chr(10) + chr(10) if base else "") + UI_SYSTEM_PROMPT}
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
            conv = d.get("conversation")
            if isinstance(conv, dict):
                _upsert_conv_snapshot(conv)
            conv_id = str(d.get("conversation_id") or (conv or {}).get("id") or "")
            mid = str(d.get("assistant_mid") or "")
            job_id = secrets.token_hex(10)
            stop_event = threading.Event()
            job = {"id": job_id, "text": "", "status": "running", "error": None,
                   "cond": threading.Condition(), "updated": time.time()}
            with stop_lock:
                STOP_EVENTS[job_id] = stop_event
            with jobs_lock:
                nowj = time.time()
                for old_id, old_job in list(JOBS.items()):
                    if old_job.get("status") != "running" and nowj - old_job.get("updated", 0) > 600:
                        JOBS.pop(old_id, None)
                JOBS[job_id] = job
            with q_lock:
                if q_state["waiting"] >= int(settings.get("max_queue") or 8):
                    with stop_lock:
                        STOP_EVENTS.pop(job_id, None)
                    with jobs_lock:
                        JOBS.pop(job_id, None)
                    return self.send_json(429, {"error": "busy, retry later"})
                q_state["waiting"] += 1
            client = {"ok": True}

            def send_evt(obj):
                if not client["ok"]:
                    return
                try:
                    self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + chr(10) + chr(10)).encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    client["ok"] = False
                except Exception:
                    client["ok"] = False

            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
            except Exception:
                client["ok"] = False
            meta = {"used": est(msgs), "budget": budget, "ctx": int(state.get("ctx") or 16384),
                    "max_tokens": mt, "trimmed": trimmed, "think_level": tlev, "job_id": job_id}
            send_evt({"meta": meta})
            if conv_id and mid:
                _update_conv_message(conv_id, mid, "", True, {"job_id": job_id})
            job_text = ""
            last_db = 0
            stopped = False
            try:
                if state["name"] != name or state["llm"] is None:
                    load_model(name)
                if state["llm"] is None:
                    raise RuntimeError("model failed to load")
                with gen_lock:
                    if stop_event.is_set():
                        stopped = True
                    else:
                        llm = state["llm"]
                        hard_prompt = _hard_off_prompt(llm, msgs) if (tlev == "off" and name in HARD_OFF_MODELS) else None
                        if hard_prompt:
                            gen = llm.create_completion(prompt=hard_prompt, max_tokens=mt, stream=True,
                                                        stop=["<|im_end|>", "<|eot_id|>"])
                            text_mode = True
                        else:
                            gen = llm.create_chat_completion(messages=msgs, max_tokens=mt, stream=True)
                            text_mode = False
                        for chunk in gen:
                            if stop_event.is_set():
                                stopped = True
                                break
                            if text_mode:
                                delta = chunk["choices"][0].get("text") or ""
                            else:
                                delta = chunk["choices"][0]["delta"].get("content") or ""
                            if not delta:
                                continue
                            job_text += delta
                            with job["cond"]:
                                job["text"] = job_text
                                job["updated"] = time.time()
                                job["cond"].notify_all()
                            send_evt({"delta": delta})
                            now = time.time()
                            if conv_id and mid and now - last_db > 0.8:
                                last_db = now
                                _update_conv_message(conv_id, mid, job_text, True, {"job_id": job_id})
                final_text = _clean_reply_text(job_text) or job_text
                if conv_id and mid:
                    _update_conv_message(conv_id, mid, final_text, False,
                                         {"job_id": None, "interrupted": (stopped and not final_text)})
                with job["cond"]:
                    job["status"] = "stopped" if stopped else "done"
                    job["updated"] = time.time()
                    job["cond"].notify_all()
                if client["ok"]:
                    try:
                        self.wfile.write(b"data: [DONE]" + bytes([10, 10]))
                        self.wfile.flush()
                    except Exception:
                        pass
            except Exception as e:
                final_text = _clean_reply_text(job_text) or job_text
                if conv_id and mid:
                    _update_conv_message(conv_id, mid, final_text, False,
                                         {"job_id": None, "interrupted": True})
                with job["cond"]:
                    job["status"] = "error"
                    job["error"] = repr(e)[:200]
                    job["updated"] = time.time()
                    job["cond"].notify_all()
                send_evt({"delta": chr(10) + "[错误] " + repr(e)[:200]})
            finally:
                with stop_lock:
                    STOP_EVENTS.pop(job_id, None)
                with q_lock:
                    q_state["waiting"] -= 1
            return
        if self.path.startswith("/api/ui"):
            name = d.get("model")
            q = str(d.get("question") or "").strip()[:2000]
            a = str(d.get("answer") or "").strip()[:6000]
            if name not in MODELS or not q or not a:
                return self.send_json(400, {"error": "model/question/answer required"})
            if state["name"] != name or state["llm"] is None:
                return self.send_json(409, {"error": "model not loaded"})
            try:
                spec = None
                txt = ""
                with gen_lock:
                    llm = state["llm"]
                    for attempt in range(2):
                        prompt = UI_PLANNER_PROMPT + chr(10) + "问题：" + q + chr(10) + "回答：" + a
                        if attempt:
                            prompt += (chr(10) + chr(10) + "上一次输出无法解析或不符合 schema：" +
                                       str(txt)[:500] + chr(10) + "请重新只输出一个合法 JSON 对象。")
                        msgs = [{"role": "system", "content": "/no_think"},
                                {"role": "user", "content": prompt}]
                        res = llm.create_chat_completion(messages=msgs, max_tokens=420, temperature=0.1)
                        txt = (res["choices"][0].get("message") or {}).get("content") or ""
                        spec = _ui_extract_json(txt)
                        if _ui_valid(spec):
                            break
                        spec = None
                return self.send_json(200, {"text": txt, "spec": spec})
            except Exception as e:
                return self.send_json(500, {"error": repr(e)[:200]})
        self.send_json(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(os.environ.get("CHAT_PORT", "8888"))
    print("chat server on", port, flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
