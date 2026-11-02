#!/usr/bin/env python3
# msagent job server v3: long-poll queue, execution-aware redelivery (stdlib only).
# Runs on the InstaBox box, port 8899, published publicly via `instabox port publish`.
# v3: reaper only requeues when the delivering worker is idle (busy = still executing),
#     hard requeue when worker presumed dead, run cap 2, /drop endpoint, worker heartbeat.
import json, os, re, time, threading, secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

DATA_DIR = os.environ.get("MSAGENT_DIR", "/data/msagent")
TOKEN_FILE = os.path.join(DATA_DIR, "token")
BASE_FILE = os.path.join(DATA_DIR, "base_url")
LOG_FILE = os.path.join(DATA_DIR, "results.log")
ACCESS_LOG = os.path.join(DATA_DIR, "access.log")
PORT = int(os.environ.get("MSAGENT_PORT", "8899"))
POLL_WAIT = 8           # server-side long-poll wait (seconds); MUST stay well under the public gateway's idle cutoff (~10-20s), else replies die as BROKENPIPE
WORKER_TIMEOUT = 75     # worker considered offline after this
REDELIVER_AFTER = 90    # requeue if delivered worker is idle and no result within this
DEAD_AFTER = 2700       # 45 min: worker presumed gone, requeue regardless
HARD_AFTER = 5400       # 90 min: absolute cap, requeue regardless
MAX_RUNS = 2            # a job may be executed at most this many times
MAX_OUT = 400_000
KEEP_RESULTS = 100

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE, "w") as f:
        f.write(secrets.token_hex(16))
TOKEN = open(TOKEN_FILE).read().strip()

jobs = []       # pending: {"id","cmd","target","queued_at","runs"}
inflight = {}   # jid -> {"job": {...}, "ts": float, "worker": str}
results = []    # newest first
workers = {}    # id -> {"info": {}, "last_seen": float}
lock = threading.Condition()


def alog(line):
    try:
        with open(ACCESS_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def jlog(obj):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def reaper():
    while True:
        time.sleep(15)
        now = time.time()
        with lock:
            for jid in list(inflight):
                e = inflight[jid]
                age = now - e["ts"]
                wrec = workers.get(e.get("worker") or "")
                last = wrec["last_seen"] if wrec else 0
                idle_worker = (now - last) < 45      # worker is polling again -> it is NOT running our job
                worker_gone = (now - last) > DEAD_AFTER
                runs = e["job"].get("runs", 1)
                should = ((idle_worker and age > REDELIVER_AFTER) or
                          (worker_gone and age > 300) or
                          age > HARD_AFTER)
                if not should:
                    continue
                if runs >= MAX_RUNS:
                    inflight.pop(jid)
                    alog("%s REAPER dropped %s (runs=%d, age=%ds)" %
                         (time.strftime("%H:%M:%S"), jid, runs, int(age)))
                    continue
                nj = dict(e["job"])
                nj["runs"] = runs + 1
                jobs.append(nj)
                inflight.pop(jid)
                lock.notify_all()
                alog("%s REAPER requeued %s (age=%ds idle=%s gone=%s runs->%d)" %
                     (time.strftime("%H:%M:%S"), jid, int(age), idle_worker,
                      worker_gone, runs + 1))


threading.Thread(target=reaper, daemon=True).start()


WORKER_TEMPLATE = r'''
import json, os, socket, subprocess, time, platform, threading, urllib.request, urllib.error
BASE = "__BASE__"
TOKEN = "__TOKEN__"
WID = socket.gethostname()[:60]
done = {}        # jid -> result dict (redelivery re-post)
current = [None] # jid currently executing (for heartbeat)
stop = threading.Event()

def call(path, data=None, timeout=70):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return (r.status, json.loads(raw) if raw else {})

def heartbeat():
    while not stop.is_set():
        try:
            call("/heartbeat?token=%s&worker=%s&job=%s" % (TOKEN, WID, current[0] or ""),
                 timeout=10)
        except Exception:
            pass
        stop.wait(20)

info = {"host": WID, "user": os.environ.get("USER", "?"), "cwd": os.getcwd(),
        "python": platform.python_version(), "platform": platform.platform()[:120]}
for i in range(5):
    try:
        call("/register?token=%s&worker=%s" % (TOKEN, WID), info)
        break
    except Exception as e:
        print("register retry", repr(e), flush=True)
        time.sleep(2 * (i + 1))
threading.Thread(target=heartbeat, daemon=True).start()
print("msagent worker v2 up:", BASE, flush=True)
while True:
    try:
        st, job = call("/job?token=%s&worker=%s" % (TOKEN, WID), timeout=70)
        if st == 204:
            continue
        jid, cmd = job["id"], job["cmd"]
        if jid in done:   # redelivered after we already ran it: just re-post the result
            call("/result?token=%s&job=%s&worker=%s" % (TOKEN, jid, WID), done[jid])
            print("re-posted", jid, flush=True)
            continue
        current[0] = jid
        print("run", jid, cmd[:80], flush=True)
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True,
                               encoding="utf-8", errors="replace", timeout=7200)
            res = {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        except subprocess.TimeoutExpired as e:
            so = e.stdout or ""
            if isinstance(so, bytes):
                so = so.decode(errors="replace")
            res = {"code": 124, "stdout": so, "stderr": "TIMEOUT after 7200s"}
        current[0] = None
        done[jid] = res
        if len(done) > 200:
            done.pop(next(iter(done)))
        call("/result?token=%s&job=%s&worker=%s" % (TOKEN, jid, WID), res)
        print("done", jid, flush=True)
    except urllib.error.HTTPError as e:
        print("http", e.code, "retry later", flush=True)
        time.sleep(120 if e.code == 409 else 15)
    except Exception as e:
        current[0] = None
        print("loop err", repr(e), flush=True)
        time.sleep(10)
'''.strip()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            alog("%s %s BROKENPIPE %d" % (time.strftime("%H:%M:%S"),
                                          self.path.split("?")[0], code))
            return
        alog("%s %s %s %d %dB" % (time.strftime("%H:%M:%S"), self.command,
                                  self.path.split("?")[0], code, len(b)))

    def serve_agent(self):
        base = ""
        try:
            base = open(BASE_FILE).read().strip()
        except Exception:
            pass
        src = WORKER_TEMPLATE.replace("__BASE__", base).replace("__TOKEN__", TOKEN)
        b = src.encode("utf-8")
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def auth(self, q):
        return q.get("token", [""])[0] == TOKEN

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self.auth(q):
            return self.reply(403, {"error": "bad token"})
        if u.path == "/status":
            now = time.time()
            ws = {w: {"info": v["info"], "last_seen": v["last_seen"],
                      "online": now - v["last_seen"] < WORKER_TIMEOUT}
                  for w, v in workers.items()}
            inf = {jid: {"age": int(now - e["ts"]), "worker": e.get("worker"),
                         "runs": e["job"].get("runs", 1)}
                   for jid, e in inflight.items()}
            return self.reply(200, {"workers": ws, "pending": len(jobs),
                                    "inflight": inf, "results": results[:50]})
        if u.path == "/job":
            w = q.get("worker", ["anon"])[0][:80]
            rec = workers.setdefault(w, {"info": {}, "last_seen": 0})
            rec["last_seen"] = time.time()
            with lock:
                lock.wait(POLL_WAIT)
                for i, j in enumerate(jobs):
                    if not j.get("target") or j["target"] == w:
                        jobs.pop(i)
                        inflight[j["id"]] = {"job": j, "ts": time.time(),
                                             "worker": w}
                        return self.reply(200, {"id": j["id"], "cmd": j["cmd"]})
            return self.reply(204, {})
        if u.path == "/agent":
            return self.serve_agent()
        if u.path.startswith("/files/"):
            name = u.path[len("/files/"):]
            if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
                return self.reply(404, {"error": "bad name"})
            fp = os.path.join(DATA_DIR, "files", name)
            if not os.path.isfile(fp):
                return self.reply(404, {"error": "not found"})
            size = os.path.getsize(fp)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with open(fp, "rb") as f:
                    while True:
                        chunk = f.read(256 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                alog("%s GET /files/%s 200 %dB" % (time.strftime("%H:%M:%S"), name, size))
            except (BrokenPipeError, ConnectionResetError):
                alog("%s GET /files/%s BROKENPIPE" % (time.strftime("%H:%M:%S"), name))
            return
        return self.reply(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self.auth(q):
            return self.reply(403, {"error": "bad token"})
        if u.path == "/submit":
            d = self.body_json()
            cmd = (d.get("cmd") or "")[:100_000]
            if not cmd:
                return self.reply(400, {"error": "cmd required"})
            target = (d.get("target") or "")[:80] or None
            jid = time.strftime("%H%M%S") + "-" + secrets.token_hex(3)
            with lock:
                jobs.append({"id": jid, "cmd": cmd, "target": target,
                             "queued_at": time.time(), "runs": 1})
                lock.notify_all()
            return self.reply(200, {"id": jid})
        if u.path == "/register":
            w = q.get("worker", ["anon"])[0][:80]
            rec = workers.setdefault(w, {"info": {}, "last_seen": 0})
            rec["info"] = self.body_json()
            rec["last_seen"] = time.time()
            return self.reply(200, {"ok": True})
        if u.path == "/heartbeat":
            w = q.get("worker", [""])[0][:80]
            jid = q.get("job", [""])[0][:40]
            now = time.time()
            with lock:
                if w in workers:
                    workers[w]["last_seen"] = now
                if jid and jid in inflight:
                    inflight[jid]["ts"] = now
            return self.reply(200, {"ok": True})
        if u.path == "/drop":
            d = self.body_json()
            jid = (d.get("id") or q.get("id", [""])[0])[:40]
            with lock:
                if d.get("all") or q.get("all", [""])[0] == "1":
                    np = len(jobs)
                    ni = len(inflight)
                    jobs.clear()
                    inflight.clear()
                    lock.notify_all()
                    alog("%s DROP all pending=%d inflight=%d" %
                         (time.strftime("%H:%M:%S"), np, ni))
                    return self.reply(200, {"dropped_pending": np,
                                            "dropped_inflight": ni})
                removed = 0
                for i, j in enumerate(list(jobs)):
                    if j["id"] == jid:
                        jobs.pop(i)
                        removed += 1
                if jid in inflight:
                    inflight.pop(jid)
                    removed += 1
                alog("%s DROP %s removed=%d" %
                     (time.strftime("%H:%M:%S"), jid, removed))
                return self.reply(200, {"dropped": removed})
        if u.path == "/result":
            jid = q.get("job", [""])[0][:40]
            with lock:
                inflight.pop(jid, None)
            d = self.body_json()
            out = {"id": jid, "worker": q.get("worker", ["?"])[0][:80],
                   "code": d.get("code"),
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "stdout": (d.get("stdout") or "")[:MAX_OUT],
                   "stderr": (d.get("stderr") or "")[:MAX_OUT]}
            results.insert(0, out)
            del results[KEEP_RESULTS:]
            jlog(out)
            return self.reply(200, {"ok": True})
        return self.reply(404, {"error": "not found"})


if __name__ == "__main__":
    print("msagent server v3 on port", PORT, flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
