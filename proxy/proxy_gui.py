# -*- coding: utf-8 -*-
"""US-111 proxy control panel.

Small tkinter GUI over the PowerShell scripts in this folder.
Run with pythonw.exe (no console window).
"""
import os
import sys
import json
import time
import socket
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext

BASE = os.path.dirname(os.path.abspath(__file__))
START_PS = os.path.join(BASE, "start-us-proxy.ps1")
STOP_PS = os.path.join(BASE, "stop-us-proxy.ps1")
LOG = os.path.join(BASE, "forward.log")

CONFIG = {}
_cfg_path = os.path.join(BASE, "config.json")
if os.path.exists(_cfg_path):
    try:
        CONFIG = json.load(open(_cfg_path, encoding="utf-8"))
    except Exception:
        CONFIG = {}

PROXY_USER = CONFIG.get("user", "proxyuser")
PROXY_PASS = CONFIG.get("pass", "change-me")
PORT = int(CONFIG.get("port", 3128))
PROXY = "http://%s:%s@127.0.0.1:%d" % (PROXY_USER, PROXY_PASS, PORT)
TASK = "US-111-Proxy-Watchdog"
HIDE = 0x08000000 if os.name == "nt" else 0

BG, FG, DIM, CARD = "#1e1f22", "#e6e6e6", "#9aa0a6", "#2b2d31"
OKC, BADC, WARNC = "#3fb950", "#f85149", "#d29922"
FUI = ("Microsoft YaHei UI", 10)
FMO = ("Consolas", 9)


def decode_bytes(b):
    if not b:
        return ""
    for enc in ("utf-8", "gbk"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", "replace")


def ps_file(path, timeout=90):
    p = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
        capture_output=True, timeout=timeout, creationflags=HIDE)
    return (decode_bytes(p.stdout) + decode_bytes(p.stderr)).strip()


def ps_cmd(cmd, timeout=30):
    p = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
        capture_output=True, timeout=timeout, creationflags=HIDE)
    return (decode_bytes(p.stdout) + decode_bytes(p.stderr)).strip()


def listening():
    s = socket.socket()
    s.settimeout(1.0)
    try:
        return s.connect_ex(("127.0.0.1", PORT)) == 0
    finally:
        s.close()


def egress():
    try:
        p = subprocess.run(
            ["curl.exe", "-s", "--max-time", "12", "-x", PROXY, "https://api.ipify.org"],
            capture_output=True, text=True, timeout=18,
            creationflags=HIDE, encoding="utf-8", errors="replace")
        return (p.stdout or "").strip()
    except Exception:
        return ""


def wd_running():
    try:
        p = subprocess.run(["schtasks", "/Query", "/TN", TASK, "/FO", "LIST"],
                           capture_output=True, text=True, timeout=10,
                           creationflags=HIDE, encoding="utf-8", errors="replace")
        return p.returncode == 0
    except Exception:
        return False


class Panel:
    def __init__(self, root):
        self.root = root
        self.busy = False
        self.q = queue.Queue()
        self._probing = False
        self._tick = 0
        root.title('US-111 代理控制台')
        root.configure(bg=BG)
        root.geometry('600x700+%d+%d' % (max(0,(root.winfo_screenwidth()-600)//2), max(0,(root.winfo_screenheight()-700)//2)))
        root.minsize(520, 600)
        self._head()
        self._cards()
        self._buttons()
        self._log()
        self.root.after(80, self.pump)
        self.root.after(200, self.refresh)
        self.root.after(5000, self.auto_refresh)

    def _head(self):
        f = tk.Frame(self.root, bg=BG)
        f.pack(fill='x', padx=16, pady=(14, 6))
        self.dot = tk.Canvas(f, width=14, height=14, bg=BG, highlightthickness=0)
        self.dot.pack(side='left')
        self.circle = self.dot.create_oval(2, 2, 12, 12, fill=DIM, outline='')
        self.title = tk.Label(f, text='检查中', bg=BG, fg=FG,
                              font=(FUI[0], 15, 'bold'))
        self.title.pack(side='left', padx=8)
        tk.Button(f, text='刷新', command=self.refresh, bg=CARD, fg=FG,
                  relief='flat', padx=10, pady=2, font=FUI,
                  cursor='hand2').pack(side='right')

    def _cards(self):
        card = tk.Frame(self.root, bg=CARD)
        card.pack(fill='x', padx=16, pady=6, ipady=8)
        self.v = {}
        rows = [('本地地址', 'addr'), ('出口 IP', 'ip'),
                ('延迟', 'lat'), ('守护任务', 'wd')]
        for i, (k, key) in enumerate(rows):
            tk.Label(card, text=k, bg=CARD, fg=DIM, font=FUI, width=11,
                     anchor='w').grid(row=i, column=0, sticky='w',
                                      padx=(12, 4), pady=3)
            lab = tk.Label(card, text='--', bg=CARD, fg=FG, font=FMO, anchor='w')
            lab.grid(row=i, column=1, sticky='w', padx=(0, 12), pady=3)
            self.v[key] = lab

    def _buttons(self):
        f = tk.Frame(self.root, bg=BG)
        f.pack(fill='x', padx=16, pady=6)
        spec = [('启动代理', self.on_start, OKC), ('停止代理', self.on_stop, BADC),
                ('重启', self.on_restart, DIM)]
        self.btns = []
        for i, (t, cb, c) in enumerate(spec):
            b = self._mk(f, t, cb, c)
            b.grid(row=0, column=i, sticky='ew', padx=(0, 6), ipady=6)
            f.columnconfigure(i, weight=1)
            self.btns.append(b)
        g = tk.Frame(self.root, bg=BG)
        g.pack(fill='x', padx=16, pady=(0, 6))
        spec2 = [('守护任务', self.on_wd, DIM), ('测试', self.on_test, DIM),
                 ('日志', self.on_log, DIM), ('退出', self.on_exit, DIM)]
        for i, (t, cb, c) in enumerate(spec2):
            b = self._mk(g, t, cb, c)
            b.grid(row=0, column=i, sticky='ew', padx=(0, 6), ipady=4)
            g.columnconfigure(i, weight=1)
            if cb == self.on_wd:
                self.btn_wd = b
            self.btns.append(b)

    def _mk(self, parent, text, cb, color):
        return tk.Button(parent, text=text, command=cb, bg=CARD, fg=color,
                         relief='flat', font=FUI, cursor='hand2', bd=0,
                         activebackground='#3a3d42', activeforeground=color)

    def _log(self):
        f = tk.Frame(self.root, bg=BG)
        f.pack(fill='both', expand=True, padx=16, pady=(4, 10))
        tk.Label(f, text='日志', bg=BG, fg=DIM, font=FUI,
                 anchor='w').pack(fill='x')
        self.log = scrolledtext.ScrolledText(
            f, bg='#141517', fg='#c9d1d9', font=FMO, relief='flat',
            wrap='word', height=10, width=40, insertbackground=FG)
        self.log.pack(fill='both', expand=True, pady=(3, 0))

    def logln(self, msg):
        ts = time.strftime('%H:%M:%S')
        self.log.insert('end', '[' + ts + '] ' + str(msg) + chr(10))
        self.log.see('end')

    def status(self, text, color):
        self.title.config(text=text, fg=color)
        self.dot.itemconfig(self.circle, fill=color)

    def set_busy(self, on, msg=None):
        self.busy = on
        if msg:
            self.status(msg, WARNC)
        st = 'disabled' if on else 'normal'
        for b in self.btns:
            b.config(state=st)

    def bg(self, fn, done):
        def w():
            try:
                r = fn()
            except Exception as e:
                r = '错误 ' + repr(e)
            self.q.put((done, r))
        threading.Thread(target=w, daemon=True).start()

    def pump(self):
        try:
            while True:
                cb, payload = self.q.get_nowait()
                try:
                    cb(payload)
                except Exception as e:
                    self.logln('回调错误: ' + repr(e))
        except queue.Empty:
            pass
        self.root.after(80, self.pump)

    def _apply_quick(self, up):
        if up:
            self.status('运行中', OKC)
        else:
            self.status('已停止', BADC)

    def auto_refresh(self):
        if not self.busy and not self._probing:
            self._tick += 1
            self.refresh(deep=(self._tick % 3 == 0))
        self.root.after(5000, self.auto_refresh)

    def refresh(self, deep=True):
        if self._probing:
            return
        self._probing = True

        def job():
            up = listening()
            self.q.put((self._apply_quick, up))
            if not deep or not up:
                return (up, None, None)
            ip = egress()
            return (up, ip, wd_running())

        def done(r):
            self._probing = False
            if isinstance(r, str):
                self.logln('探测: ' + r)
                return
            up, ip, wd = r
            if ip is not None:
                if up and ip:
                    self.status('运行中', OKC)
                elif up:
                    self.status('运行中 (IP 未知)', WARNC)
                self.v['ip'].config(text=ip or '--')
            if wd is not None:
                self.v['wd'].config(text='开启' if wd else '关闭',
                                    fg=OKC if wd else DIM)
                self.btn_wd.config(text='停用守护' if wd else '启用守护',
                                   fg=BADC if wd else OKC)
            self.v['addr'].config(text='127.0.0.1:' + str(PORT))
        self.bg(job, done)

    def on_start(self):
        self.set_busy(True, '正在启动')
        self.logln('正在启动隧道...')
        def job():
            return ps_file(START_PS)
        def done(out):
            for line in str(out).splitlines():
                if line.strip():
                    self.logln(line.strip())
            self.set_busy(False)
            self.refresh()
        self.bg(job, done)

    def on_stop(self):
        self.set_busy(True, '正在停止')
        self.logln('正在停止隧道...')
        def job():
            return ps_file(STOP_PS)
        def done(out):
            for line in str(out).splitlines():
                if line.strip():
                    self.logln(line.strip())
            self.set_busy(False)
            self.refresh()
        self.bg(job, done)

    def on_restart(self):
        self.set_busy(True, '正在重启')
        self.logln('正在重启隧道...')
        def job():
            a = ps_file(STOP_PS)
            b = ps_file(START_PS)
            return a + chr(10) + b
        def done(out):
            for line in str(out).splitlines():
                if line.strip():
                    self.logln(line.strip())
            self.set_busy(False)
            self.refresh()
        self.bg(job, done)

    def on_wd(self):
        self.set_busy(True, '正在设置守护')
        on = not wd_running()
        self.logln('守护任务 -> ' + ('ON' if on else 'OFF'))
        def job():
            vbs = os.path.join(BASE, 'watchdog-hidden.vbs')
            if on:
                tr = 'wscript.exe "' + vbs + '"'
                return ps_cmd('schtasks /Create /TN "' + TASK + '" /TR ' +
                              chr(39) + tr + chr(39) +
                              ' /SC MINUTE /MO 1 /F')
            return ps_cmd('schtasks /Delete /TN "' + TASK + '" /F')
        def done(out):
            if str(out).strip():
                self.logln(str(out).strip())
            self.set_busy(False)
            self.refresh()
        self.bg(job, done)

    def on_test(self):
        self.set_busy(True, '正在测试')
        self.logln('正在测试代理...')
        def job():
            t0 = time.time()
            p = subprocess.run(
                ['curl.exe', '-s', '-o', 'NUL', '-w', '%{http_code}',
                 '--max-time', '15', '-x', PROXY,
                 'https://www.google.com/generate_204'],
                capture_output=True, text=True, timeout=20,
                creationflags=HIDE, encoding='utf-8', errors='replace')
            return ((p.stdout or '').strip(), int((time.time() - t0) * 1000))
        def done(r):
            code, ms = r
            self.v['lat'].config(text=str(ms) + ' ms')
            if code == '204':
                self.logln('测试通过: HTTP 204, ' + str(ms) + ' ms')
            else:
                self.logln('测试失败: ' + str(code) + ' in ' + str(ms) + ' ms')
            self.set_busy(False)
        self.bg(job, done)

    def on_log(self):
        if os.path.exists(LOG):
            subprocess.Popen(['notepad.exe', LOG])
        else:
            self.logln('日志文件不存在: ' + LOG)

    def on_exit(self):
        self.root.destroy()


def main():
    root = tk.Tk()
    pan = Panel(root)
    root.after(150, pan.refresh)
    if os.environ.get('PROXY_GUI_DEBUG'):
        def check():
            root.update_idletasks()
            lines = ['window %d x %d' % (root.winfo_width(), root.winfo_height()),
                     'requested %d x %d' % (root.winfo_reqwidth(), root.winfo_reqheight()),
                     'log width %d req %d' % (pan.log.winfo_width(), pan.log.winfo_reqwidth())]
            open(os.path.join(BASE, 'gui_debug.txt'), 'w').write(chr(10).join(lines))
            root.destroy()
        root.after(1200, check)
    root.mainloop()


if __name__ == '__main__':
    main()
