#!/usr/bin/env python3
"""Local segment-annotation tool for building creampie training data.

Serves a small web UI that plays videos straight out of Immich and lets you mark
labelled time ranges with the keyboard. Labels are written as one JSON file per asset
under training-data/labels/, ready to be turned into clip-level training examples.

Run on the Immich host:

    python3 scripts/annotate_server.py --album "200.001.001 - Creampie Compilation"

then open http://localhost:8765/ (or http://wopr:8765/ from another machine).

Video bytes are proxied through this server so the Immich API key never reaches the
browser. HTTP Range requests are passed through, so scrubbing/seeking works normally.
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR = os.path.join(REPO, "training-data", "labels")

# Hotkeys 1-6. "creampie" is the target class; the rest are the confusable negatives that
# have actually bitten us -- those are scarcer and more valuable than the positives.
LABELS = ["creampie", "facial", "cum-on-tits", "anal-cum", "pullout-no-cum", "lube"]


def load_env():
    env = {}
    with open(os.path.join(REPO, ".env")) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env["IMMICH_URL"].rstrip("/"), env["IMMICH_API_KEY"]


IMMICH_URL, API_KEY = load_env()


def immich(path, body=None, method="GET"):
    req = urllib.request.Request(
        IMMICH_URL + path,
        data=json.dumps(body).encode() if body else None,
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
        method=method,
    )
    return json.loads(urllib.request.urlopen(req).read())


def album_id_by_name(name):
    for a in immich("/api/albums"):
        if a["albumName"] == name:
            return a["id"]
    raise SystemExit(f"album not found: {name}")


def album_videos(album_id):
    out, page = [], 1
    while True:
        d = immich("/api/search/metadata",
                   {"albumIds": [album_id], "type": "VIDEO", "page": page, "size": 500}, "POST")
        items = d.get("assets", {}).get("items", [])
        if not items:
            break
        for a in items:
            out.append({"id": a["id"], "name": a.get("originalFileName", a["id"]),
                        "duration_ms": a.get("duration")})
        nxt = d.get("assets", {}).get("nextPage")
        if nxt is None:
            break
        page = int(nxt)
    return out


VIDEOS = []

HTML = """<!doctype html><meta charset=utf-8><title>segment annotator</title>
<style>
 body{background:#141414;color:#ddd;font:14px/1.4 system-ui,sans-serif;margin:0;display:flex;height:100vh}
 #main{flex:1;display:flex;flex-direction:column;min-width:0}
 video{width:100%;flex:1;min-height:0;background:#000}
 #bar{padding:8px 12px;background:#1c1c1c;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 #side{width:330px;background:#1a1a1a;overflow:auto;border-left:1px solid #333}
 h3{margin:12px 12px 6px;font-size:12px;text-transform:uppercase;color:#888;letter-spacing:.5px}
 .vid{padding:6px 12px;cursor:pointer;border-bottom:1px solid #262626;font-size:12px}
 .vid:hover{background:#242424} .vid.on{background:#2d4a2d} .vid .n{color:#777;font-size:11px}
 .seg{padding:5px 12px;border-bottom:1px solid #262626;display:flex;justify-content:space-between;font-size:12px}
 .seg b{color:#7ec87e;font-weight:600}
 .x{color:#c66;cursor:pointer;padding:0 4px}
 kbd{background:#333;border-radius:3px;padding:1px 5px;font-size:11px}
 #pend{color:#e8c07d} #lbl{color:#7ec87e;font-weight:600}
 #help{padding:8px 12px;color:#777;font-size:11px;border-top:1px solid #333}
</style>
<div id=main>
  <video id=v controls preload=auto></video>
  <div id=bar>
    <span>t=<b id=t>0.00</b></span>
    <span>label: <span id=lbl>creampie</span></span>
    <span id=pend></span>
    <span id=status style="margin-left:auto;color:#777"></span>
  </div>
  <div id=help>
    <kbd>space</kbd> play/pause &nbsp; <kbd>&larr;/&rarr;</kbd> 1s &nbsp; <kbd>shift+&larr;/&rarr;</kbd> 10s &nbsp;
    <kbd>,</kbd>/<kbd>.</kbd> frame &nbsp; <kbd>[</kbd> start &nbsp; <kbd>]</kbd> end+save &nbsp;
    <kbd>1-6</kbd> label &nbsp; <kbd>n</kbd>/<kbd>p</kbd> next/prev video &nbsp; <kbd>esc</kbd> cancel
  </div>
</div>
<div id=side><h3>videos</h3><div id=vids></div><h3>segments</h3><div id=segs></div></div>
<script>
const LABELS=%LABELS%; let VIDS=[], cur=-1, segs=[], pend=null, lbl=LABELS[0];
const v=document.getElementById('v');
const $=(i)=>document.getElementById(i);
function fmt(s){const m=Math.floor(s/60),x=(s%60).toFixed(2).padStart(5,'0');return m+':'+x}
async function boot(){VIDS=await (await fetch('/api/videos')).json();drawVids();if(VIDS.length)load(0)}
function drawVids(){$('vids').innerHTML=VIDS.map((x,i)=>
  `<div class="vid${i==cur?' on':''}" onclick="load(${i})">${x.name}<div class=n>${x.nseg||0} segments</div></div>`).join('')}
async function load(i){
  if(cur>=0) await save();
  cur=i; pend=null; v.src='/video/'+VIDS[i].id;
  segs=await (await fetch('/api/labels/'+VIDS[i].id)).json();
  drawVids(); drawSegs(); $('status').textContent=VIDS[i].name;
}
function drawSegs(){
  segs.sort((a,b)=>a.start-b.start);
  $('segs').innerHTML=segs.map((s,i)=>
   `<div class=seg><span><b>${s.label}</b> ${fmt(s.start)} &rarr; ${fmt(s.end)}
    <span style=color:#666>(${(s.end-s.start).toFixed(1)}s)</span></span>
    <span class=x onclick="del(${i})">&times;</span></div>`).join('') || '<div class=seg style=color:#666>none yet</div>';
  if(cur>=0){VIDS[cur].nseg=segs.length;drawVids()}
}
async function save(){ if(cur<0)return;
  await fetch('/api/labels/'+VIDS[cur].id,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:VIDS[cur].name,duration_ms:VIDS[cur].duration_ms,segments:segs})});
}
function del(i){segs.splice(i,1);drawSegs();save()}
document.onkeydown=e=>{
  if(e.target.tagName=='INPUT')return;
  const k=e.key;
  if(k==' '){e.preventDefault(); v.paused?v.play():v.pause()}
  else if(k=='ArrowLeft'){e.preventDefault(); v.currentTime-=e.shiftKey?10:1}
  else if(k=='ArrowRight'){e.preventDefault(); v.currentTime+=e.shiftKey?10:1}
  else if(k==','){v.currentTime-=0.1} else if(k=='.'){v.currentTime+=0.1}
  else if(k=='['){pend=v.currentTime; $('pend').textContent='start '+fmt(pend)}
  else if(k==']'){ if(pend==null){$('pend').textContent='press [ first';return}
     const a=Math.min(pend,v.currentTime), b=Math.max(pend,v.currentTime);
     segs.push({label:lbl,start:+a.toFixed(2),end:+b.toFixed(2)}); pend=null;
     $('pend').textContent=''; drawSegs(); save()}
  else if(k=='Escape'){pend=null;$('pend').textContent=''}
  else if(k>='1'&&k<='6'){const i=+k-1; if(i<LABELS.length){lbl=LABELS[i];$('lbl').textContent=lbl}}
  else if(k=='n'&&cur<VIDS.length-1){load(cur+1)}
  else if(k=='p'&&cur>0){load(cur-1)}
};
v.ontimeupdate=()=>$('t').textContent=v.currentTime.toFixed(2);
window.onbeforeunload=save; boot();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p == "/":
            body = HTML.replace("%LABELS%", json.dumps(LABELS)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/videos":
            self._json(VIDEOS)
        elif p.startswith("/api/labels/"):
            aid = p.rsplit("/", 1)[1]
            fp = os.path.join(LABEL_DIR, aid + ".json")
            segs = json.load(open(fp))["segments"] if os.path.exists(fp) else []
            self._json(segs)
        elif p.startswith("/video/"):
            self._proxy(p.rsplit("/", 1)[1])
        else:
            self.send_error(404)

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        if not p.startswith("/api/labels/"):
            return self.send_error(404)
        aid = p.rsplit("/", 1)[1]
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        os.makedirs(LABEL_DIR, exist_ok=True)
        rec = {"asset_id": aid, "name": payload.get("name"),
               "duration_ms": payload.get("duration_ms"),
               "segments": payload.get("segments", [])}
        with open(os.path.join(LABEL_DIR, aid + ".json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        self._json({"ok": True, "n": len(rec["segments"])})

    def _proxy(self, asset_id):
        """Stream the original from Immich, passing Range through so seeking works."""
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", asset_id):
            return self.send_error(400)
        headers = {"x-api-key": API_KEY}
        rng = self.headers.get("Range")
        if rng:
            headers["Range"] = rng
        req = urllib.request.Request(f"{IMMICH_URL}/api/assets/{asset_id}/original", headers=headers)
        try:
            up = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            return self.send_error(e.code)
        self.send_response(up.status)
        for h in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
            if up.headers.get(h):
                self.send_header(h, up.headers[h])
        self.end_headers()
        try:
            while True:
                chunk = up.read(256 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser seeked away mid-stream; normal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--album", default="200.001.001 - Creampie Compilation")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    global VIDEOS
    VIDEOS = album_videos(album_id_by_name(args.album))
    os.makedirs(LABEL_DIR, exist_ok=True)
    done = sum(1 for v in VIDEOS
               if os.path.exists(os.path.join(LABEL_DIR, v["id"] + ".json")))
    print(f"album: {args.album}")
    print(f"videos: {len(VIDEOS)}  ({done} already have a label file)")
    print(f"labels -> {LABEL_DIR}")
    print(f"\n  http://localhost:{args.port}/   (or http://wopr:{args.port}/)\n")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
