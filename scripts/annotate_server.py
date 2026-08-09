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

# Hotkeys 1-7. "creampie" is the target class (cum inside / dripping out of her); the rest
# are the confusable negatives that have actually bitten us -- scarcer and more valuable
# than the positives. "cum-on-pussy" is external cum on the vulva, which looks very close
# to a creampie in a still frame and is exactly the distinction the classifier keeps
# getting wrong.
LABELS = ["creampie", "cum-on-pussy", "facial", "cum-on-tits", "anal-cum",
          "pullout-no-cum", "lube"]


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


def album_videos(album_id, album_name=""):
    out, page = [], 1
    while True:
        d = immich("/api/search/metadata",
                   {"albumIds": [album_id], "type": "VIDEO", "page": page, "size": 500}, "POST")
        items = d.get("assets", {}).get("items", [])
        if not items:
            break
        for a in items:
            out.append({"id": a["id"], "name": a.get("originalFileName", a["id"]),
                        "duration_ms": a.get("duration"), "album": album_name})
        nxt = d.get("assets", {}).get("nextPage")
        if nxt is None:
            break
        page = int(nxt)
    return out


VIDEOS = []

HTML = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>segment annotator</title>
<style>
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 body{background:#141414;color:#ddd;font:14px/1.4 system-ui,sans-serif;margin:0;
      display:flex;height:100dvh;overflow:hidden}
 #main{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0}
 video{width:100%;flex:1;min-height:0;background:#000}
 #side{width:330px;background:#1a1a1a;overflow:auto;border-left:1px solid #333;flex-shrink:0}
 h3{margin:12px 12px 6px;font-size:12px;text-transform:uppercase;color:#888;letter-spacing:.5px}
 .vid{padding:10px 12px;cursor:pointer;border-bottom:1px solid #262626;font-size:13px}
 .vid.on{background:#2d4a2d} .vid .n{color:#777;font-size:11px;margin-top:2px}
 .seg{padding:9px 12px;border-bottom:1px solid #262626;display:flex;
      justify-content:space-between;align-items:center;font-size:13px}
 .seg b{color:#7ec87e;font-weight:600}
 .x{color:#c66;cursor:pointer;padding:4px 10px;font-size:18px}
 .st{padding:3px 12px;display:flex;justify-content:space-between;font-size:12px}
 .st .c{color:#7ec87e;font-weight:600} .st.zero .c{color:#c66}
 #labels{padding:6px 8px;background:#181818;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
 .lb{background:#2a2a2a;border:1px solid #3a3a3a;color:#bbb;border-radius:6px;
     padding:9px 12px;cursor:pointer;font-size:13px;user-select:none;flex:1;text-align:center;
     min-height:42px;display:flex;align-items:center;justify-content:center;gap:5px}
 .lb.on{background:#2d5a2d;border-color:#5aa85a;color:#eaffea;font-weight:600}
 .lb .k{background:#111;border-radius:3px;padding:0 4px;font-size:10px;color:#888}
 .lb.on .k{color:#cfc}
 #transport{display:flex;gap:6px;padding:6px 8px;background:#1c1c1c}
 .tb{flex:1;min-height:52px;background:#262626;border:1px solid #3a3a3a;border-radius:6px;
     color:#ddd;font-size:15px;cursor:pointer;user-select:none;display:flex;
     align-items:center;justify-content:center;font-weight:600}
 .tb:active{background:#3a3a3a}
 #mark{display:flex;gap:8px;padding:0 8px 8px;background:#1c1c1c}
 .mk{flex:1;min-height:60px;border-radius:8px;border:none;color:#fff;font-size:17px;
     font-weight:700;cursor:pointer;user-select:none}
 #bStart{background:#2d6a2d} #bStart:active{background:#3d8a3d}
 #bEnd{background:#7a4a1a} #bEnd:active{background:#9a6a2a}
 #bEnd.armed{background:#a03030} #bEnd.armed:active{background:#c04040}
 #bCancel{flex:0 0 68px;background:#3a3a3a;font-size:22px;line-height:1}
 #bCancel:active{background:#555}
 #bCancel:disabled{opacity:.3}
 #bar{padding:6px 10px;background:#1c1c1c;display:flex;gap:12px;align-items:center;
      flex-wrap:wrap;font-size:13px}
 #pend{color:#e8c07d;font-weight:600}
 #help{padding:6px 10px;color:#666;font-size:11px;border-top:1px solid #333}
 kbd{background:#333;border-radius:3px;padding:1px 5px;font-size:11px}
 @media (max-width:900px){
   body{flex-direction:column;overflow:auto}
   #main{flex:none}
   video{height:36dvh;flex:none}
   #side{width:100%;border-left:none;border-top:1px solid #333;overflow:visible}
   #help{display:none}
   .lb{font-size:12px;padding:9px 6px;min-width:30%}
 }
</style>
<div id=main>
  <video id=v controls playsinline preload=auto></video>
  <div id=labels></div>
  <div id=transport>
    <div class=tb onclick="seek(-5)">&#171; 5s</div>
    <div class=tb onclick="seek(-1)">&#8249; 1s</div>
    <div class=tb onclick="seek(-0.1)">&#8249;</div>
    <div class=tb id=bPlay onclick="toggle()">&#9654;</div>
    <div class=tb onclick="seek(0.1)">&#8250;</div>
    <div class=tb onclick="seek(1)">1s &#8250;</div>
    <div class=tb onclick="seek(5)">5s &#187;</div>
  </div>
  <div id=mark>
    <button class=mk id=bStart onclick="markStart()">[ &nbsp;MARK START</button>
    <button class=mk id=bCancel onclick="cancelPend()" disabled>&#10007;</button>
    <button class=mk id=bEnd onclick="markEnd()">MARK END&nbsp; ]</button>
  </div>
  <div id=bar>
    <span>t=<b id=t>0.00</b></span>
    <span id=pend>tap MARK START at the beginning of an event</span>
    <span id=status style="margin-left:auto;color:#777"></span>
  </div>
  <div id=help>
    <kbd>space</kbd> play/pause &nbsp; <kbd>&larr;/&rarr;</kbd> 1s &nbsp; <kbd>shift+&larr;/&rarr;</kbd> 5s &nbsp;
    <kbd>,</kbd>/<kbd>.</kbd> 0.1s &nbsp; <kbd>[</kbd> start &nbsp; <kbd>]</kbd> end &nbsp;
    <kbd>1-7</kbd> label &nbsp; <kbd>n</kbd>/<kbd>p</kbd> video &nbsp; <kbd>esc</kbd> cancel
  </div>
</div>
<div id=side>
  <h3>label balance</h3><div id=stats></div>
  <h3>segments (this video)</h3><div id=segs></div>
  <h3>videos</h3><div id=vids></div>
</div>
<script>
const LABELS=%LABELS%; let VIDS=[], cur=-1, segs=[], pend=null, lbl=LABELS[0];
const v=document.getElementById('v');
const $=(i)=>document.getElementById(i);
function fmt(s){const m=Math.floor(s/60),x=(s%60).toFixed(2).padStart(5,'0');return m+':'+x}
async function boot(){drawLabels();drawStats();
  VIDS=await (await fetch('/api/videos')).json();drawVids();if(VIDS.length)load(0)}
function drawLabels(){$('labels').innerHTML=LABELS.map((L,i)=>
  `<div class="lb${L==lbl?' on':''}" onclick="pick('${L}')"><span class=k>${i+1}</span>${L}</div>`).join('')}
function pick(L){lbl=L;drawLabels()}
async function drawStats(){
  const d=await (await fetch('/api/stats')).json();
  $('stats').innerHTML=LABELS.map(L=>{const n=d.counts[L]||0;
    return `<div class="st${n?'':' zero'}"><span>${L}</span><span class=c>${n}</span></div>`}).join('')
    +`<div class=st style="border-top:1px solid #333;margin-top:4px;padding-top:5px">
       <span style=color:#777>videos labelled</span><span class=c>${d.videos_with_segments}</span></div>`;
}
function drawVids(){$('vids').innerHTML=VIDS.map((x,i)=>
  `<div class="vid${i==cur?' on':''}" onclick="load(${i})">${x.name}
     <div class=n>${(x.album||'').replace(/^[0-9.]+ - /,'')} &middot; ${x.nseg||0} seg</div></div>`).join('')}
async function load(i){
  if(cur>=0) await save();
  cur=i; setPend(null); v.src='/video/'+VIDS[i].id;
  segs=await (await fetch('/api/labels/'+VIDS[i].id)).json();
  drawVids(); drawSegs(); $('status').textContent=VIDS[i].name;
  window.scrollTo(0,0);
}
function drawSegs(){
  segs.sort((a,b)=>a.start-b.start);
  $('segs').innerHTML=segs.map((s,i)=>
   `<div class=seg><span><b>${s.label}</b> ${fmt(s.start)} &rarr; ${fmt(s.end)}
    <span style=color:#666>(${(s.end-s.start).toFixed(1)}s)</span></span>
    <span class=x onclick="del(${i})">&times;</span></div>`).join('')
    || '<div class=seg style=color:#666>none yet</div>';
  if(cur>=0){VIDS[cur].nseg=segs.length;drawVids()}
}
async function save(){ if(cur<0)return;
  await fetch('/api/labels/'+VIDS[cur].id,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:VIDS[cur].name,duration_ms:VIDS[cur].duration_ms,segments:segs})});
  drawStats();
}
function del(i){segs.splice(i,1);drawSegs();save()}
function cancelPend(){setPend(null); $('pend').textContent='start cancelled'}
function setPend(x){pend=x;
  $('bEnd').classList.toggle('armed', x!=null);
  $('bCancel').disabled = x==null;
  $('pend').textContent = x==null ? 'tap MARK START at the beginning of an event'
    : '\u25cf '+lbl+' started at '+fmt(x)+' \u2014 now tap MARK END';
}
function seek(d){v.currentTime=Math.max(0,v.currentTime+d)}
function toggle(){v.paused?v.play():v.pause()}
function markStart(){setPend(v.currentTime)}
function markEnd(){
  if(pend==null){$('pend').textContent='tap MARK START first';return}
  const a=Math.min(pend,v.currentTime), b=Math.max(pend,v.currentTime);
  segs.push({label:lbl,start:+a.toFixed(2),end:+b.toFixed(2)});
  setPend(null); $('pend').textContent='saved '+lbl+' '+fmt(a)+' \u2192 '+fmt(b);
  drawSegs(); save();
}
document.onkeydown=e=>{
  if(e.target.tagName=='INPUT')return;
  const k=e.key;
  if(k==' '){e.preventDefault(); toggle()}
  else if(k=='ArrowLeft'){e.preventDefault(); seek(e.shiftKey?-5:-1)}
  else if(k=='ArrowRight'){e.preventDefault(); seek(e.shiftKey?5:1)}
  else if(k==','){seek(-0.1)} else if(k=='.'){seek(0.1)}
  else if(k=='['){markStart()} else if(k==']'){markEnd()}
  else if(k=='Escape'){cancelPend()}
  else if(k>='1'&&k<='7'){const i=+k-1; if(i<LABELS.length){pick(LABELS[i])}}
  else if(k=='n'&&cur<VIDS.length-1){load(cur+1)}
  else if(k=='p'&&cur>0){load(cur-1)}
};
v.ontimeupdate=()=>$('t').textContent=v.currentTime.toFixed(2);
v.onplay=()=>$('bPlay').innerHTML='&#10073;&#10073;';
v.onpause=()=>$('bPlay').innerHTML='&#9654;';
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
        elif p == "/api/stats":
            counts, vids_done = {}, 0
            if os.path.isdir(LABEL_DIR):
                for fn in os.listdir(LABEL_DIR):
                    if not fn.endswith(".json"):
                        continue
                    try:
                        segs = json.load(open(os.path.join(LABEL_DIR, fn))).get("segments", [])
                    except (OSError, ValueError):
                        continue
                    if segs:
                        vids_done += 1
                    for sg in segs:
                        counts[sg["label"]] = counts.get(sg["label"], 0) + 1
            self._json({"counts": counts, "videos_with_segments": vids_done})
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
    ap.add_argument("--album", action="append", metavar="NAME",
                    help="repeatable; defaults to Compilation + Single + Multiple")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    albums = args.album or ["200.001.001 - Creampie Compilation",
                            "200.001.000 - Single Creampie",
                            "200.000.000 - Multiple Creampie"]
    global VIDEOS
    seen = set()
    for name in albums:
        got = album_videos(album_id_by_name(name), name)
        fresh = [v for v in got if v["id"] not in seen]
        seen.update(v["id"] for v in fresh)
        VIDEOS.extend(fresh)
        print(f"album: {name:<40} {len(got):>4} videos ({len(fresh)} new)")
    os.makedirs(LABEL_DIR, exist_ok=True)
    done = sum(1 for v in VIDEOS
               if os.path.exists(os.path.join(LABEL_DIR, v["id"] + ".json")))
    print(f"total: {len(VIDEOS)} unique videos ({done} already have a label file)")
    print(f"labels -> {LABEL_DIR}")
    print(f"\n  http://localhost:{args.port}/   (or http://wopr:{args.port}/)\n")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
