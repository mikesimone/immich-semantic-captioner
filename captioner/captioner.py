#!/usr/bin/env python3
import io
import os
import sys
import time
import json
import re
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

import psycopg2
import psycopg2.extras


# ----------------------------
# Config
# ----------------------------
IMMICH_URL = os.environ.get("IMMICH_URL", "").rstrip("/")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "0.2"))
IDLE_SLEEP_SECONDS = int(os.environ.get("IDLE_SLEEP_SECONDS", "60"))

# Postgres (container network) - only used if USE_API_ONLY=false
PGHOST = os.environ.get("PGHOST", "immich_postgres")
PGPORT = int(os.environ.get("PGPORT", "5432"))
PGDATABASE = os.environ.get("PGDATABASE", "immich")
PGUSER = os.environ.get("PGUSER", "postgres")
PGPASSWORD = os.environ.get("PGPASSWORD", "")

# Behavior
MAX_CAPTION_CHARS = int(os.environ.get("MAX_CAPTION_CHARS", "900"))
USER_AGENT = os.environ.get("USER_AGENT", "immich-captioner/2.3")
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# If 0, videos are skipped and marked in captioner_skip.
CAPTION_VIDEOS = os.environ.get("CAPTION_VIDEOS", "0") == "1"

# Tagging: best-effort (won't crash if API changes)
ENABLE_TAGS = os.environ.get("ENABLE_TAGS", "0") == "1"  # default OFF until you want it

USE_API_ONLY = os.environ.get("USE_API_ONLY", "true").lower() == "true"
print(f"[config] USE_API_ONLY: {USE_API_ONLY}", flush=True)

# ----------------------------
# Identity rules (albums -> person)
# ----------------------------
# Format: IDENTITY_ALBUM_MAP="Lydia=Lydia;Me=Me;Meagan=Meagan"
IDENTITY_ALBUM_MAP = os.environ.get("IDENTITY_ALBUM_MAP", "Lydia=Lydia;Me=Me")
IDENTITY_NOUN_HINTS = os.environ.get(
    "IDENTITY_NOUN_HINTS",
    "Lydia=woman,girl,person;Me=man,guy,person;Meagan=woman,girl,person",
)
IDENTITY_ENSURE_MODE = os.environ.get("IDENTITY_ENSURE_MODE", "prefix").strip().lower()

def _parse_kv_map(spec: str, item_sep: str = ";") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in (spec or "").split(item_sep):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out

def _parse_noun_hints(spec: str, item_sep: str = ";") -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for chunk in (spec or "").split(item_sep):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k = k.strip()
        nouns = [x.strip() for x in v.split(",") if x.strip()]
        if k and nouns:
            out[k] = nouns
    return out

_IDENTITY_MAP = _parse_kv_map(IDENTITY_ALBUM_MAP)
_IDENTITY_HINTS = _parse_noun_hints(IDENTITY_NOUN_HINTS)

_IDENTITY_ALBUM_REGEXES: Dict[str, re.Pattern] = {
    album_kw: re.compile(rf"\b{re.escape(album_kw)}\b", re.IGNORECASE)
    for album_kw in _IDENTITY_MAP.keys()
}

_WS_REGEX = re.compile(r"\s+")

# ----------------------------
# Caption cleanup (watermarks, boilerplate)
# ----------------------------
_WATERMARK_PATTERNS = [
    r"\bimgflip\b", r"\bifunny(?:\.co)?\b", r"\bgfycat\b", r"\btenor\b", r"\bredgifs\b",
    r"\b9gag\b", r"\bmemedroid\b", r"\bknow your meme\b", r"\bmematic(?:\.net)?\b",
    r"\bmakeagif(?:\.com)?\b", r"\bmakeameme(?:\.org)?\b", r"\bimgflip\s+meme\s+maker\b",
    r"\bmeme\s+maker\b", r"\bmeme\s+generator\b", r"made\s+w(?:/|ith)\s+imgflip\s+meme\s+maker",
    r"made\s+with\s+imgflip", r"made\s+w(?:/|ith)\s+meme\s+maker", r"\bposted\s+in\s+r/[\w_]+\b",
    r"\br/[\w_]+\b", r"\breddit\b",
]
_WATERMARK_REGEXES = [re.compile(p, re.IGNORECASE) for p in _WATERMARK_PATTERNS]

_SEP_REGEX = re.compile(r"\s*[\|\u2022•·\-–—]+\s*")
_JUNK_FULLCAPTION_REGEXES = [re.compile(r"^watch and share .* gifs on gfycat$", re.IGNORECASE)]
_TRAILING_HANDLE_RE = re.compile(r"\s*@[\w.]+\s*$", re.IGNORECASE)

def clean_caption(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    for r in _JUNK_FULLCAPTION_REGEXES:
        if r.match(s):
            return ""
    s = _TRAILING_HANDLE_RE.sub("", s).strip()
    parts = [p.strip() for p in _SEP_REGEX.split(s) if p.strip()]
    if not parts:
        parts = [s]
    cleaned_parts: List[str] = []
    for p in parts:
        if any(r.search(p) for r in _WATERMARK_REGEXES):
            q = p
            for r in _WATERMARK_REGEXES:
                q = r.sub("", q)
            q = _WS_REGEX.sub(" ", q).strip(" -–—|•·")
            if q and len(q) >= 12:
                cleaned_parts.append(q)
            continue
        q = p
        for r in _WATERMARK_REGEXES:
            q = r.sub("", q)
        q = _WS_REGEX.sub(" ", q).strip(" -–—|•·")
        if q:
            cleaned_parts.append(q)
    out = " | ".join(cleaned_parts).strip()
    out = _WS_REGEX.sub(" ", out).strip()
    return out[:MAX_CAPTION_CHARS]

# ----------------------------
# Identity overrides
# ----------------------------
def extract_identities_from_albums(albums: List[str]) -> List[str]:
    found: List[str] = []
    for album in albums or []:
        if not album:
            continue
        for album_kw, rx in _IDENTITY_ALBUM_REGEXES.items():
            if rx.search(album):
                canonical = _IDENTITY_MAP.get(album_kw)
                if canonical:
                    found.append(canonical)
    seen = set()
    out: List[str] = []
    for name in found:
        k = name.lower()
        if k not in seen:
            seen.add(k)
            out.append(name)
    return out

def apply_identity_overrides(caption: str, albums: List[str]) -> Tuple[str, List[str]]:
    if not caption:
        return caption, []
    identities = extract_identities_from_albums(albums)
    if not identities:
        return caption, []
    out = caption
    for name in identities:
        nouns = _IDENTITY_HINTS.get(name) or _IDENTITY_HINTS.get(name.split()[0])
        if nouns:
            noun_alt = "|".join(re.escape(n) for n in nouns)
            out = re.sub(
                rf"\b(a|the)\s+([a-z]+\s+)?({noun_alt})\b",
                name,
                out,
                flags=re.IGNORECASE,
            )
    for name in identities:
        if not re.search(rf"\b{re.escape(name)}\b", out, flags=re.IGNORECASE):
            if IDENTITY_ENSURE_MODE == "suffix":
                out = f"{out} | {name}"
            else:
                out = f"{name}: {out}"
    out = _WS_REGEX.sub(" ", out).strip()[:MAX_CAPTION_CHARS]
    return out, identities

# ----------------------------
# Florence-2
# ----------------------------
def load_florence():
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch

    model_id = os.environ.get("FLORENCE_MODEL", "microsoft/Florence-2-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[model] Loading {model_id} on {device} dtype={compute_dtype}", flush=True)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    ).to(device)
    model.eval()

    def _run(prompt: str, pil_image: Image.Image) -> str:
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                v = v.to(device)
                if torch.is_floating_point(v):
                    v = v.to(dtype=compute_dtype)
                inputs[k] = v
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=256)
        txt = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return " ".join(txt.strip().split())[:MAX_CAPTION_CHARS]

    def ocr_is_meaningful(s: str) -> bool:
        if not s:
            return False
        t = s.strip()
        alnum = re.findall(r"[A-Za-z0-9]", t)
        if len(alnum) < 10:
            return False
        if len(t.split()) < 3:
            return False
        if re.match(r"^(a|an|the)\s+(man|woman|person|dog|cat|cartoon|photo|picture)\b", t, re.IGNORECASE):
            return False
        return True

    def caption_image(pil_image: Image.Image) -> Tuple[str, str]:
        ocr = _run("<OCR>", pil_image)
        if ocr_is_meaningful(ocr):
            return ocr, "OCR"
        detailed = _run("<MORE_DETAILED_CAPTION>", pil_image)
        return detailed, "DETAILED"

    return caption_image

# ----------------------------
# Immich API helpers
# ----------------------------
def must_env(name: str, val: str):
    if not val:
        print(f"[fatal] Missing required env var: {name}", file=sys.stderr, flush=True)
        sys.exit(2)

def immich_headers():
    return {
        "x-api-key": IMMICH_API_KEY,
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }

class ThumbnailNotFound(Exception):
    pass

def immich_get_thumbnail(asset_id: str) -> Image.Image:
    url = f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail"
    r = requests.get(url, headers=immich_headers(), timeout=120)
    if r.status_code == 404:
        raise ThumbnailNotFound(f"404 Not Found for url: {url}")
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def immich_update_description(asset_id: str, caption: str) -> bool:
    url = f"{IMMICH_URL}/api/assets"
    payload = {"ids": [asset_id], "description": caption}
    if DRY_RUN:
        print(f"[dryrun] Would update desc {asset_id} => {caption}", flush=True)
        return True
    r = requests.put(
        url,
        headers={**immich_headers(), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=60,
    )
    if r.status_code >= 300:
        print(f"[immich] PUT /api/assets failed {r.status_code}: {r.text}", flush=True)
        return False
    return True

# Tagging (unchanged)
_tag_cache: Dict[str, Optional[str]] = {}
_tag_list_cache: Optional[List[dict]] = None

def immich_list_tags() -> List[dict]:
    global _tag_list_cache
    if _tag_list_cache is not None:
        return _tag_list_cache
    url = f"{IMMICH_URL}/api/tags"
    r = requests.get(url, headers=immich_headers(), timeout=60)
    r.raise_for_status()
    _tag_list_cache = r.json()
    return _tag_list_cache

def immich_ensure_tag_id(tag_value: str) -> Optional[str]:
    if tag_value in _tag_cache:
        return _tag_cache[tag_value]
    try:
        tags = immich_list_tags()
        for t in tags:
            if str(t.get("value", "")).lower() == tag_value.lower():
                _tag_cache[tag_value] = str(t.get("id"))
                return _tag_cache[tag_value]
        url = f"{IMMICH_URL}/api/tags"
        payload = {"value": tag_value}
        r = requests.post(
            url,
            headers={**immich_headers(), "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        if r.status_code >= 300:
            print(f"[tag] create failed {r.status_code}: {r.text}", flush=True)
            _tag_cache[tag_value] = None
            return None
        created = r.json()
        _tag_list_cache = None
        _tag_cache[tag_value] = str(created.get("id")) if created.get("id") else None
        return _tag_cache[tag_value]
    except Exception as e:
        print(f"[tag] ensure_tag_id({tag_value}) failed: {e}", flush=True)
        _tag_cache[tag_value] = None
        return None

def immich_apply_tags(asset_id: str, tag_values: List[str]) -> None:
    if not ENABLE_TAGS:
        return
    try:
        tag_ids = [immich_ensure_tag_id(v) for v in tag_values]
        tag_ids = [t for t in tag_ids if t]
        if not tag_ids:
            return
        url = f"{IMMICH_URL}/api/tags/assets"
        payload = {"assetIds": [asset_id], "tagIds": tag_ids}
        r = requests.put(
            url,
            headers={**immich_headers(), "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        if r.status_code >= 300:
            print(f"[tag] apply failed {r.status_code}: {r.text}", flush=True)
    except Exception as e:
        print(f"[tag] apply failed: {e}", flush=True)

# ----------------------------
# API-only candidate fetch
# ----------------------------
def get_uncaptioned_candidates_api() -> List[Dict]:
    candidates = []
    page = 1
    skipped_in_memory = set()  # In-memory skips for this run (no persistence)

    while True:
        try:
            response = requests.post(
                f"{IMMICH_URL}/api/search/metadata",
                headers={"x-api-key": IMMICH_API_KEY, "Content-Type": "application/json"},
                json={
                    "withExif": True,
                    "page": page,
                    "size": BATCH_SIZE * 5,  # Larger page size for efficiency (adjust if rate-limited)
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])  # Immich uses "items" key

            if not items:
                break

            has_more = data.get("nextPage") is not None  # Reliable end-of-pagination check

            for item in items:
                asset_id = item.get("id")
                if asset_id in skipped_in_memory:
                    continue
                exif = item.get("exifInfo", {})
                desc = exif.get("description") if exif else None
                if not desc or desc.strip() == "":
                    candidates.append(item)
                    print(f"[api-candidate] Found uncaptioned: {asset_id}", flush=True)

            if not has_more:
                break

            page = data["nextPage"]
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"[api-error] Pagination failed on page {page}: {e}", flush=True)
            break

    print(f"[api] Found {len(candidates)} uncaptioned assets via API scan", flush=True)
    return candidates

def get_asset_albums(asset_id: str) -> List[str]:
    """Fetch album names for an asset via API (needed in API-only mode)"""
    try:
        url = f"{IMMICH_URL}/api/assets/{asset_id}"
        r = requests.get(url, headers=immich_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        return [album.get("albumName") for album in data.get("albums", []) if album.get("albumName")]
    except Exception as e:
        print(f"[api] Failed to fetch albums for {asset_id}: {e}", flush=True)
        return []

# ----------------------------
# Postgres helpers (ONLY used if not USE_API_ONLY)
# ----------------------------
def pg_connect():
    if not PGPASSWORD:
        raise RuntimeError("PGPASSWORD is empty. Set it in the captioner container environment.")
    conn = psycopg2.connect(
        host=PGHOST,
        port=PGPORT,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        connect_timeout=10,
    )
    conn.autocommit = True
    return conn

def pg_column_exists(conn, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s AND column_name=%s
            """,
            (table, column),
        )
        return cur.fetchone() is not None

def pg_ensure_skip_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS captioner_skip (
              asset_id uuid PRIMARY KEY,
              reason text NOT NULL,
              created_at timestamptz NOT NULL DEFAULT now()
            );
            """
        )

def pg_mark_skip(conn, asset_id: str, reason: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO captioner_skip(asset_id, reason)
            VALUES (%s, %s)
            ON CONFLICT (asset_id) DO UPDATE
              SET reason = EXCLUDED.reason
            """,
            (asset_id, reason),
        )

def pg_fetch_candidates(conn, limit: int) -> List[dict]:
    has_type = pg_column_exists(conn, "asset", "type")
    select_type = 'a."type",' if has_type else "NULL::text as type,"

    sql = f"""
    SELECT
      a.id as id,
      {select_type}
      ae.description as description,
      COALESCE(array_remove(array_agg(al."albumName"), NULL), '{{}}'::text[]) as albums
    FROM asset a
    JOIN asset_exif ae ON ae."assetId" = a.id
    LEFT JOIN captioner_skip cs ON cs.asset_id = a.id
    LEFT JOIN album_asset aa ON aa."assetId" = a.id
    LEFT JOIN album al ON al.id = aa."albumId"
    WHERE
      cs.asset_id IS NULL
      AND (ae.description IS NULL OR btrim(ae.description) = '')
    GROUP BY a.id, ae.description {', a."type"' if has_type else ''}
    ORDER BY a."createdAt" ASC
    LIMIT %s;
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        return list(cur.fetchall())

# ----------------------------
# Main loop
# ----------------------------
def main():
    must_env("IMMICH_URL", IMMICH_URL)
    must_env("IMMICH_API_KEY", IMMICH_API_KEY)

    caption_image = load_florence()

    conn = None
    if not USE_API_ONLY:
        print(f"[pg] Connecting to {PGHOST}:{PGPORT} db={PGDATABASE} user={PGUSER}", flush=True)
        conn = pg_connect()
        pg_ensure_skip_table(conn)
    else:
        print("[mode] Running in safe API-only mode (no DB access, no custom tables)", flush=True)

    total_done = 0

    while True:
        if USE_API_ONLY:
            candidates = get_uncaptioned_candidates_api()
        else:
            candidates = pg_fetch_candidates(conn, BATCH_SIZE)

        if not candidates:
            print(f"[done] No more blank assets. Sleeping {IDLE_SLEEP_SECONDS}s and rechecking...", flush=True)
            time.sleep(IDLE_SLEEP_SECONDS)
            continue

        print(f"[batch] {len(candidates)} candidates", flush=True)

        for row in candidates:
            if USE_API_ONLY:
                asset_id = row.get("id")
                asset_type = row.get("type", "UNKNOWN").upper()  # API may not have type; fallback
                albums = get_asset_albums(asset_id)  # Fetch separately
            else:
                asset_id = str(row["id"])
                asset_type = (row.get("type") or "").upper()
                albums = row.get("albums") or []

            try:
                if asset_type == "VIDEO" and not CAPTION_VIDEOS:
                    if not USE_API_ONLY:
                        pg_mark_skip(conn, asset_id, "SKIP_VIDEO")
                    print(f"[skip] {asset_id} is VIDEO (skipping)", flush=True)
                    continue

                img = immich_get_thumbnail(asset_id)
                raw_caption, mode = caption_image(img)

                caption = clean_caption(raw_caption)
                if not caption.strip():
                    if not USE_API_ONLY:
                        pg_mark_skip(conn, asset_id, "EMPTY_OR_JUNK_CAPTION")
                    print(f"[skip] {asset_id} produced empty/junk caption (marked skip)", flush=True)
                    continue

                caption, implied_tags = apply_identity_overrides(caption, albums)
                caption = " ".join(caption.split()).strip()[:MAX_CAPTION_CHARS]

                ok = immich_update_description(asset_id, caption)
                if ok:
                    total_done += 1
                    alb = ", ".join(albums[:3]) + ("..." if len(albums) > 3 else "")
                    print(f"[ok] {asset_id} [{mode}] albums=[{alb}] => {caption}", flush=True)

                    if implied_tags:
                        immich_apply_tags(asset_id, implied_tags)
                else:
                    print(f"[fail] {asset_id} update failed", flush=True)

                time.sleep(SLEEP_SECONDS)

            except ThumbnailNotFound as e:
                if not USE_API_ONLY:
                    pg_mark_skip(conn, asset_id, "THUMBNAIL_404")
                print(f"[skip] {asset_id}: {e} (marked skip)", flush=True)
                time.sleep(0.2)

            except Exception as e:
                print(f"[error] {asset_id}: {e}", flush=True)
                time.sleep(1.0)

        print(f"[progress] total updated this run: {total_done}", flush=True)


if __name__ == "__main__":
    main()
