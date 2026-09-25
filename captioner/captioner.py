#!/usr/bin/env python3
import io
import os
import sys
import time
import json
import re
import subprocess
import tempfile
import threading
import queue
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

# psycopg2 is only needed for DB-direct mode (USE_API_ONLY=false) -- imported lazily
# inside pg_connect()/pg_fetch_candidates() instead of here, so API-only deployments
# (no DB access at all, e.g. a captioner instance running on a separate machine) don't
# need a Postgres client library installed.


# ----------------------------
# Config
# ----------------------------
IMMICH_URL = os.environ.get("IMMICH_URL", "").rstrip("/")
IMMICH_API_KEY = os.environ.get("IMMICH_API_KEY", "")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))
SLEEP_SECONDS = float(os.environ.get("SLEEP_SECONDS", "0.2"))

# Newly-uploaded assets can be picked up as candidates before Immich's thumbnail job has
# run, producing a 404 that's purely a timing issue rather than a real problem with the
# asset. Retry a few times with a short delay before giving up and marking it skip.
THUMBNAIL_RETRY_ATTEMPTS = int(os.environ.get("THUMBNAIL_RETRY_ATTEMPTS", "5"))
THUMBNAIL_RETRY_DELAY_SECONDS = float(os.environ.get("THUMBNAIL_RETRY_DELAY_SECONDS", "10"))

# How many DB candidates to fetch per poll before re-checking priority order (images
# before videos, newest-first). Deliberately small and separate from BATCH_SIZE --
# fetching a full BATCH_SIZE-sized batch up front would commit to working through it
# entirely (potentially many long videos) before ever re-querying, so newly-available
# higher-priority work (e.g. images cleared while a long video is mid-processing) would
# sit waiting instead of jumping the queue like it's supposed to.
DB_REPRIORITIZE_BATCH = int(os.environ.get("DB_REPRIORITIZE_BATCH", "1"))
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
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# Keyframe sampling: sparse coverage of the "setup" of the video, dense fixed-interval
# coverage of the tail window (where the action tends to concentrate in longer clips).
VIDEO_TAIL_SECONDS = float(os.environ.get("VIDEO_TAIL_SECONDS", "240"))
VIDEO_TAIL_INTERVAL_SECONDS = float(os.environ.get("VIDEO_TAIL_INTERVAL_SECONDS", "15"))
VIDEO_HEAD_FRAME_COUNT = int(os.environ.get("VIDEO_HEAD_FRAME_COUNT", "4"))
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "24"))

# For clips shorter than VIDEO_TAIL_INTERVAL_SECONDS * this many samples, the tail loop
# below would otherwise land on a single frame at t=0 -- often an establishing shot before
# the actual content, since the whole clip is shorter than one sampling interval. Shrink
# the interval so short clips still get multiple frames spread across their full duration.
VIDEO_MIN_SAMPLES = int(os.environ.get("VIDEO_MIN_SAMPLES", "4"))

# If 1, the detailed-caption model is instructed to describe sexual content
# directly and explicitly (no euphemisms) instead of writing a sanitized caption.
EXPLICIT_CAPTIONS = os.environ.get("EXPLICIT_CAPTIONS", "1") == "1"

# Sampling knobs for JoyCaption generation. Higher temperature = more varied phrasing
# (helps avoid the model looping on the same few words like "slutty"/"smutty").
JOYCAPTION_TEMPERATURE = float(os.environ.get("JOYCAPTION_TEMPERATURE", "0.75"))
JOYCAPTION_REPETITION_PENALTY = float(os.environ.get("JOYCAPTION_REPETITION_PENALTY", "1.15"))

# Batched GPU generation: N images processed in a single forward pass instead of one at a
# time. 0 = auto-calibrate against real GPU memory at startup (probes increasing batch
# sizes with a real generate() call at full max_new_tokens, backs off on CUDA OOM, then
# applies a safety margin) instead of guessing a fixed number that may not fit this
# GPU/model/VRAM-headroom combination.
MAX_GEN_BATCH = int(os.environ.get("MAX_GEN_BATCH", "0"))
GEN_BATCH_SAFETY_FACTOR = float(os.environ.get("GEN_BATCH_SAFETY_FACTOR", "0.75"))

# Albums whose videos get dense, uniform-interval frame sampling across the whole
# clip instead of head-sparse/tail-dense -- for compilation-style videos where multiple
# distinct events (e.g. creampies) can occur anywhere, not just near the end, sometimes
# back to back with little gap between them.
DENSE_SAMPLING_ALBUM_KEYWORDS = os.environ.get("DENSE_SAMPLING_ALBUM_KEYWORDS", "creampie")
DENSE_INTERVAL_SECONDS = float(os.environ.get("DENSE_INTERVAL_SECONDS", "2"))
DENSE_MAX_VIDEO_FRAMES = int(os.environ.get("DENSE_MAX_VIDEO_FRAMES", "120"))

# How far the container-declared duration may differ from the frame-count-derived one before
# the container is treated as wrong. 5% absorbs ordinary rounding and VFR jitter.
DURATION_MISMATCH_TOLERANCE = float(os.environ.get("DURATION_MISMATCH_TOLERANCE", "0.05"))

# Retained for the cleanup/reset scripts only. The captioner itself no longer files anything
# into Single Creampie -- creampie counts are reported in the caption and the human sorts.
SINGLE_CREAMPIE_ALBUM_ID = os.environ.get("SINGLE_CREAMPIE_ALBUM_ID", "3a22144e-143c-4f43-a508-8b3f7fadbcb5")

# Same idea for furry/anthro content -- any image or video whose caption indicates it,
# regardless of existing album, gets added here too.
FURRY_ALBUM_ID = os.environ.get("FURRY_ALBUM_ID", "b135f926-dd5b-4230-aa05-32bbdb2cf315")

# Same idea again for lactation (milk visibly coming from the nipples/breasts) and hucow
# (cow-print clothing/accessories) content.
LACTATION_ALBUM_ID = os.environ.get("LACTATION_ALBUM_ID", "2921493a-b6ba-4dcd-947c-d2fd3bd12f68")
HUCOW_ALBUM_ID = os.environ.get("HUCOW_ALBUM_ID", "d526cf69-8aed-4fdc-b93e-6dacafe7bec4")

# Album-routing configuration. Everything in the 200.000.xxx range is multiple-creampie
# content; CATEGORIZED_ALBUM_PREFIXES are the ranges the captioner knows how to handle, so
# nudity found outside them is parked at "Please categorize" for manual sorting instead of
# being guessed at.
MULTI_CREAMPIE_PREFIX = os.environ.get("MULTI_CREAMPIE_PREFIX", "200.000.")

# For single-creampie detection, ignore any CUM sighting before this fraction of the way
# through the video -- the creampie ends the scene, so an early one is a classifier
# confabulation rather than the real thing.
CREAMPIE_EARLIEST_FRACTION = float(os.environ.get("CREAMPIE_EARLIEST_FRACTION", "0.5"))

# Multi-creampie counting (see count_creampie_events): a return to the CUM state is a new
# creampie once this much time has passed since the previous counted one. The gap scales
# with the video's length (CREAMPIE_GAP_FRACTION of it), clamped to the min/max below.
CREAMPIE_MIN_GAP_SECONDS = float(os.environ.get("CREAMPIE_MIN_GAP_SECONDS", "8"))
CREAMPIE_MAX_GAP_SECONDS = float(os.environ.get("CREAMPIE_MAX_GAP_SECONDS", "90"))
CREAMPIE_GAP_FRACTION = float(os.environ.get("CREAMPIE_GAP_FRACTION", "0.1"))
# The one album whose members are guaranteed to hold multiples, so its count never reads
# below MULTI_CREAMPIE_MIN_COUNT. Other 200.000.x albums (studios etc.) aren't floored.
GUARANTEED_MULTI_ALBUM_NUMBER = os.environ.get("GUARANTEED_MULTI_ALBUM_NUMBER", "200.000.000")
MULTI_CREAMPIE_MIN_COUNT = int(os.environ.get("MULTI_CREAMPIE_MIN_COUNT", "2"))
# Which albums count as "the human already filed this", so it gets captioned instead of
# parked at "Please categorize". Empty (the default) means ANY album membership qualifies,
# which is the intent: freshly-imported porn lands in no album and needs sorting, while
# anything already sorted -- Lydia Captions (001.x), Me (002.x), the 100-500 porn sections,
# whatever numbering gets added later -- is captioned where it sits. Set to a comma-separated
# prefix list to restrict it.
CATEGORIZED_ALBUM_PREFIXES = tuple(
    p.strip()
    for p in os.environ.get("CATEGORIZED_ALBUM_PREFIXES", "").split(",")
    if p.strip()
)
FULL_CAPTION_ALBUM_KEYWORDS = [
    k.strip().lower()
    for k in os.environ.get("FULL_CAPTION_ALBUMS", "camspy,lv hookers").split(",")
    if k.strip()
]
UNCATEGORIZED_CAPTION = os.environ.get("UNCATEGORIZED_CAPTION", "Please Categorize")

# Assets shot on these EXIF make/model get auto-filed into Camspy.
CAMSPY_ALBUM_ID = os.environ.get("CAMSPY_ALBUM_ID", "64643582-0623-4bc2-931f-30149cbd6e45")
CAMSPY_EXIF_MAKE = os.environ.get("CAMSPY_EXIF_MAKE", "Meta").strip().lower()

# If 1, porn assets get their "Date Taken" stamped with the moment they were captioned, so
# freshly-processed material sorts to the top of the timeline for review. Deliberately gated
# on the caption *mode* (VIDEO-PORN-COMPACT -- i.e. nudity was actually detected) rather than
# on album membership, so it can never touch a family video that merely happens to sit in a
# broadly-named album.
STAMP_PORN_CAPTION_DATE = os.environ.get("STAMP_PORN_CAPTION_DATE", "1") == "1"

# Tagging: best-effort (won't crash if API changes)
ENABLE_TAGS = os.environ.get("ENABLE_TAGS", "0") == "1"  # default OFF until you want it

USE_API_ONLY = os.environ.get("USE_API_ONLY", "true").lower() == "true"
print(f"[config] USE_API_ONLY: {USE_API_ONLY}", flush=True)

# API-only mode: optionally restrict the metadata scan to one asset type server-side
# (e.g. "VIDEO"), so a second API-only instance can work a disjoint slice of the queue
# (e.g. videos only) without re-scanning or re-processing what a DB-direct instance
# elsewhere is already handling (images). Empty string = no filter, scan everything.
API_ASSET_TYPE_FILTER = os.environ.get("API_ASSET_TYPE_FILTER", "").strip().upper()

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
# Comma-separated album-name substrings where album membership alone settles the identity,
# so the "no person in this caption means it was misfiled" cleanup never fires -- see
# is_identity_authoritative_album().
IDENTITY_AUTHORITATIVE_ALBUM_KEYWORDS = os.environ.get(
    "IDENTITY_AUTHORITATIVE_ALBUM_KEYWORDS", "Lydia Dog"
)
# Numeric-prefix identity rule, for a whole numbered branch that depicts one person.
# IDENTITY_ALBUM_MAP can only match a name written inside the album title, which breaks down
# as soon as sibling albums are named after WHAT they show rather than WHO -- "300.006.001 -
# Doggystyle" has no name in it to match. A prefix entry says every album under that branch
# is that person by construction, the same shape of rule as MULTI_CREAMPIE_PREFIX. It also
# implies authoritative: a branch that only ever holds one character's renders has no
# "maybe it was misfiled" case to protect against.
# Format: IDENTITY_ALBUM_PREFIX_MAP="300.006.=LydiaDog"
IDENTITY_ALBUM_PREFIX_MAP = os.environ.get("IDENTITY_ALBUM_PREFIX_MAP", "300.006.=LydiaDog")

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
_IDENTITY_PREFIX_MAP = _parse_kv_map(IDENTITY_ALBUM_PREFIX_MAP)

def _identity_prefix_matches(album: str) -> List[str]:
    """Identities claimed by an album's numbered prefix, independent of its wording."""
    a = (album or "").strip()
    return [name for pfx, name in _IDENTITY_PREFIX_MAP.items() if a.startswith(pfx)]

_IDENTITY_HINTS = _parse_noun_hints(IDENTITY_NOUN_HINTS)

_IDENTITY_ALBUM_REGEXES: Dict[str, re.Pattern] = {
    album_kw: re.compile(rf"\b{re.escape(album_kw)}\b", re.IGNORECASE)
    for album_kw in _IDENTITY_MAP.keys()
}

# Second identity source: Immich's own People (face-recognition) tags, independent of the
# album-based system above -- covers ordinary photos (friends, coworkers) that never go near
# a named album. PEOPLE_NAME_OVERRIDES renames raw Immich Person names that shouldn't be
# injected verbatim (e.g. the owner's self-tag is literally named "Me" in the People UI, same
# as the "Me" album-filing shorthand, but should read as "Mike" in generated prose).
PEOPLE_NAME_OVERRIDES = os.environ.get("PEOPLE_NAME_OVERRIDES", "Me=Mike")
_PEOPLE_NAME_OVERRIDES = _parse_kv_map(PEOPLE_NAME_OVERRIDES)
INCLUDE_HIDDEN_PEOPLE = os.environ.get("INCLUDE_HIDDEN_PEOPLE", "0") == "1"

def canonical_people_name(raw_name: str) -> str:
    return _PEOPLE_NAME_OVERRIDES.get(raw_name, raw_name)

def get_asset_people_names(asset_id: str, raw: bool = False) -> List[str]:
    """Fetch named Immich People tagged on this asset (independent of album membership).
    GET /api/assets/{id} includes a "people" array with name/isHidden per tagged face --
    verified empirically to be populated there even though /api/search/metadata's own
    "people" field is not. raw=True returns the names as spelled in Immich, before
    PEOPLE_NAME_OVERRIDES -- upload routing matches those against album titles."""
    try:
        r = requests.get(f"{IMMICH_URL}/api/assets/{asset_id}", headers=immich_headers(), timeout=30)
        r.raise_for_status()
        people = r.json().get("people") or []
        out: List[str] = []
        seen = set()
        for p in people:
            name = (p.get("name") or "").strip()
            if not name:
                continue
            if p.get("isHidden") and not INCLUDE_HIDDEN_PEOPLE:
                continue
            cname = name if raw else canonical_people_name(name)
            key = cname.lower()
            if key not in seen:
                seen.add(key)
                out.append(cname)
        return out
    except Exception as e:
        print(f"[api] Failed to fetch people for {asset_id}: {e}", flush=True)
        return []

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

# Bare ASCII hyphens only count as a separator when surrounded by whitespace (" - "),
# so hyphenated compound words ("close-up", "dark-skinned") survive intact. Pipes/bullets/
# dashes always split, since those are never legitimately part of a normal word.
_SEP_REGEX = re.compile(r"\s*[\|\u2022•·–—]+\s*|\s+-\s+")
_JUNK_FULLCAPTION_REGEXES = [re.compile(r"^watch and share .* gifs on gfycat$", re.IGNORECASE)]
_TRAILING_HANDLE_RE = re.compile(r"\s*@[\w.]+\s*$", re.IGNORECASE)

# Sentence-level junk filter, reused for a few unrelated failure modes: the model
# narrating a watermark/site-name/URL despite being told not to (e.g. "The watermark
# 'Princess69.com' is in the top right corner", "OnlyFans URL is visible at the bottom"),
# and declaring the *absence* of nudity/sexual content instead of just not bringing it up
# (e.g. "No nudity or sexual content is depicted" on an otherwise-normal meme image --
# pure bloat that also crowds out anything actually useful for search). Split on real
# sentence boundaries (period-followed-by-space) rather than every period, since site
# domains like "OnlyFans.com" contain a period with no following space -- a naive
# per-period split would chop the sentence there and leave a dangling ".com/whatever"
# fragment behind.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_JUNK_SENTENCE_RE = re.compile(
    r"\b(?:watermarks?|logos?|onlyfans|url)\b"
    r"|\b[a-z0-9][a-z0-9-]*\.(?:com|net|org|co|xyz|vip|me|tv)\b"
    r"|\bno\b.{0,40}\b(?:nudity|nude|sexual content|explicit content|explicit acts?|"
    r"genitalia|genitals?)\b"
    r"|\b(?:nudity|nude|sexual content|explicit content|genitalia|genitals?)\b.{0,30}"
    r"\b(?:not|isn't|is\s+not)\b.{0,20}\b(?:present|depicted|shown|visible|apply)\b"
    r"|\bnote\s*:.{0,100}\b(?:guidelines?|instructions?|described|treated as such|"
    r"based on the description)\b",
    re.IGNORECASE,
)

def _strip_watermark_sentences(s: str) -> str:
    sentences = _SENTENCE_SPLIT_RE.split(s)
    kept = [sent for sent in sentences if not _JUNK_SENTENCE_RE.search(sent)]
    return " ".join(kept).strip()

# Backstop for the model leaking meta-commentary about its own instructions into the
# caption (e.g. "(Note: this is a non-sexual illustration and should be described
# accordingly.)").
_META_NOTE_PAREN_RE = re.compile(r"\(\s*note\s*:?[^)]*\)", re.IGNORECASE)

# Backstop for the model opening with meta-commentary about the caption itself instead of
# describing the image (e.g. "A smutty, degrading caption for the image: ...").
_META_PREAMBLE_RE = re.compile(
    r"^(a|an)\s+[\w,\s-]{0,40}\bcaption\b[\w\s]{0,20}\b(for|of)\s+(this|the)\s+"
    r"(?:[a-z-]+\s+){0,2}(image|video|photo|frame)[^:]{0,10}:\s*",
    re.IGNORECASE,
)

# Backstop for generic insult-labels ("slut", "whore", "slutty", "smutty") that carry no
# searchable information -- strip the adjective form and swap the noun form for something
# neutral rather than leaving a dangling article ("a " with nothing after it).
_BANNED_LABEL_PHRASE_RE = re.compile(r"\b(a|an|the)\s+(?:slutty|smutty)\s+", re.IGNORECASE)
_BANNED_NOUN_RE = re.compile(r"\b(?:sluts?|whores?)\b", re.IGNORECASE)
_BANNED_ADJ_RE = re.compile(r"\b(?:slutty|smutty)\b\s*", re.IGNORECASE)

# Backstop for the dead "Photograph of"/"Image of" opener ("no shit it's a photo, it's an
# image server"). Only strips a BARE opener right at the start of the string or right after
# a "[TS] " video-frame marker -- a real shot-type prefix like "Close-up photograph of" is
# left alone since "Close-up" carries real information.
_LEADING_PHOTO_PHRASE_RE = re.compile(
    r"(^|\]\s)(?:this\s+is\s+)?(?:an?\s+)?(?:photographs?|photos?|images?|pictures?)\s+"
    r"(?:of|showing|depicting)\s+",
    re.IGNORECASE,
)

# Backstop for clinical/textbook anatomy words the prompt explicitly forbids -- swap for
# the crude equivalent rather than leaving the clinical word in when the model slips.
_CLINICAL_TERM_SWAPS = [
    (re.compile(r"\bbuttocks?\b", re.IGNORECASE), "ass"),
    (re.compile(r"\banus\b", re.IGNORECASE), "asshole"),
    (re.compile(r"\bvulvas?\b", re.IGNORECASE), "pussy"),
    (re.compile(r"\blabia\b", re.IGNORECASE), "pussy lips"),
]

def clean_caption(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    for r in _JUNK_FULLCAPTION_REGEXES:
        if r.match(s):
            return ""
    s = _META_PREAMBLE_RE.sub("", s).strip()
    s = _META_NOTE_PAREN_RE.sub("", s).strip()
    s = _LEADING_PHOTO_PHRASE_RE.sub(lambda m: m.group(1), s)
    s = _strip_watermark_sentences(s)
    s = _BANNED_LABEL_PHRASE_RE.sub(lambda m: f"{m.group(1)} ", s)
    s = _BANNED_NOUN_RE.sub("woman", s)
    s = _BANNED_ADJ_RE.sub("", s)
    for pattern, replacement in _CLINICAL_TERM_SWAPS:
        s = pattern.sub(replacement, s)
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
    if out:
        out = out[0].upper() + out[1:]
    return out[:MAX_CAPTION_CHARS]

# ----------------------------
# Generation info (local render pipeline metadata riding in EXIF ImageDescription)
# ----------------------------
# Images from the local generation pipeline carry their generation parameters -- checkpoint,
# LoRAs and strengths, seed, sampler, the raw positive prompt -- as a JSON blob in EXIF
# ImageDescription. Immich copies that field verbatim into asset_exif.description, which is
# the SAME field this captioner writes captions to, so a naive caption write destroys the
# only record of how the image was made. That record is not reliably recoverable from the
# file either: the pipeline's own PNG/JPEG uploads keep it, but anything that arrives via
# Signal has had its embedded metadata stripped in transport.
#
# So a description is treated as two slots joined by GEN_INFO_SEPARATOR: prose caption
# first (that's what semantic search matches on), generation JSON last, byte-for-byte
# unchanged. Assets whose description holds nothing BUT generation JSON still count as
# uncaptioned -- see description_is_captionable() -- which is what finally gets the
# pipeline's own uploads captioned instead of skipped forever by the "non-empty description
# means already captioned" rule they used to trip.
GEN_INFO_SEPARATOR = "\n\n--- generation info ---\n"

# Keys that mark a JSON blob as OUR generation metadata rather than some unrelated JSON a
# camera or another tool happened to leave in ImageDescription. Requiring a real key hit
# (rather than just "it parses as JSON") keeps a stray payload from being silently retained
# as if it were generation info -- and, more importantly, keeps it from being treated as an
# uncaptioned asset and re-queued on every single pass.
_GEN_INFO_KEYS = frozenset({
    "checkpoint", "lora", "loras", "lora_path", "character_lora",
    "seed", "sampler", "scheduler", "steps", "cfg", "prompt", "class_type",
})

def parse_generation_info(text: str) -> Optional[str]:
    """Return the generation-info JSON when `text` is exactly that and nothing else."""
    if not text:
        return None
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    if _GEN_INFO_KEYS & {str(k) for k in obj}:
        return s
    # ComfyUI full-workflow dumps are keyed by node id ("1", "17", ...) with the real
    # markers one level down, so check the values before giving up.
    for v in obj.values():
        if isinstance(v, dict) and _GEN_INFO_KEYS & {str(k) for k in v}:
            return s
        if isinstance(v, dict) and isinstance(v.get("inputs"), dict):
            if _GEN_INFO_KEYS & {str(k) for k in v["inputs"]}:
                return s
    return None

def split_description(desc: Optional[str]) -> Tuple[str, Optional[str]]:
    """Split a stored description into its (caption, generation_info) slots."""
    if not desc:
        return "", None
    if GEN_INFO_SEPARATOR in desc:
        caption, _, gen = desc.partition(GEN_INFO_SEPARATOR)
        return caption.strip(), (gen.strip() or None)
    gen = parse_generation_info(desc)
    if gen is not None:
        return "", gen
    return desc.strip(), None

def compose_description(caption: str, gen_info: Optional[str]) -> str:
    """Join a caption back together with the generation info it must not lose."""
    if not gen_info:
        return caption
    if not caption:
        return gen_info
    return f"{caption}{GEN_INFO_SEPARATOR}{gen_info}"

def description_is_captionable(desc: Optional[str]) -> bool:
    """True when an asset still needs a caption: either no description at all, or one that
    holds nothing but generation info."""
    caption, _ = split_description(desc)
    return not caption

# ----------------------------
# Identity overrides
# ----------------------------
# One album title can match several identity tokens when one token is a longer form of
# another ("300.000.006 - Lydia Dog" matches both "Lydia Dog" and "Lydia"), which would
# otherwise inject two names into the same caption. The longer token is the deliberate, more
# specific filing -- that album is Lydia stylized as a dog, not an ordinary Lydia photo -- so
# a match whose token sits inside a longer matched token is dropped. Two unrelated names
# co-occurring in one title ("Lydia and Meagan") are unaffected: neither contains the other.
def _identities_for_album(album: str) -> List[str]:
    matched_kws = [kw for kw, rx in _IDENTITY_ALBUM_REGEXES.items() if rx.search(album)]
    out = [
        _IDENTITY_MAP[kw]
        for kw in matched_kws
        if _IDENTITY_MAP.get(kw)
        and not any(
            len(other) > len(kw) and kw.lower() in other.lower()
            for other in matched_kws
        )
    ]
    # A numbered-branch prefix claims its identity regardless of how the album is worded,
    # so these are added on top of any name actually spelled out in the title. Every caller
    # that resolves album identities goes through here, so they all pick this up.
    for name in _identity_prefix_matches(album):
        if name not in out:
            out.append(name)
    return out

def extract_identities_from_albums(albums: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for album in albums or []:
        if not album:
            continue
        for name in _identities_for_album(album):
            k = name.lower()
            if k not in seen:
                seen.add(k)
                out.append(name)
    return out

def find_albums_matching_identity(albums: List[str], identity_name: str) -> List[str]:
    matches: List[str] = []
    for album in albums or []:
        if not album:
            continue
        if identity_name in _identities_for_album(album):
            matches.append(album)
    return matches

# If a caption doesn't reference a person at all (no pronoun, no generic person noun),
# an identity-matched album ("Me", "Lydia", etc.) is almost certainly a misfile -- there's
# no one in the frame to be that person. Broad on purpose: the goal is to catch "no person
# whatsoever" (a meme, a screenshot, an object), not to second-guess borderline captions.
_PERSON_WORD_RE = re.compile(
    r"\b(?:he|him|his|she|her|hers|they|them|their|man|men|woman|women|guy|girl|"
    r"person|people|individual|figure)\b",
    re.IGNORECASE,
)

# Generic person words are only half the vocabulary. The configured noun hints are, by
# definition, the words that stand in for a given identity, and plenty of them fall outside
# _PERSON_WORD_RE: explicit slang in the human albums, and animal nouns for an identity that
# is deliberately depicted as one ("Lydia Dog" captions read "a dog ...", with no person word
# anywhere). Counting a hint as a reference is what keeps those albums from being read as
# "nobody is depicted here" and then auto-emptied by the misfile cleanup.
def _caption_references_someone(caption: str, identities: List[str]) -> bool:
    if _PERSON_WORD_RE.search(caption):
        return True
    for name in identities:
        hints = _IDENTITY_HINTS.get(name) or _IDENTITY_HINTS.get(name.split()[0]) or []
        for noun in hints:
            if re.search(rf"\b{re.escape(noun)}\b", caption, re.IGNORECASE):
                return True
    return False

def is_identity_authoritative_album(albums: List[str]) -> bool:
    """Albums where filing IS the identity claim, overriding the misfile heuristic.

    The misfile check below exists for photo albums, where a caption with no person in it
    means the asset was filed wrong. That reasoning does not hold for a wholly generated
    character album: every asset in "300.000.006 - Lydia Dog" is her by construction -- the
    album is the render output, not a pile of photos someone sorted -- so a caption the
    heuristic fails to find a person in is a caption-wording problem, not evidence the
    asset does not belong. Treating it as a misfile there would drop the name AND pull the
    asset out of the album, which is precisely backwards.
    """
    # A prefix-mapped branch is authoritative by construction -- it only ever holds one
    # character's renders -- so it needs no keyword spelled out in the title to qualify.
    if any(_identity_prefix_matches(album) for album in (albums or [])):
        return True
    keywords = [k.strip().lower() for k in IDENTITY_AUTHORITATIVE_ALBUM_KEYWORDS.split(",")
                if k.strip()]
    if not keywords:
        return False
    return any(kw in (album or "").lower() for album in (albums or []) for kw in keywords)

def collapse_name_redundancy(caption: str, identities: List[str]) -> str:
    """Tidy up the two ways the noun substitution ends up saying a name twice.

    The prompt tells the model to use the name and it usually does -- then describes the
    subject with a generic noun phrase anyway, and substituting that phrase for the name
    leaves the name doubled. Two shapes turn up in practice:

      "LydiaDog, a dog, running forward"        -> "LydiaDog, LydiaDog, running forward"
      "an anthropomorphic dog character named LydiaDog"
                                                -> "LydiaDog character named LydiaDog"

    Both collapse to a single mention. Only adjacent repeats are touched -- a caption that
    legitimately names the person again in a later sentence is left alone.
    """
    out = caption
    for name in identities:
        esc = re.escape(name)
        # "NAME, NAME" / "NAME NAME" -- separated by nothing but punctuation and space.
        out = re.sub(rf"\b{esc}\b(?:\s*[,;:]?\s+{esc}\b)+", name, out, flags=re.IGNORECASE)
        # "NAME character named NAME" -- keep the intervening noun, drop the restatement.
        out = re.sub(rf"\b{esc}\b(\s+[a-z]+)?\s+named\s+{esc}\b",
                     lambda m: name + (m.group(1) or ""), out, flags=re.IGNORECASE)
        # Re-run the plain collapse: dropping the "named NAME" tail can leave the first
        # shape behind ("NAME, NAME character").
        out = re.sub(rf"\b{esc}\b(?:\s*[,;:]?\s+{esc}\b)+", name, out, flags=re.IGNORECASE)
    return out

def apply_identity_overrides(caption: str, albums: List[str]) -> Tuple[str, List[str], List[str]]:
    """Returns (updated_caption, implied_tags, misfiled_identities)."""
    if not caption:
        return caption, [], []
    identities = extract_identities_from_albums(albums)
    if not identities:
        return caption, [], []

    if not _caption_references_someone(caption, identities) \
            and not is_identity_authoritative_album(albums):
        # Nobody appears to be depicted at all -- don't force any identity name into the
        # caption. Flag every expected identity so the caller can clean up the misfile.
        return caption, [], identities

    out = caption
    for name in identities:
        nouns = _IDENTITY_HINTS.get(name) or _IDENTITY_HINTS.get(name.split()[0])
        if nouns:
            noun_alt = "|".join(re.escape(n) for n in nouns)
            # The trailing group absorbs a SECOND hint noun when the phrase stacks two of
            # them ("an anthropomorphic dog girl"). Without it the substitution consumes
            # only up to the first noun and leaves the second stranded on the name --
            # "LydiaDog girl with pink hair" instead of "LydiaDog with pink hair".
            out = re.sub(
                rf"\b(a|an|the)\s+([a-z]+\s+)?({noun_alt})(?:\s+(?:{noun_alt}))?\b",
                name,
                out,
                flags=re.IGNORECASE,
            )
    out = collapse_name_redundancy(out, identities)
    for name in identities:
        if not re.search(rf"\b{re.escape(name)}\b", out, flags=re.IGNORECASE):
            if IDENTITY_ENSURE_MODE == "suffix":
                out = f"{out} | {name}"
            else:
                out = f"{name}: {out}"
    out = _WS_REGEX.sub(" ", out).strip()[:MAX_CAPTION_CHARS]
    return out, identities, []

# ----------------------------
# Florence-2 (OCR only -- cheap pre-pass to catch text-heavy images/memes)
# ----------------------------
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

def load_florence_ocr():
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

    def ocr(pil_image: Image.Image) -> str:
        inputs = processor(text="<OCR>", images=pil_image, return_tensors="pt")
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

    return ocr

# ----------------------------
# JoyCaption (explicit detailed captions; images + video frames)
# ----------------------------
def _name_instruction(person_names: Optional[List[str]]) -> str:
    if not person_names:
        return ""
    who = " and ".join(person_names)
    verb = "is" if len(person_names) == 1 else "are"
    quoted = " / ".join(f"\"{n}\"" for n in person_names)
    return (
        f"\n\n{who} {verb} known by name -- always refer to them as {quoted} instead of "
        "\"the woman\"/\"the man\"/\"the girl\"/\"the guy\", even when the rest of the "
        "description is explicit."
    )

_COMMON_CAPTION_RULES = (
    "Regardless of your answer below, follow these rules:\n"
    "- Never use the words \"photograph\", \"photo\", \"image\", or \"picture\" ANYWHERE in the "
    "caption, not even glued onto a shot-type prefix. This is an image server -- of course it's "
    "a photo. Bad: \"Close-up photograph of a woman...\", \"A photograph taken at eye level "
    "showing...\". Good: \"Close-up, eye level: a woman...\" or just start straight on the "
    "subject, e.g. \"A woman kneeling on a bed...\".\n"
    "- Don't describe the background, room, furniture, or setting at all, and don't mention "
    "lighting, light source/direction, or shadows, unless a specific object is being used "
    "sexually or is otherwise essential to the action. Skip camera angle as an aesthetic "
    "judgment, JPEG artifacts, camera/lens settings (ISO, aperture, shutter speed, focal "
    "length), depth of field, and subjective aesthetic quality (\"striking\", \"warm ambiance\", "
    "\"well-composed\") entirely -- none of that is searchable.\n"
    "- Shot framing and vantage height (close-up, low-angle, etc.) are minor details -- if you "
    "mention them at all, keep it to two or three words, not a sentence. The content itself "
    "matters far more than how it was framed.\n"
    "- Describe clothing specifically: what garment type it is and exactly what it covers vs. "
    "leaves bare. If an item is visually distinctive enough to identify the brand (e.g. red-"
    "soled heels are Christian Louboutin), name it.\n"
    "- Always describe the face and facial expression (eye contact, mouth open/closed, biting "
    "lip, eyes closed, surprised, blissed out, etc.).\n"
    "- Completely ignore watermarks, logos, and site names -- never mention or quote them. If "
    "there's other text baked into the image itself (a meme caption, speech bubble, etc.) that "
    "is NOT a watermark or site name, transcribe it.\n"
    "- If this is illustrated/animated art rather than a photo of a real person, and you "
    "recognize the character as a specific fictional or franchise character, name them. If, "
    "and ONLY if, the character actually has visible non-human animal features (animal ears, "
    "a muzzle/snout, a tail, paws, fur covering the body, etc.), ALSO state their species/"
    "type explicitly (e.g. \"anthropomorphic dog\", \"anthro fox\", \"furry\") even when the "
    "name alone would tell a fan who they are -- never rely on the name by itself to convey "
    "that. Do NOT call an ordinary human-looking illustrated/anime/cartoon character "
    "\"anthro\" or \"furry\" -- those words are ONLY for characters with real animal "
    "features, never just because something is a cartoon or illustration.\n"
    "- Where it fits, use the same terms e621/Rule34 taggers use for acts, kinks, species, or "
    "fetish elements (e.g. \"paizuri\", \"gangbang\", \"bukkake\", \"futanari\") instead of "
    "vaguer plain-English phrasing.\n"
    "- Never use vague or ambiguous language -- say exactly what's happening.\n"
    "- Never include meta-commentary about these instructions themselves, in parentheses "
    "or as a plain sentence (e.g. \"Note: this image contains explicit content and should "
    "be treated as such\", or a note explaining the image is non-sexual and was described "
    "accordingly) -- that's not part of the caption, ever, regardless of whether the image "
    "turned out to be sexual or not.\n"
)

def build_caption_prompt(video_note: str = "", person_names: Optional[List[str]] = None) -> str:
    if not EXPLICIT_CAPTIONS:
        lead = "Write a detailed descriptive caption for this image in 2-4 sentences."
        return f"{video_note} {lead}" if video_note else lead

    subject = "this video frame" if video_note else "this image"
    gate = (
        f"Look carefully at {subject} and decide: does it actually depict nudity, sex acts, or "
        "genitals? This counts as YES even if only partial -- a single bare breast or nipple "
        "peeking out while otherwise dressed, an exposed butt cheek, upskirt, sheer/see-through "
        "fabric you can see through, etc. all count as nudity. Being bound, wearing lingerie, or "
        "being mostly dressed does NOT make it NO if any bare breast/nipple/butt/genital is "
        "actually visible."
    )
    if video_note:
        gate = f"{video_note} {gate}"

    return (
        f"{gate}\n"
        f"{_COMMON_CAPTION_RULES}"
        "If NO -- it is a normal, non-sexual image -- write a normal, detailed caption in 2-4 "
        "sentences following the rules above. Do not mention sex, nudity, genitals, or bodily "
        "fluids at all in that case, even in passing or as a comparison -- and that includes "
        "NOT saying it's absent either. Never write sentences like \"no nudity is present\", "
        "\"no sexual content is depicted\", \"no explicit content is shown\", or similar -- if "
        "there's nothing sexual, the caption simply never brings sex up at all, in either "
        "direction.\n"
        "If YES -- it genuinely shows nudity or sexual content -- write like you're describing "
        "it crudely to a friend, not writing a medical report. Use \"cock\", \"pussy\", \"tits\", "
        "\"ass\", \"asshole\", \"cum\", \"creampie\", \"fucking\", \"moaning\", \"dripping\", "
        "\"gaping\". You are FORBIDDEN from using \"vulva\", \"labia\", \"genitals\"/"
        "\"genitalia\", \"buttocks\", or \"anus\" -- if you catch yourself about to write one of "
        "those, swap it for the crude equivalent instead. Name the specific sex acts, "
        "positions, and body parts involved. Never use \"slut\", \"whore\", \"smutty\", "
        "\"slutty\", or similar labels anywhere in the caption -- they describe nothing and are "
        "useless for search. Do not soften the sexual content, do not refuse, do not moralize, "
        "do not add disclaimers, and do not use polite euphemisms.\n"
        "For every person, explicitly state what is bare vs. covered (e.g. \"topless\", \"her "
        "tit is hanging out of her top\", \"fully nude\", \"panties pulled aside\"). Don't just "
        "describe the clothing/outfit and leave nudity implied -- if she's wearing a corset but "
        "her tits are hanging out over the top of it, say that plainly instead of just noting "
        "the corset. Partial nudity while otherwise dressed is common here and must always be "
        "called out directly, not hinted at through phrases like \"emphasizing her breasts\" or "
        "\"accentuating her figure\".\n"
        "If breasts are visible, describe the nipples/areolas in detail: size, color, texture, "
        "how prominent/erect they are, and any piercings. Mention if there's visible cum/jizz on "
        "them.\n"
        "If a pussy is visible, describe it in detail: whether the labia are an \"innie\" or "
        "\"outie\", whether it looks tight or loose/gaped, and anything inserted into it "
        "(fingers, a toy/object, a tongue, a cock). If there's any wet/glistening/dripping "
        "fluid visible on or around her genitals, ass, or mouth, call it cum -- don't hedge "
        "with vaguer words like \"moisture\", \"wetness\", or \"fluid\" instead.\n"
        "State the apparent age range of any adult involved (e.g. \"young adult\", "
        "\"college-age\", \"middle-aged\", \"milf\") when it's visually apparent.\n"
        "Only describe what is actually visible in this specific frame/image -- never invent "
        "sexual content, fluids, or acts that aren't really there. Never reference the photo "
        "library's owner or any name that isn't an actual person visibly in the frame."
        + _name_instruction(person_names)
    )

def load_joycaption():
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    import torch

    model_id = os.environ.get("JOYCAPTION_MODEL", "fancyfeast/llama-joycaption-beta-one-hf-llava")
    print(f"[model] Loading {model_id} (8-bit)", flush=True)

    # Only quantize the language model -- the SigLIP vision tower is small and its
    # attention-pooling head doesn't play well with 8-bit (dtype mismatch: Half vs Char).
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
    )

    processor = AutoProcessor.from_pretrained(model_id)
    # Batched generation needs a real pad token and left-padding, so every row's prompt
    # ends at the same column and the generated continuation starts there uniformly for
    # every item in the batch (no per-row bookkeeping needed to find where each answer
    # begins).
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()

    def _raw_batch_generate(batch_items: List[dict]) -> List[str]:
        prompts = []
        images = []
        max_new_tokens = 1
        for item in batch_items:
            prompt = item.get("prompt_override")
            if prompt is None:
                prompt = build_caption_prompt(item.get("video_note", ""), item.get("person_names"))
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": prompt},
            ]
            prompts.append(processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True))
            images.append(item["pil_image"])
            max_new_tokens = max(max_new_tokens, item.get("max_new_tokens", 256))

        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # Classification/field-extraction prompts (video signal detection, the compact
        # person-description triplet) want a single consistent answer, not creative variety
        # -- temperature sampling was confirmed in testing to flip the answer to the exact
        # same frame between runs (e.g. GENITALS: Y vs. N on 5 back-to-back calls with
        # identical input), which is a real source of the inconsistent creampie counts seen
        # in testing, separate from any counting-logic issue. Narrative captioning still
        # wants temperature sampling (it's what avoids repetitive phrasing across a library),
        # so this only switches to greedy decoding when every item in the batch opts in via
        # "greedy": True -- callers never mix classification and narrative items in one batch.
        greedy = bool(batch_items) and all(item.get("greedy") for item in batch_items)
        with torch.inference_mode():
            if greedy:
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=JOYCAPTION_REPETITION_PENALTY,
                )
            else:
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=JOYCAPTION_TEMPERATURE,
                    top_p=0.9,
                    repetition_penalty=JOYCAPTION_REPETITION_PENALTY,
                )
        # Left-padding means every row's prompt occupies the same width, so the generated
        # continuation starts at the same column for every row -- no per-row offset math.
        gen_only = generate_ids[:, inputs["input_ids"].shape[1]:]
        out = []
        for row in gen_only:
            txt = processor.tokenizer.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            out.append(" ".join(txt.strip().split())[:MAX_CAPTION_CHARS])
        return out

    _batch_state = {"max_batch": 1}

    def caption_detailed_batch(batch_items: List[dict]) -> List[str]:
        if not batch_items:
            return []
        cap = _batch_state["max_batch"]
        if len(batch_items) <= cap:
            return _raw_batch_generate(batch_items)
        out: List[str] = []
        for i in range(0, len(batch_items), cap):
            out.extend(_raw_batch_generate(batch_items[i:i + cap]))
        return out

    def _calibrate_max_batch() -> int:
        if MAX_GEN_BATCH > 0:
            print(f"[calibrate] MAX_GEN_BATCH override: {MAX_GEN_BATCH}", flush=True)
            return MAX_GEN_BATCH

        # Solid-color stand-in at a generous resolution -- the vision tower resizes/crops
        # to a fixed input size regardless, so this exercises the same tensor shapes as a
        # real photo. Use the longest/most detailed real prompt (full explicit-caption
        # instructions) and the real max_new_tokens ceiling, so the probe reflects actual
        # worst-case memory use, not a lighter approximation of it.
        test_image = Image.new("RGB", (1024, 1024), color=(128, 64, 32))
        test_prompt = build_caption_prompt()

        found = 1
        size = 32
        while size >= 1:
            try:
                _raw_batch_generate(
                    [{"pil_image": test_image, "prompt_override": test_prompt, "max_new_tokens": 256}] * size
                )
                found = size
                break
            except torch.cuda.OutOfMemoryError:
                size = size // 2
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                size = size // 2
            finally:
                torch.cuda.empty_cache()

        safe = max(1, int(found * GEN_BATCH_SAFETY_FACTOR))
        print(
            f"[calibrate] Max viable generation batch ~{found}, using {safe} "
            f"({int(GEN_BATCH_SAFETY_FACTOR * 100)}% safety margin)",
            flush=True,
        )
        return safe

    _batch_state["max_batch"] = _calibrate_max_batch()

    def caption_detailed(
        pil_image: Image.Image,
        video_note: str = "",
        person_names: Optional[List[str]] = None,
        prompt_override: Optional[str] = None,
        max_new_tokens: int = 256,
        greedy: bool = False,
    ) -> str:
        return caption_detailed_batch([{
            "pil_image": pil_image,
            "video_note": video_note,
            "person_names": person_names,
            "prompt_override": prompt_override,
            "max_new_tokens": max_new_tokens,
            "greedy": greedy,
        }])[0]

    caption_detailed.batch = caption_detailed_batch
    caption_detailed.max_batch = _batch_state["max_batch"]
    return caption_detailed

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
    for attempt in range(1, THUMBNAIL_RETRY_ATTEMPTS + 1):
        r = requests.get(url, headers=immich_headers(), timeout=120)
        if r.status_code == 404:
            if attempt < THUMBNAIL_RETRY_ATTEMPTS:
                time.sleep(THUMBNAIL_RETRY_DELAY_SECONDS)
                continue
            raise ThumbnailNotFound(f"404 Not Found for url: {url} (after {attempt} attempts)")
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

def immich_set_date_taken_now(asset_id: str) -> None:
    """Stamp an asset's "Date Taken" (EXIF dateTimeOriginal) with the current time, so
    freshly-captioned porn sorts to the top of the timeline for review.

    Only dateTimeOriginal moves -- fileCreatedAt/localDateTime are NOT recomputed from it on
    this Immich version (v3.1.0, verified empirically), so takenAfter/takenBefore API search
    won't match on the new value even though the UI's date sort does reflect it.

    Safe against reprocessing loops: candidate selection keys off an empty description, and
    the DB-direct queue orders by the DB's own createdAt column, neither of which this
    touches.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    if DRY_RUN:
        print(f"[dryrun] Would set dateTaken {asset_id} => {now}", flush=True)
        return
    try:
        r = requests.put(
            f"{IMMICH_URL}/api/assets",
            headers={**immich_headers(), "Content-Type": "application/json"},
            data=json.dumps({"ids": [asset_id], "dateTimeOriginal": now}),
            timeout=30,
        )
        if r.status_code >= 300:
            print(f"[datestamp] {asset_id} failed {r.status_code}: {r.text}", flush=True)
    except Exception as e:
        print(f"[datestamp] {asset_id} failed: {e}", flush=True)

# ----------------------------
# Video handling (download original, sample frames via ffmpeg, caption each)
# ----------------------------
def immich_download_original(asset_id: str, dest_path: str) -> None:
    url = f"{IMMICH_URL}/api/assets/{asset_id}/original"
    with requests.get(url, headers=immich_headers(), timeout=300, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def probe_duration_seconds(video_path: str) -> float:
    """Real playable duration, cross-checked against the video stream's frame count.

    Some files declare a duration in their MP4 container header that is far shorter than the
    actual content, and ffprobe reports the header value (as does Immich, which ingests it).
    Confirmed case: a compilation declaring 409.34s that really decodes to 1268.30s -- 38,012
    frames at 29.97fps. Every sample timestamp is derived from this number, so believing the
    header meant only ever looking at the first 6:49 of a 21:08 video and never seeing two
    thirds of it.

    nb_frames / r_frame_rate is metadata too, but it comes from the stream rather than the
    container and disagrees in exactly the cases that matter. When the two disagree by more
    than DURATION_MISMATCH_TOLERANCE, trust whichever is longer -- under-sampling is the
    failure mode we're fixing, and over-estimating merely wastes a few ffmpeg seeks past the
    end, which return no frame and are dropped.

    Deliberately avoids a full decode: that is accurate but takes ~5s per file even at 170x.
    """
    result = subprocess.run(
        [FFPROBE_BIN, "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "format=duration",
         "-show_entries", "stream=nb_frames,r_frame_rate",
         "-of", "json", video_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        data = json.loads(result.stdout)
    except (ValueError, TypeError):
        return 0.0

    try:
        container = float((data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        container = 0.0

    streams = data.get("streams") or []
    frame_based = 0.0
    if streams:
        st = streams[0]
        try:
            num, den = (st.get("r_frame_rate") or "0/0").split("/")
            fps = float(num) / float(den) if float(den) else 0.0
            nb = float(st.get("nb_frames") or 0)
            if fps > 0 and nb > 0:
                frame_based = nb / fps
        except (TypeError, ValueError, ZeroDivisionError):
            frame_based = 0.0

    if container > 0 and frame_based > 0:
        longer, shorter = max(container, frame_based), min(container, frame_based)
        if shorter > 0 and (longer - shorter) / shorter > DURATION_MISMATCH_TOLERANCE:
            print(f"[duration] {os.path.basename(video_path)}: container says "
                  f"{container:.1f}s but {frame_based:.1f}s of frames -- using "
                  f"{longer:.1f}s", flush=True)
            return longer
    return container or frame_based or 0.0

def compute_video_timestamps(duration: float) -> List[float]:
    if duration <= 0:
        return [0.0]

    # Dense, fixed-interval coverage of the tail (last VIDEO_TAIL_SECONDS) -- this
    # window size/interval is constant regardless of total video length, so a 5-minute
    # video and a 2-hour video both get the same tight coverage of their final minutes.
    tail_start = max(0.0, duration - VIDEO_TAIL_SECONDS)
    tail_span = duration - tail_start
    tail_interval = VIDEO_TAIL_INTERVAL_SECONDS
    if tail_span < tail_interval * VIDEO_MIN_SAMPLES:
        tail_interval = max(1.0, tail_span / VIDEO_MIN_SAMPLES)

    tail_timestamps: List[float] = []
    t = tail_start
    while t < duration:
        tail_timestamps.append(round(t, 2))
        t += tail_interval
    if not tail_timestamps:
        tail_timestamps = [round(max(0.0, duration - 1), 2)]

    # Sparse coverage of whatever comes before the tail window (the "setup").
    head_timestamps: List[float] = []
    if tail_start > 0 and VIDEO_HEAD_FRAME_COUNT > 0:
        n = VIDEO_HEAD_FRAME_COUNT
        head_timestamps = [round(tail_start * (i + 1) / (n + 1), 2) for i in range(n)]

    timestamps = sorted(head_timestamps + tail_timestamps)

    # Safety cap for pathological configs (e.g. a tiny interval on a huge tail window).
    # Trim from the front first so the tail -- the part we care most about -- survives.
    if len(timestamps) > MAX_VIDEO_FRAMES:
        timestamps = timestamps[len(timestamps) - MAX_VIDEO_FRAMES:]

    return timestamps

def compute_dense_timestamps(duration: float) -> List[float]:
    if duration <= 0:
        return [0.0]
    interval = DENSE_INTERVAL_SECONDS
    if duration < interval * VIDEO_MIN_SAMPLES:
        interval = max(1.0, duration / VIDEO_MIN_SAMPLES)
    timestamps: List[float] = []
    t = 0.0
    while t < duration:
        timestamps.append(round(t, 2))
        t += interval
    if len(timestamps) > DENSE_MAX_VIDEO_FRAMES:
        # Thin evenly across the full duration rather than truncating -- for a dense
        # scan we care about coverage of the whole clip, not just one end of it.
        step = len(timestamps) / DENSE_MAX_VIDEO_FRAMES
        timestamps = [timestamps[int(i * step)] for i in range(DENSE_MAX_VIDEO_FRAMES)]
    return timestamps

def is_dense_sampling_album(albums: List[str]) -> bool:
    keywords = [k.strip().lower() for k in DENSE_SAMPLING_ALBUM_KEYWORDS.split(",") if k.strip()]
    if not keywords:
        return False
    for album in albums or []:
        al = album.lower()
        if any(kw in al for kw in keywords):
            return True
    return False

# Compilation-style albums (e.g. "Creampie Compilation") are the one place multiple
# distinct creampies genuinely happen within seconds of each other -- everywhere else that
# pattern is classifier jitter, so the minimum-gap dedup in count_creampie_events gets
# disabled only for these.
def is_compilation_album(albums: List[str]) -> bool:
    return any("compilation" in (album or "").lower() for album in (albums or []))

# A real feral-on-human creampie plays out over several minutes (mount, tie, knot), which
# defeats the event-counting heuristic entirely -- there's no reliable "scene break" signal
# to find within one continuous encounter. Per explicit instruction: just assume exactly one
# creampie for this content and skip multi-event counting altogether; a genuine feral
# gangbang (if one ever turns up) gets corrected by hand.
def is_feral_album(albums: List[str]) -> bool:
    return any("feral on human" in (album or "").lower() for album in (albums or []))

# Per explicit instruction: multi-event creampie counting is expensive to get right and
# error-prone (false positives from lube being misread as cum, scene-break ambiguity), so
# it's now opt-in rather than something the captioner tries to detect on its own. Every video
# is assumed to have at most one creampie UNLESS it's already been manually placed in the
# Multiple Creampie album -- the counting logic only runs for those. New multi-creampie
# uploads are handled by the human moving them into that album and clearing the description
# before the captioner ever sees them, not by the captioner guessing.
def is_single_creampie_album(albums: List[str]) -> bool:
    return any("single creampie" in (album or "").lower() for album in (albums or []))

def is_multiple_creampie_album(albums: List[str]) -> bool:
    # The whole 200.000.xxx range is multiple-creampie content: the base album plus its
    # studio/kink sub-albums (Puta Locura, Creampie Squad, Gangbang, Slutwife, Czech,
    # Orgy, Hentaied...). Being filed into any of them is the human saying "this is a
    # multi", so event counting runs for all of them.
    return any((album or "").strip().startswith(MULTI_CREAMPIE_PREFIX) for album in (albums or []))

def is_guaranteed_multi_album(albums: List[str]) -> bool:
    want = normalize_album_number(GUARANTEED_MULTI_ALBUM_NUMBER)
    return any(normalize_album_number(album_number(a) or "") == want for a in (albums or []))

# Albums that are ordinary life footage filed under a porn-ish heading -- Ray-Ban Meta
# glasses capture, mostly. Nudity may appear, but these want the full narrative caption
# rather than the compact porn field format.
def is_full_caption_album(albums: List[str]) -> bool:
    return any(
        any(kw in (album or "").lower() for kw in FULL_CAPTION_ALBUM_KEYWORDS)
        for album in (albums or [])
    )

def is_masturbation_album(albums: List[str]) -> bool:
    return any("masturbation" in (album or "").lower() for album in (albums or []))

# Has the human already filed this somewhere the captioner knows how to caption? Porn that
# hasn't been sorted yet gets parked at "Please categorize" instead of being guessed at --
# most creampie videos open with her fingering herself, so auto-routing between e.g.
# Masturbation and Single Creampie isn't reliable enough to do unattended.
def is_categorized_album(albums: List[str]) -> bool:
    named = [a for a in (albums or []) if (a or "").strip()]
    if not named:
        return False
    if not CATEGORIZED_ALBUM_PREFIXES:
        return True
    return any(a.strip().startswith(CATEGORIZED_ALBUM_PREFIXES) for a in named)

# Furry/anthro content shouldn't get a creampie count at all -- the whole detection was
# built and tuned around live-action photography (vagina location, cum vs. lube texture),
# none of which reliably translates to illustrated art. Keyed off existing Furry Stuff album
# membership rather than trying to detect it live for every frame.
def is_furry_album(albums: List[str]) -> bool:
    return any("furry stuff" in (album or "").lower() for album in (albums or []))

# Anthro Video is curated, 100% non-human illustrated content -- a much more trustworthy
# "this isn't a human" signal than Furry Stuff membership, which is contaminated with human
# videos the captioner auto-filed there off incidental caption words ("furry handcuffs" and
# friends). Verified: 97 of 106 Furry-Stuff-tagged videos sitting in Single Creampie were
# actually imported into the human Single Creampie folder.
def is_anthro_album(albums: List[str]) -> bool:
    return any("anthro video" in (album or "").lower() for album in (albums or []))

# Content that is definitionally not a human woman being creampied, so creampie detection
# and all human-category auto-filing are skipped for it.
def is_nonhuman_album(albums: List[str]) -> bool:
    return is_anthro_album(albums) or is_feral_album(albums) or is_furry_album(albums)

# Lactation/Hucow don't have dedicated album-name keyword functions elsewhere in this file
# (they're only referenced by ALBUM_ID for auto-filing), so check for their name keywords
# directly here.
_ADULT_ALBUM_KEYWORD_RE = re.compile(r"\blactation\b|\bhucow\b", re.IGNORECASE)

# This user's whole library follows a numbered top-level convention: 000-090 is ordinary life
# (pets, family, memes, travel, work, games) and 100-500 is entirely adult content (identity
# porn albums, creampie categories, furry/hucow, Camspy, Cartoon Porn, Hotwife Captions, LV
# Hookers, etc. -- see the full album listing). This is a far more complete signal than
# keyword-matching individual album names one at a time, which misses whole categories
# (Camspy, Cartoon Porn, Internet Titties) that were confirmed, via a dry run of
# strip_false_nudity_leaks, to contain real nudity descriptions the keyword-only version
# wrongly stripped. A nested album like "002.003.004 - GSX FY27" still correctly falls
# outside the range (leading "002" < 100) despite superficially resembling the porn-numbered
# branches.
_ADULT_NUMERIC_PREFIX_RE = re.compile(r"^(\d{3})\.")

def _has_adult_numeric_prefix(albums: List[str]) -> bool:
    for album in albums or []:
        m = _ADULT_NUMERIC_PREFIX_RE.match((album or "").strip())
        if m and int(m.group(1)) >= 100:
            return True
    return False

def is_adult_album(albums: List[str]) -> bool:
    return (
        _has_adult_numeric_prefix(albums)
        or bool(extract_identities_from_albums(albums))
        or is_multiple_creampie_album(albums)
        or is_single_creampie_album(albums)
        or is_masturbation_album(albums)
        or is_nonhuman_album(albums)
        or any(_ADULT_ALBUM_KEYWORD_RE.search(album or "") for album in (albums or []))
    )

# Sentences JoyCaption leaks into an otherwise non-explicit caption despite the prompt's own
# "if NO -- do not mention nudity/genitals/fluids at all, in either direction" rule -- both
# disclaimer-style negations ("no visible cum or sexual activity", "neither man's genitals
# are visible") and bare positive assertions about genitals/nipples/undress that have no
# business appearing on a fully-clothed subject ("his nipples are small and light-colored",
# "his pants are down slightly, exposing his ass"). This is intentionally broader than
# _JUNK_SENTENCE_RE above (which runs unconditionally on every caption, including genuinely
# explicit ones) -- it's only ever invoked from strip_false_nudity_leaks(), which gates it to
# non-adult albums and verifies the caption has no surviving genuine nudity vocabulary first.
_NUDITY_LEAK_SENTENCE_RE = re.compile(
    r"\b(?:nipples?|penis|penises|vaginas?|genitals?|genitalia)\b"
    r"|\bno\b.{0,40}\b(?:nudity|nude|sexual content|sexual activity|explicit content|"
    r"explicit acts?|genitalia|genitals?|cum)\b"
    r"|\b(?:nudity|nude|sexual content|explicit content|genitalia|genitals?)\b.{0,30}"
    r"\b(?:not|isn't|is\s+not|aren't|are\s+not)\b.{0,20}\b(?:present|depicted|shown|"
    r"visible|apply)\b"
    r"|\bneither\b.{0,25}\b(?:genitals?|genitalia|nipples?|penis)\b"
    r"|\b(?:pants|shorts|underwear|boxers)\b.{0,15}\bdown\b"
    r"|\bexposing\s+(?:his|her|their)\s+(?:ass|bare)\b",
    re.IGNORECASE,
)

# The vocabulary the prompt itself requires for a genuinely explicit caption (the "if YES"
# branch mandates "cock"/"pussy"/"cum"/etc., forbids clinical hedging) -- if any of it
# survives after stripping leak sentences, this caption is describing real nudity and should
# be left alone rather than risk deleting legitimate content.
_EXPLICIT_NUDITY_VOCAB_RE = re.compile(
    r"\b(?:cock|pussy|tits?|asshole|cum|creampie|fuck(?:ing|ed|s)?|moan(?:ing|ed|s)?|"
    r"dripping|gaping|nude|naked|topless|blowjob|handjob|masturbat\w*)\b",
    re.IGNORECASE,
)

def has_explicit_nudity_vocab(caption: str) -> bool:
    return bool(_EXPLICIT_NUDITY_VOCAB_RE.search(caption or ""))

def strip_false_nudity_leaks(caption: str, albums: List[str]) -> str:
    """For assets outside adult-content albums, strip JoyCaption sentences that leak nudity/
    genital-status commentary despite the prompt's own instruction never to mention it on
    non-explicit content. See _NUDITY_LEAK_SENTENCE_RE above for what counts."""
    if not caption or is_adult_album(albums):
        return caption
    sentences = _SENTENCE_SPLIT_RE.split(caption)
    kept = [sent for sent in sentences if not _NUDITY_LEAK_SENTENCE_RE.search(sent)]
    if len(kept) == len(sentences):
        return caption
    candidate = " ".join(kept).strip()
    if not candidate:
        return caption
    if has_explicit_nudity_vocab(candidate):
        return caption
    return candidate

def extract_video_frames(video_path: str, dense: bool = False) -> List[Tuple[float, Image.Image]]:
    duration = probe_duration_seconds(video_path)
    timestamps = compute_dense_timestamps(duration) if dense else compute_video_timestamps(duration)
    return extract_frames_at(video_path, timestamps)

def extract_frames_at(video_path: str, timestamps: List[float]) -> List[Tuple[float, Image.Image]]:
    frames: List[Tuple[float, Image.Image]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(tmpdir, f"frame_{i}.jpg")
            subprocess.run(
                [FFMPEG_BIN, "-y", "-ss", f"{ts:.2f}", "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", out_path],
                capture_output=True, timeout=60,
            )
            if os.path.exists(out_path):
                frames.append((ts, Image.open(out_path).convert("RGB").copy()))
    return frames

def format_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def count_creampie_events(
    frame_states: List[Tuple[float, str, bool]],
    min_gap_seconds: Optional[float] = None,
    min_count: int = 0,
) -> Tuple[int, List[str]]:
    # frame_states is already in chronological sample order: (timestamp, STATE,
    # partner_visible) per frame, where STATE is one of INSERTED/CUM/NONE. partner_visible is
    # kept in the tuple for callers but no longer used.
    #
    # Only reached for content filed as multiples (Multiple Creampie, a studio album, or a
    # 100.000.x counted album), so this errs on the high side on purpose (Mike, 2026-09-25:
    # "since we are ONLY running the counter on the multi folder, we can be more lenient").
    #
    # A new event is every switch back into CUM at least the gap after the previous counted
    # event. The gap is a tenth of the video's length, clamped to 8-90 s. Per-frame traces on 2026-09-25 showed why the earlier rules undercounted:
    # - The old "she was alone in some frame since the last event" gate almost never opened.
    #   PARTNER came back Y on 262 of 270 frames, and a 34-minute gangbang could never count
    #   past 1.
    # - The old 60 s minimum gap made four real creampies at 0:12, 0:21, 0:39 and 0:58 in a
    #   60-second clip count as 1. Eight seconds still separates 0:12 from 0:21.
    # CUMLOC comes back VAGINA on nearly every frame, so CUM effectively means "not inserted
    # right now with her genitals in view". Each return to it after penetration is the best
    # available proxy for a finish. A dedicated yes/no semen check was tried as a second pass
    # and never said yes on a real one, so it isn't used.
    #
    # Scored on 2026-09-25 against Mike's hand counts for eight Multiple Creampie videos
    # (1-9 minutes long, 2-7 creampies each). A fixed 8 s gap was off by 24 creampies in
    # total, mostly from one creampie flickering in and out of view every few seconds and
    # being counted again each time. A fixed 30 s gap was off by 15, because it merged the
    # four real ones in the 60-second clip. Scaling the gap to a tenth of the length was off
    # by 8. It's still a proxy; the temporal classifier in porn-classifier is the real fix.
    #
    # min_count floors the result for albums that guarantee multiples (only 200.000.000
    # Multiple Creampie does). The timestamps list stays as detected.
    if min_gap_seconds is None:
        length = frame_states[-1][0] if frame_states else 0.0
        min_gap_seconds = min(CREAMPIE_MAX_GAP_SECONDS,
                              max(CREAMPIE_MIN_GAP_SECONDS, length * CREAMPIE_GAP_FRACTION))
    event_starts: List[float] = []
    was_cum = False
    for ts, state, _partner_visible in frame_states:
        if state == "CUM":
            if not was_cum and (not event_starts or ts - event_starts[-1] >= min_gap_seconds):
                event_starts.append(ts)
            was_cum = True
        else:
            was_cum = False
    return max(len(event_starts), min_count), [format_ts(ts) for ts in event_starts]

# Video porn no longer gets a scene-by-scene narrative -- just a compact structured
# summary. This cheap per-frame classifier drives that: nudity presence (to decide porn
# vs. not), the existing INSERTED/CUM state machine (for creampie counting), plus bondage
# and real-animal-interspecies presence, all in one small batched pass per frame.
_VIDEO_SIGNAL_PROMPT = (
    "Look at this single video frame. Respond with ONLY the following, nothing else -- no "
    "other sentences or explanations.\n"
    "If this frame is just a text card, logo, watermark screen, or loading screen with no "
    "person shown, respond with exactly: TITLECARD\n"
    "Otherwise respond with exactly these six labeled lines, in this order:\n"
    "NUDITY: Y or N -- Y if any bare breast, bare nipple, bare butt, or genitals are visible "
    "right now, even partially (e.g. a nipple peeking out of clothing). Otherwise N.\n"
    "PARTNER: Y or N -- Y if a male sexual partner (any part of him -- body, hand, cock, "
    "etc.) is visible anywhere in this frame with her right now, even if not currently "
    "penetrating her. N if she alone is visible with no partner in frame at all.\n"
    "GENITALS: Y or N -- Y if her vagina/pussy is actually visible somewhere in this frame "
    "right now (even partially). N if her vagina is not in view at all -- e.g. this frame "
    "only shows her face, breasts, hands, or any other body part with her crotch out of "
    "frame or fully covered by clothing.\n"
    "INSERTED: Y or N -- Y if a cock or toy is actively penetrating her VAGINA right now. "
    "Anal penetration is N. Oral is N.\n"
    "CUMLOC: where a visible load of cum/jizz actually IS in this frame. Answer with exactly "
    "one of these words -- do not explain, do not judge whether it 'counts':\n"
    "  VAGINA -- cum is in, on, or dripping out of her vagina\n"
    "  FACE -- cum is on her face, lips, tongue, in her mouth, or in her hair\n"
    "  TITS -- cum is on her breasts, nipples, or chest\n"
    "  ASS -- cum is in, on, or dripping out of her ass/anus, or on her butt cheeks\n"
    "  BODY -- cum is somewhere else on her (stomach, back, thighs, hands, feet)\n"
    "  NONE -- there is no visible load of cum anywhere in this frame\n"
    "Report where the cum you can SEE is located. If cum is visible in more than one place, "
    "answer with whichever single location has the most cum. Three things that are NOT cum "
    "and must be answered NONE: (1) lubricant -- often thick and white like cum, but it stays "
    "smeared and coating the penis or toy shaft itself and tends to be present throughout the "
    "whole scene rather than appearing only after sustained thrusting; if the white fluid is "
    "clinging to the shaft, it's lube; (2) her own natural arousal wetness, which is thin, "
    "clear and glossy rather than opaque and white; (3) milk leaking or spraying from her "
    "nipples, which belongs to the LACTATING field below, never here. If GENITALS is N you "
    "cannot answer VAGINA, because her vagina isn't even in view.\n"
    "BOUND: Y or N -- Y if she is visibly tied up, chained, cuffed, or otherwise physically "
    "restrained right now. Otherwise N.\n"
    "SPECIES: NONE, or a single animal name (e.g. dog, horse) if a REAL (photographic, "
    "non-illustrated, non-anthropomorphic) animal is sexually engaging with her right now. "
    "Illustrated/anime/furry/anthro humanoid-animal characters do NOT count -- answer NONE "
    "for those, and NONE if only humans are involved.\n"
    "LACTATING: Y or N -- Y if milk is visibly leaking, dripping, or spraying from her bare "
    "nipples/breasts right now. Otherwise N.\n"
    "Example full answer:\nNUDITY: Y\nPARTNER: Y\nGENITALS: Y\nINSERTED: N\nCUMLOC: VAGINA\n"
    "BOUND: N\nSPECIES: NONE\nLACTATING: N"
)

def _parse_video_signal(text: str) -> dict:
    t = text.upper()
    if "TITLECARD" in t:
        return {
            "titlecard": True, "nudity": False, "partner_visible": False, "state": "NONE",
            "cum_location": None, "bound": False, "species": None, "lactating": False,
        }
    nudity = bool(re.search(r"NUDITY\s*:\s*Y", t))
    partner_visible = bool(re.search(r"PARTNER\s*:\s*Y", t))
    genitals_visible = bool(re.search(r"GENITALS\s*:\s*Y", t))
    inserted = bool(re.search(r"INSERTED\s*:\s*Y", t))

    # Ask the model *where* the cum is (a factual observation it's good at) and decide what
    # counts here in code, rather than asking it to apply the "only vaginal counts" rule
    # itself -- it kept ignoring that instruction and reporting facials and cum-on-tits as
    # creampies, which put them in Single Creampie.
    m = re.search(r"CUMLOC\s*:\s*(VAGINA|FACE|TITS|ASS|BODY|NONE)\b", t)
    cum_location = m.group(1).lower() if m else None

    if inserted:
        state = "INSERTED"
    elif cum_location == "vagina":
        state = "CUM"
    else:
        state = "NONE"

    # Defense in depth: the model doesn't reliably honor "you can't answer VAGINA when
    # GENITALS is N" -- confirmed earlier, a frame of milk streaming from a nipple with no
    # genitals in view still came back as a creampie. Enforcing it here doesn't depend on
    # the model following that instruction.
    if not genitals_visible and state == "CUM":
        state = "NONE"
    bound = bool(re.search(r"BOUND\s*:\s*Y", t))
    # Matched against a fixed vocabulary with a trailing word boundary rather than a bare
    # [A-Z]+ capture -- the model doesn't always put a line break between fields, and a
    # greedy capture would swallow the next label too (observed in testing: "SPECIES: NONE"
    # run straight into "LACTATING: Y" on the same line got captured whole as the species
    # "nonelactating"). A missed word boundary here just falls back to no species detected,
    # which is a far safer failure mode than a garbage species value.
    species = None
    m = re.search(r"SPECIES\s*:\s*(DOG|HORSE|WOLF|PIG|DONKEY|BULL|GOAT|SNAKE|CAT|FOX|BEAR|MONKEY)\b", t)
    if m:
        species = m.group(1).lower()
    lactating = bool(re.search(r"LACTATING\s*:\s*Y", t))
    return {
        "titlecard": False, "nudity": nudity, "partner_visible": partner_visible, "state": state,
        "cum_location": cum_location, "bound": bound, "species": species, "lactating": lactating,
    }

# Anthropomorphic/furry art -- deliberately keys off words our own prompt uses for
# illustrated animal-humanoid characters, not real animals (which would never be
# described this way under the "identify fictional/franchise character... if illustrated
# art" instruction), so this shouldn't fire on real-animal content.
_FURRY_TRIGGER_RE = re.compile(r"\banthro(?:pomorphic)?\b|\bfurry\b", re.IGNORECASE)

# Milk visibly coming from the nipples/breasts. Matches the direct e621-style term
# ("lactating"/"lactation") as well as descriptive phrasing that doesn't use that exact
# word ("milk leaking from her nipples", "nipples dripping milk"), by requiring "milk"
# and "nipple"/"breast"/"tit" to co-occur within the same clause rather than firing on
# "milk" alone (which shows up in plenty of non-lactation captions, e.g. "milky skin").
_LACTATION_TRIGGER_RE = re.compile(
    r"\blactat\w*\b"
    r"|\bmilk\w*\b[^.]{0,40}\b(?:nipple|breast|tit)\w*\b"
    r"|\b(?:nipple|breast|tit)\w*\b[^.]{0,40}\bmilk\w*\b",
    re.IGNORECASE,
)

# Cow-print clothing/accessories (bra, panties, ears, etc.) on a real person -- distinct
# from the furry/anthro trigger above, which is about illustrated animal-humanoid
# characters, not real people dressed in a cow-spotted pattern.
_HUCOW_TRIGGER_RE = re.compile(
    r"\bcow[\s-]?print\b|\bcow[\s-]?spot(?:ted|s)?\b|\bcow[\s-]?pattern(?:ed)?\b|\bhucow\b",
    re.IGNORECASE,
)

def _classify_video_frames(
    frames: List[Tuple[float, Image.Image]],
    caption_detailed,
) -> List[dict]:
    # All frames of one video share the exact same short classifier prompt, differing
    # only by image -- an ideal, low-risk batching target (no padding complexity from
    # varying prompt lengths) that turns up to DENSE_MAX_VIDEO_FRAMES sequential
    # single-frame generate() calls into a handful of batched ones.
    batch_fn = getattr(caption_detailed, "batch", None)
    if batch_fn:
        items = [
            {"pil_image": img, "prompt_override": _VIDEO_SIGNAL_PROMPT, "max_new_tokens": 100, "greedy": True}
            for _, img in frames
        ]
        raw = batch_fn(items)
    else:
        raw = [
            caption_detailed(img, prompt_override=_VIDEO_SIGNAL_PROMPT, max_new_tokens=100, greedy=True)
            for _, img in frames
        ]

    signals = []
    for (ts, img), text in zip(frames, raw):
        parsed = _parse_video_signal(text)
        parsed["ts"] = ts
        parsed["img"] = img
        signals.append(parsed)
    return signals

# What she's masturbating with -- only asked for content filed in the Masturbation album,
# since auto-detecting "this is a masturbation video" doesn't work (most creampie videos
# open with her fingering herself, so the signal is present in half the library).
_MASTURBATION_PROMPT = (
    "Look at this image. Is she masturbating, and if so what with? Answer with ONLY a short "
    "phrase, nothing else:\n"
    "- one of: hand, fingers, dildo, vibrator, buttplug\n"
    "- or, if it's some other object, a two-or-three word description of it (e.g. "
    "\"hairbrush handle\", \"shower head\")\n"
    "If she is not masturbating in this frame, answer exactly: NONE"
)

def _parse_masturbation_answer(text: str) -> Optional[str]:
    t = " ".join((text or "").split()).strip().strip(".")
    if not t or t.upper().startswith("NONE") or len(t) > 40:
        return None
    return t.lower()

# Compact structured description for video porn -- no scene narrative, just the handful
# of searchable facts that matter: breast size, race, and age of whichever woman/women are
# visible. Skipping Lydia entirely is handled by the caller (based on album identity), not
# here, since the model has no way to know who's who -- it just describes whoever it sees.
def build_compact_person_prompt() -> str:
    # Output is post-processed by collapsing all whitespace to single spaces (see
    # _raw_batch_generate), so newlines can't be relied on to separate multiple women --
    # use an explicit " | " delimiter between women instead, all on one line.
    return (
        "Look at this image. For every adult woman CLEARLY visible (ignore any men, and "
        "ignore anyone only partially/ambiguously in frame), give: "
        "<breast size>, <race>, <age>\n"
        "- breast size: one of small, medium, large, huge\n"
        "- race: your best guess (e.g. white, Black, Latina, Asian, Middle Eastern, mixed)\n"
        "- age: one of young adult, middle age, old\n"
        "If more than one woman is clearly visible, separate each woman's triplet with ' | ' "
        "-- e.g. \"large, white, young adult | small, Black, middle age\". If only one woman "
        "is clearly visible, give just her one triplet -- do not add extra entries for anyone "
        "who isn't clearly visible, and never write placeholder text like \"none visible\" or "
        "\"not visible\". Respond with ONLY the triplet(s) -- no names, no other commentary, "
        "no extra sentences."
    )

# Asset-identity names for whom we already know exactly who she is from album membership --
# describing her physically (breast size/race/age) is redundant, so we skip it entirely
# when she's the only person identified for this asset.
COMPACT_DESC_SKIP_NAMES = {"Lydia", "LydiaDog"}

_VALID_BREAST_SIZES = {"small", "medium", "large", "huge"}
_AGE_KEYWORDS = ("young", "middle", "old")

def _parse_person_desc(text: str) -> Tuple[List[str], List[str], List[str]]:
    breasts, races, ages = [], [], []
    for woman in text.split("|"):
        # The model doesn't always honor "no other commentary" -- occasionally tacks on a
        # parenthetical aside onto the last field (e.g. "middle age (Note: the visible skin
        # tone suggests...)"), or drops the triplet format entirely for one entry and writes
        # a free-form physical description instead (observed: "tattoo on right shoulder" in
        # the breast-size slot). Truncate at the first "(", and validate breast size/age
        # against their known-small vocabularies rather than trusting position alone --
        # race has no fixed vocabulary to check against, so it rides along with whichever
        # entries pass the other two checks.
        fields = [p.split("(")[0].strip() for p in woman.split(",")]
        if len(fields) < 3 or not all(fields[:3]):
            continue
        breast, race, age = fields[0], fields[1], fields[2]
        if breast.lower() not in _VALID_BREAST_SIZES:
            continue
        if not any(kw in age.lower() for kw in _AGE_KEYWORDS):
            continue
        breasts.append(breast)
        races.append(race)
        ages.append(age)
    return breasts, races, ages

def detect_single_creampie(signals: List[dict]) -> Optional[float]:
    """Timestamp of the one creampie in a video's dense signals, or None. See the reasoning
    in caption_video() where this is called for the at-most-one-creampie case."""
    last_ts = signals[-1]["ts"] if signals else 0.0
    earliest_plausible = last_ts * CREAMPIE_EARLIEST_FRACTION

    cum_ts = None
    seen_insertion = False
    seen_partner = False
    for s in signals:
        if s["partner_visible"]:
            seen_partner = True
        if s["state"] == "INSERTED":
            seen_insertion = True
        elif (s["state"] == "CUM" and seen_insertion and seen_partner
              and s["ts"] >= earliest_plausible):
            cum_ts = s["ts"]
    return cum_ts

def compact_porn_caption(
    signals: List[dict],
    frames: List[Tuple[float, Image.Image]],
    caption_detailed,
    person_names: Optional[List[str]] = None,
    masturbation: bool = False,
    count: int = 0,
    event_times: Optional[List[str]] = None,
) -> str:
    """The compact "Field | value" porn caption, built from a video's dense signals."""
    event_times = event_times or []
    bound_ever = any(s["bound"] for s in signals)
    species = next((s["species"] for s in signals if s["species"]), None)
    lactating_ever = any(s["lactating"] for s in signals)

    # Representative frame for the compact person description -- prefer an actually-nude
    # frame over an arbitrary one, and skip title cards/intro screens.
    nude_frames = [(s["ts"], s["img"]) for s in signals if not s["titlecard"] and s["nudity"]]
    candidates = nude_frames or frames
    _, desc_img = candidates[len(candidates) // 2]

    breasts, races, ages = [], [], []
    if not (person_names and set(person_names) <= COMPACT_DESC_SKIP_NAMES):
        person_desc = caption_detailed(
            desc_img, prompt_override=build_compact_person_prompt(), max_new_tokens=80, greedy=True
        ).strip()
        if person_desc:
            breasts, races, ages = _parse_person_desc(person_desc)

    # Only asked for Masturbation-album content -- see _MASTURBATION_PROMPT. Prefer a
    # frame with no partner in it, since that's where she'd actually be using something
    # on herself rather than being fucked.
    implement = None
    if masturbation:
        solo = [(s["ts"], s["img"]) for s in signals
                if not s["titlecard"] and s["nudity"] and not s["partner_visible"]]
        solo_candidates = solo or nude_frames or frames
        _, solo_img = solo_candidates[len(solo_candidates) // 2]
        implement = _parse_masturbation_answer(
            caption_detailed(solo_img, prompt_override=_MASTURBATION_PROMPT,
                             max_new_tokens=24, greedy=True)
        )

    # Labeled "Field | value" lines -- self-documenting on purpose, since a bare
    # comma-separated blob is meaningless to re-read weeks later.
    fields: List[Tuple[str, str]] = []
    if count >= 1:
        fields.append(("Separate Creampies", f"{count} (~{', '.join(event_times)})"))
    if breasts:
        fields.append(("Breast Size", ", ".join(breasts)))
    if races:
        fields.append(("Race", ", ".join(races)))
    if ages:
        fields.append(("Approximate Age", ", ".join(ages)))
    if bound_ever:
        fields.append(("Restrained", "yes"))
    if species:
        fields.append(("Interspecies", species))
    if lactating_ever:
        fields.append(("Lactating", "yes"))
    if implement:
        fields.append(("Masturbating With", implement))

    if fields:
        caption = " | ".join(f"{label} | {value}" for label, value in fields)
    else:
        caption = "Explicit content -- no further detail detected."
    return caption

def caption_video(
    asset_id: str,
    caption_detailed,
    person_names: Optional[List[str]] = None,
    dense: bool = False,
    compilation: bool = False,
    feral: bool = False,
    multiple: bool = False,
    guaranteed_multi: bool = False,
    single: bool = False,
    nonhuman: bool = False,
    full_caption: bool = False,
    masturbation: bool = False,
    categorized: bool = True,
    video_path: Optional[str] = None,
) -> Tuple[str, str]:
    # A caller that already downloaded the original (upload routing does, to triage it)
    # passes it in and keeps ownership of the file; otherwise it's fetched here and removed.
    owns_file = video_path is None
    if owns_file:
        fd, video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
    try:
        if owns_file:
            immich_download_original(asset_id, video_path)
        frames = extract_video_frames(video_path, dense=dense)
        if not frames:
            raise RuntimeError("no frames extracted")

        _, tag_frame = frames[len(frames) // 2]
        generate_and_apply_e621_tags(asset_id, tag_frame, caption_detailed)

        signals = _classify_video_frames(frames, caption_detailed)
        any_nudity = any(
            (s["nudity"] or s["state"] != "NONE") for s in signals if not s["titlecard"]
        )

        # Full narrative captioning for: anything non-sexual (family video etc.), and for
        # the "real life that happens to be filed under porn" albums (Camspy / LV Hookers --
        # Ray-Ban Meta capture), which want the whole description even when nudity shows up.
        if not any_nudity or full_caption:
            parts = []
            for ts, img in frames:
                cap = caption_detailed(img, video_note="This is one frame from a video.", person_names=person_names)
                parts.append(f"[{format_ts(ts)}] {cap}")
            return " || ".join(parts), "VIDEO-FRAMES"

        # Porn that hasn't been filed anywhere the captioner understands: park it rather than
        # guess. Auto-routing between e.g. Masturbation and Single Creampie isn't reliable
        # (most creampie videos open with her fingering herself), so the human sorts it into
        # an album and clears the description, which puts it back in the queue.
        if not categorized:
            return UNCATEGORIZED_CAPTION, "VIDEO-UNCATEGORIZED"

        # Porn: make sure creampie counting, bondage, and interspecies detection cover the
        # whole runtime, not just this sampling pass -- re-scan with full dense/uniform
        # coverage if the initial pass wasn't already dense (mirrors the old cross-listed
        # creampie re-scan trick, just triggered by the NUDITY signal instead of a text
        # regex on a narrative caption that no longer gets generated).
        if not dense:
            frames = extract_video_frames(video_path, dense=True)
            signals = _classify_video_frames(frames, caption_detailed)

        if nonhuman or compilation or feral or not (single or multiple):
            # No creampie count at all for non-human content (anthro/furry/feral) -- the
            # detection was built around live-action photography and doesn't reliably apply
            # to illustrated art.
            #
            # Creampie counting only runs for content the human has actually filed as a
            # creampie -- Single Creampie, or the 200.000.xxx multi range. Everything else
            # (Bondage / Cheating Wife / Prostitution / Internet Titties / Lactation /
            # Masturbation / Hotwife ...) gets the compact caption with no count until it's
            # filed, because the human sorts those into Single or Multiple by hand.
            #
            # Feral is included for a different reason: feral-on-human always ends in
            # ejaculation inside the woman, so the count carries no information -- it would
            # be 1 for every single one of them. Reporting it is just noise.
            #
            # Compilation content is a different thing entirely from Multiple Creampie, not
            # a looser version of it: a compilation cuts between DIFFERENT women each getting
            # creampied once, edited together to skip to the good part -- there's no single
            # "how many creampies did she get" answer to compute, and it's never a "just one
            # creampie" video either (confirmed: a compilation got auto-filed into Single
            # Creampie, and a compilation manually placed in Multiple Creampie got a bogus
            # per-woman event count). So compilation membership skips creampie detection
            # unconditionally, regardless of Multiple Creampie placement.
            count, event_times = 0, []
        elif not multiple:
            # Assume at most one creampie for anything not already manually placed in the
            # Multiple Creampie album (feral overrides even that placement) -- just detect
            # whether one happened at all, don't try to count how many. An isolated CUM
            # reading with no INSERTED evidence anywhere earlier in the video is much more
            # likely a misclassified non-sexual frame than a real creampie -- confirmed in
            # testing, a fully-clothed dialogue scene at the start of a video was confidently
            # (and repeatably, under greedy decoding) misread as CUM, while the real creampie
            # much later sat inside an actual cluster of INSERTED/CUM readings. Requiring a
            # prior INSERTED sighting filters out the isolated false positives without
            # needing the CUM reading itself to ever be more reliable.
            #
            # Also require a partner to have been seen at some point -- INSERTED alone isn't
            # enough evidence, since it fires on toy penetration too, and solo masturbation
            # can't produce a real creampie no matter how wet things get (confirmed: a solo
            # video with no partner in any frame still got a creampie detected).
            #
            # Take the LAST qualifying sighting, not the first: in a real single-creampie
            # video, the creampie itself is near the end (guys stop after they cum), so the
            # latest CUM reading with insertion evidence behind it is the best estimate of
            # the real moment, and it also naturally loses to any earlier isolated false
            # positive if a later, better-supported reading exists.
            #
            # For the same reason, ignore sightings in the opening stretch of the video
            # entirely. The per-frame classifier does confabulate -- observed a fully-clothed
            # setup scene one minute into a twenty-minute video answering GENITALS: Y,
            # CUMLOC: VAGINA -- and an early "creampie" is essentially always one of those,
            # because the scene hasn't happened yet.
            cum_ts = detect_single_creampie(signals)
            count, event_times = (1, [format_ts(cum_ts)]) if cum_ts is not None else (0, [])
        else:
            count, event_times = count_creampie_events(
                [(s["ts"], s["state"], s["partner_visible"]) for s in signals],
                min_count=MULTI_CREAMPIE_MIN_COUNT if guaranteed_multi else 0,
            )
        caption = compact_porn_caption(
            signals, frames, caption_detailed, person_names=person_names,
            masturbation=masturbation, count=count, event_times=event_times,
        )
        return caption, "VIDEO-PORN-COMPACT"
    finally:
        if owns_file:
            try:
                os.remove(video_path)
            except OSError:
                pass

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
        payload = {"name": tag_value}
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

_E621_TAGS_PROMPT = (
    "List e621/Rule34-style tags for this image: species, body type, sex acts, kinks, "
    "objects, clothing, and any other elements relevant to search. Answer with ONLY a "
    "comma-separated list of short lowercase tags (e.g. \"anthro, elephant, breasts, "
    "bondage, rope\") -- no sentences, no numbering, no other text. If nothing tag-worthy "
    "applies, answer exactly: none"
)

def _parse_e621_tags(text: str) -> List[str]:
    if not text or text.strip().lower().startswith("none"):
        return []
    seen = set()
    tags: List[str] = []
    for raw in text.split(","):
        t = raw.strip().lower().strip(".")
        if not t or len(t) > 40 or t in seen:
            continue
        seen.add(t)
        tags.append(t)
    return tags[:20]

def generate_and_apply_e621_tags(asset_id: str, pil_image: Image.Image, caption_detailed) -> None:
    if not ENABLE_TAGS:
        return
    try:
        raw = caption_detailed(pil_image, prompt_override=_E621_TAGS_PROMPT, max_new_tokens=120)
        tags = _parse_e621_tags(raw)
        if tags:
            immich_apply_tags(asset_id, tags)
    except Exception as e:
        print(f"[e621-tags] failed for {asset_id}: {e}", flush=True)

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

def immich_add_to_album(asset_id: str, album_id: str) -> None:
    # Best-effort: adds without removing from any existing album. Immich returns
    # success=False with reason "duplicate" if it's already a member, which is fine.
    try:
        url = f"{IMMICH_URL}/api/albums/{album_id}/assets"
        r = requests.put(
            url,
            headers={**immich_headers(), "Content-Type": "application/json"},
            data=json.dumps({"ids": [asset_id]}),
            timeout=30,
        )
        if r.status_code >= 300:
            print(f"[album] add {asset_id} -> {album_id} failed {r.status_code}: {r.text}", flush=True)
    except Exception as e:
        print(f"[album] add {asset_id} -> {album_id} failed: {e}", flush=True)

def immich_remove_from_album(asset_id: str, album_id: str) -> None:
    try:
        url = f"{IMMICH_URL}/api/albums/{album_id}/assets"
        r = requests.request(
            "DELETE",
            url,
            headers={**immich_headers(), "Content-Type": "application/json"},
            data=json.dumps({"ids": [asset_id]}),
            timeout=30,
        )
        if r.status_code >= 300:
            print(f"[album] remove {asset_id} <- {album_id} failed {r.status_code}: {r.text}", flush=True)
    except Exception as e:
        print(f"[album] remove {asset_id} <- {album_id} failed: {e}", flush=True)

def immich_unarchive(asset_id: str) -> None:
    # Immich's newer API uses the "visibility" enum (timeline/archive/locked) as the
    # authoritative field -- the legacy "isArchived" boolean is derived from it, not
    # independently settable (confirmed empirically: setting isArchived alone no-ops).
    try:
        url = f"{IMMICH_URL}/api/assets"
        r = requests.put(
            url,
            headers={**immich_headers(), "Content-Type": "application/json"},
            data=json.dumps({"ids": [asset_id], "visibility": "timeline"}),
            timeout=30,
        )
        if r.status_code >= 300:
            print(f"[unarchive] {asset_id} failed {r.status_code}: {r.text}", flush=True)
    except Exception as e:
        print(f"[unarchive] {asset_id} failed: {e}", flush=True)

def immich_archive(asset_id: str) -> None:
    try:
        url = f"{IMMICH_URL}/api/assets"
        r = requests.put(
            url,
            headers={**immich_headers(), "Content-Type": "application/json"},
            data=json.dumps({"ids": [asset_id], "visibility": "archive"}),
            timeout=30,
        )
        if r.status_code >= 300:
            print(f"[archive] {asset_id} failed {r.status_code}: {r.text}", flush=True)
    except Exception as e:
        print(f"[archive] {asset_id} failed: {e}", flush=True)

_album_list_cache: Optional[List[dict]] = None
_album_list_cached_at = 0.0
# Albums get created and renumbered while the captioner runs for days at a time, and routing
# looks albums up by number, so the list can't be cached for the life of the process.
ALBUM_LIST_TTL_SECONDS = float(os.environ.get("ALBUM_LIST_TTL_SECONDS", "300"))

def immich_list_albums() -> List[dict]:
    global _album_list_cache, _album_list_cached_at
    if _album_list_cache is not None and time.time() - _album_list_cached_at < ALBUM_LIST_TTL_SECONDS:
        return _album_list_cache
    try:
        r = requests.get(f"{IMMICH_URL}/api/albums", headers=immich_headers(), timeout=60)
        r.raise_for_status()
        _album_list_cache = r.json()
        _album_list_cached_at = time.time()
    except Exception as e:
        print(f"[album] list failed: {e}", flush=True)
        return _album_list_cache or []
    return _album_list_cache

def immich_album_id_by_name(name: str) -> Optional[str]:
    for a in immich_list_albums():
        if a.get("albumName") == name:
            return a.get("id")
    return None

# ----------------------------
# API-only candidate fetch
# ----------------------------
def get_uncaptioned_candidates_api() -> List[Dict]:
    candidates = []
    page = 1
    skipped_in_memory = set()  # In-memory skips for this run (no persistence)

    while True:
        try:
            body = {
                "withExif": True,
                "page": page,
                "size": BATCH_SIZE * 5,  # Larger page size for efficiency (adjust if rate-limited)
            }
            if API_ASSET_TYPE_FILTER:
                body["type"] = API_ASSET_TYPE_FILTER
            response = requests.post(
                f"{IMMICH_URL}/api/search/metadata",
                headers={"x-api-key": IMMICH_API_KEY, "Content-Type": "application/json"},
                json=body,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            # Current Immich API nests results under "assets", not top-level -- verified
            # empirically against this server (not documented consistently across versions).
            assets = data.get("assets", {})
            items = assets.get("items", [])

            if not items:
                break

            # nextPage comes back as a JSON string (e.g. "2"), but the request schema
            # requires "page" to be a number -- passing the raw value straight back in
            # fails validation with a 400. Verified empirically against this server.
            next_page_raw = assets.get("nextPage")
            next_page = int(next_page_raw) if next_page_raw is not None else None

            for item in items:
                asset_id = item.get("id")
                if asset_id in skipped_in_memory:
                    continue
                exif = item.get("exifInfo", {})
                desc = exif.get("description") if exif else None
                if description_is_captionable(desc):
                    candidates.append(item)
                    print(f"[api-candidate] Found uncaptioned: {asset_id}", flush=True)

            if next_page is None:
                break

            page = next_page
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"[api-error] Pagination failed on page {page}: {e}", flush=True)
            break

    print(f"[api] Found {len(candidates)} uncaptioned assets via API scan", flush=True)

    # Same priority as the DB path: images before videos, newest first within each type.
    # Two stable sorts: createdAt descending first, then type-priority ascending -- the
    # second sort preserves the createdAt ordering within each type group.
    candidates.sort(key=lambda item: item.get("createdAt", "") or "", reverse=True)
    candidates.sort(key=lambda item: 0 if str(item.get("type", "")).upper() == "IMAGE" else 1)

    return candidates

def refresh_asset_albums(asset_id: str, fallback: List[str]) -> List[str]:
    """Album membership as of right now, falling back to what the caller already had.

    In DB-direct mode the album list arrives from the prefetch thread's candidate query,
    which by design runs well ahead of the GPU -- easily long enough for a just-uploaded
    asset to be queued in the same second it was created, before the uploading client has
    finished adding it to its album. Captioning that snapshot loses the identity name and,
    worse, reads as "no album" for every routing rule downstream. Re-reading here costs one
    localhost round-trip against a job that takes tens of seconds.

    A failed lookup returns the fallback rather than [], since an asset wrongly seen as
    album-less gets parked or stripped of its identity.

    GET /api/assets/{id} does NOT include album membership on this Immich version (verified
    empirically -- AssetResponseDto has no "albums" field). GET /api/albums?assetId={id} is
    the endpoint that works, returning the albums directly, each with an "albumName".
    """
    try:
        url = f"{IMMICH_URL}/api/albums"
        r = requests.get(url, headers=immich_headers(), params={"assetId": asset_id}, timeout=30)
        r.raise_for_status()
        return [a.get("albumName") for a in r.json() if a.get("albumName")]
    except Exception as e:
        print(f"[api] Album re-read failed for {asset_id}, keeping queued list: {e}", flush=True)
        return fallback

# ----------------------------
# Postgres helpers (ONLY used if not USE_API_ONLY)
# ----------------------------
def pg_connect():
    import psycopg2
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

def pg_fetch_candidates(conn, limit: int, exclude_face_wait: bool = False) -> List[dict]:
    import psycopg2.extras
    has_type = pg_column_exists(conn, "asset", "type")
    select_type = 'a."type",' if has_type else "NULL::text as type,"

    # Images before videos (videos are far more expensive per-asset), newest first within
    # each type -- so anything newly added always jumps to the front of its type's queue
    # instead of waiting behind the whole existing backlog.
    if has_type:
        order_clause = "ORDER BY CASE WHEN a.\"type\" = 'IMAGE' THEN 0 ELSE 1 END ASC, a.\"createdAt\" DESC"
    else:
        order_clause = 'ORDER BY a."createdAt" DESC'

    # New uploads waiting on face recognition (upload routing, step 2) are left out until
    # their next check is due, so the queue moves on to other work meanwhile.
    face_wait_join = (
        "LEFT JOIN captioner_face_wait fw ON fw.asset_id = a.id" if exclude_face_wait else ""
    )
    face_wait_filter = (
        "AND (fw.asset_id IS NULL OR fw.next_check <= now())" if exclude_face_wait else ""
    )

    sql = f"""
    SELECT
      a.id as id,
      {select_type}
      ae.description as description,
      ae.make as exif_make,
      a."originalFileName" as original_file_name,
      COALESCE(array_remove(array_agg(al."albumName"), NULL), '{{}}'::text[]) as albums
    FROM asset a
    JOIN asset_exif ae ON ae."assetId" = a.id
    LEFT JOIN captioner_skip cs ON cs.asset_id = a.id
    LEFT JOIN album_asset aa ON aa."assetId" = a.id
    LEFT JOIN album al ON al.id = aa."albumId"
    {face_wait_join}
    WHERE
      cs.asset_id IS NULL
      {face_wait_filter}
      AND (
        ae.description IS NULL
        OR btrim(ae.description) = ''
        -- A description holding nothing but generation info is still uncaptioned. Matched
        -- loosely here (any JSON-looking blob) because Postgres can't judge the keys;
        -- description_is_captionable() makes the real call on the fetched row.
        -- The LIKE wildcards below are doubled: this statement is executed with a bound
        -- parameter, so psycopg2 treats a lone percent sign as the start of a placeholder
        -- and raises IndexError on a literal one. Keep any comment in this string free of
        -- percent signs for the same reason.
        OR (btrim(ae.description) LIKE '{{%%' AND btrim(ae.description) LIKE '%%}}')
      )
    GROUP BY a.id, ae.description, ae.make, a."originalFileName" {', a."type"' if has_type else ''}
    {order_clause}
    LIMIT %s;
    """

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (limit,))
        return list(cur.fetchall())

# ----------------------------
# Upload routing (new assets) and album-move handling
# ----------------------------
# Two things happen here that the rest of this file doesn't do:
#
#   * A genuinely NEW asset -- in no album, and never processed before -- is walked through a
#     fixed decision order before it gets captioned: CamSpy, then "is this one of us" (face
#     recognition), then anthro, then nudity, then the porn categories. Each step can file it
#     into albums, archive it, and stop. Assets that are already filed, or that were
#     captioned once and had their description cleared (apply_people_names.py does this
#     hourly, and so does the human to re-queue something), skip all of that and are simply
#     re-captioned by the albums they sit in, exactly as before.
#
#   * An asset the HUMAN adds to certain albums gets acted on: Multiple Creampie and the
#     100.000.x branch run the CumCounter, Single Creampie archives, and anything parked at
#     "Please Categorize" loses that prefix once it's filed. Immich has no event for "asset
#     added to album", so membership is snapshotted in Postgres and diffed on a timer.
#
# Both need state that survives restarts (what's been routed, what's waiting on face
# recognition, the membership snapshot), which lives in three small captioner_* tables in the
# Immich database. Without Postgres credentials, routing and move handling are switched off
# and everything is captioned by album the old way.
ROUTING_ENABLED = os.environ.get("ROUTING_ENABLED", "1") == "1"

# Face recognition runs a while after upload. A new asset waits for Immich to finish it
# (asset_job_status."facesRecognizedAt"), plus a grace period for the per-face recognition
# jobs that detection queues, before step 2 decides whether it shows one of us. After
# FACE_WAIT_MAX_SECONDS it goes ahead regardless, treated as nobody known.
FACE_WAIT_GRACE_SECONDS = int(os.environ.get("FACE_WAIT_GRACE_SECONDS", "120"))
FACE_WAIT_MAX_SECONDS = int(os.environ.get("FACE_WAIT_MAX_SECONDS", "3600"))
FACE_WAIT_RECHECK_SECONDS = int(os.environ.get("FACE_WAIT_RECHECK_SECONDS", "60"))

MOVE_POLL_SECONDS = float(os.environ.get("MOVE_POLL_SECONDS", "60"))
# A move whose handling keeps failing (e.g. the video won't download) is given up on after
# this many attempts, so one broken file can't hold the GPU in a retry loop.
MOVE_MAX_ATTEMPTS = int(os.environ.get("MOVE_MAX_ATTEMPTS", "3"))

# Per-frame classifier answers are noisy even under greedy decoding, so a video only counts
# as showing something when at least this many sampled frames say so (videos with fewer
# than three sampled frames need just one).
VIDEO_FLAG_MIN_FRAMES = int(os.environ.get("VIDEO_FLAG_MIN_FRAMES", "2"))
# Cap on how many frames get the (longer) porn-category prompt.
PORN_PROMPT_MAX_FRAMES = int(os.environ.get("PORN_PROMPT_MAX_FRAMES", "12"))
# Extra early timestamps checked for studio title cards/logos, which open the video and
# fall before the regular head samples.
TITLECARD_TIMESTAMPS = [
    float(x) for x in os.environ.get("TITLECARD_TIMESTAMPS", "0.5,2,5,10").split(",") if x.strip()
]

CAMSPY_FILENAME_KEYWORD = os.environ.get("CAMSPY_FILENAME_KEYWORD", "SpyPhoto").strip().lower()

# Immich People who must NOT be treated as "one of us" in step 2. LydiaDog is a generated
# character with her own step-3 rule, not a person album.
ROUTING_PERSON_EXCLUDE = {
    n.strip().casefold()
    for n in os.environ.get("ROUTING_PERSON_EXCLUDE", "Lydia Dog,LydiaDog").split(",")
    if n.strip()
}

# LydiaDog match for anthro stills. Three signals, strongest first: the generation-info JSON
# naming her LoRA (every render from the local pipeline carries it; the LoRA has shipped under
# several filenames, so any mention of "lydia" counts -- only renders carry generation info,
# so a real photo of Lydia can never hit this), Immich's own Person tag,
# and a low-threshold face re-detection compared against the faces already tagged as her --
# the same technique scripts/tag_album_people_faces.py uses, because the stock detector finds
# a face on only a few percent of these renders.
LYDIADOG_PERSON_NAME = os.environ.get("LYDIADOG_PERSON_NAME", "Lydia Dog")
LYDIADOG_GEN_INFO_KEYWORDS = [
    k.strip().lower()
    for k in os.environ.get("LYDIADOG_GEN_INFO_KEYWORDS", "lydia").split(",")
    if k.strip()
]
LYDIADOG_MIN_SIMILARITY = float(os.environ.get("LYDIADOG_MIN_SIMILARITY", "0.70"))
LYDIADOG_DETECT_MIN_SCORE = float(os.environ.get("LYDIADOG_DETECT_MIN_SCORE", "0.05"))
# Anything filed under this album number -- by the human or by routing -- gets the LydiaDog
# Person tag on its main face, so it shows up on her People page.
LYDIADOG_ALBUM_PREFIX = os.environ.get("LYDIADOG_ALBUM_PREFIX", "300.006.")
# Face acceptance, same tiers as scripts/tag_album_people_faces.py: a box at or above
# LYDIADOG_SCORE_CONFIDENT is taken outright; a weaker one needs LYDIADOG_TAG_MIN_SIMILARITY
# to her already-tagged faces.
LYDIADOG_SCORE_CONFIDENT = float(os.environ.get("LYDIADOG_SCORE_CONFIDENT", "0.20"))
LYDIADOG_TAG_MIN_SIMILARITY = float(os.environ.get("LYDIADOG_TAG_MIN_SIMILARITY", "0.60"))
ML_URL = os.environ.get("ML_URL", "http://immich-machine-learning:3003/predict")
ML_FACE_MODEL = os.environ.get("ML_FACE_MODEL", "buffalo_l")

ANTHRO_VIDEO_ALBUM_ID = os.environ.get("ANTHRO_VIDEO_ALBUM_ID", "9d09367b-1416-4488-996d-6f4caca26ae1")
MULTIPLE_CREAMPIE_ALBUM_ID = os.environ.get("MULTIPLE_CREAMPIE_ALBUM_ID", "e7479905-44b5-42ca-86d0-aaf8fb7c36e3")

# Routing albums are found by their number, so renaming the text after the number (or
# renumbering, via ROUTING_ALBUM_NUMBERS="key=number;...") needs no code change. Trailing
# ".000" groups are ignored when comparing, so "300.001" also finds "300.001.000 - ...".
# Where the number isn't found, the album's known ID is used instead if there is one.
_DEFAULT_ROUTING_ALBUM_NUMBERS = {
    "camspy": "400.001",
    "furry": "300.000.000",
    "cow_anthro": "300.000.002",
    "anthro_video": "300.001",
    "anthro_sex_video": "300.002",
    "human_anthro_video": "300.004",
    "human_anthro_still": "300.005",
    "lydia_dog": "300.006.000",
    "multi": "200.000.000",
    "puta_locura": "200.000.001",
    "creampie_squad": "200.000.002",
    "gangbang_creampie": "200.000.003",
    "slutwife_jessica": "200.000.004",
    "slutwife_marion": "200.000.005",
    "glorywall": "200.000.006",
    "hentaied": "200.000.008",
    "single": "200.001.000",
    "bondage_creampie": "200.002.000",
    "internet_titties": "200.010.000",
    "masturbation": "200.010.001",
    "lactation": "200.010.002",
}
ROUTING_ALBUM_NUMBERS = {
    **_DEFAULT_ROUTING_ALBUM_NUMBERS,
    **_parse_kv_map(os.environ.get("ROUTING_ALBUM_NUMBERS", "")),
}
_ROUTING_ALBUM_FALLBACK_IDS = {
    "camspy": CAMSPY_ALBUM_ID,
    "furry": FURRY_ALBUM_ID,
    "anthro_video": ANTHRO_VIDEO_ALBUM_ID,
    "multi": MULTIPLE_CREAMPIE_ALBUM_ID,
    "single": SINGLE_CREAMPIE_ALBUM_ID,
    "lactation": LACTATION_ALBUM_ID,
    # Hucow has no number in the routing spec, so it's only ever found by ID.
    "hucow": HUCOW_ALBUM_ID,
}
# Albums that are porn categories: filing into any of them makes the asset archivable.
_PORN_ALBUM_KEYS = {
    "multi", "puta_locura", "creampie_squad", "gangbang_creampie", "slutwife_jessica",
    "slutwife_marion", "glorywall", "hentaied", "single", "bondage_creampie",
    "internet_titties", "masturbation", "lactation", "hucow",
}
# Studio logo / title card / filename -> album. Matched against lowercase text with
# everything but letters and digits removed, so "SLUTWIFE-Jessica", "Slutwife Jessica" and
# "slutwifejessica" all hit. Every studio match also goes into Multiple Creampie.
STUDIO_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"slutwifejessica"), "slutwife_jessica"),
    (re.compile(r"slutwifemarion"), "slutwife_marion"),
    (re.compile(r"putalocura"), "puta_locura"),
    (re.compile(r"creampiesquad"), "creampie_squad"),
    (re.compile(r"gangbangcreampie|(?:5|five)guycreampie"), "gangbang_creampie"),
]
_HENTAIED_TEXT_RE = re.compile(r"hentaied")
MULTI_EVENT_COUNT_ALBUM_PREFIXES = tuple(
    p.strip() for p in os.environ.get("MULTI_EVENT_COUNT_ALBUM_PREFIXES", "100.000.").split(",")
    if p.strip()
)

_ALBUM_NUMBER_RE = re.compile(r"^\s*(\d{3}(?:\.\d{3})*)(?!\d)")
_ALBUM_TITLE_RE = re.compile(r"^\s*\d{3}(?:\.\d{3})*\s*-\s*(.+?)\s*$")

def album_number(album_name: str) -> Optional[str]:
    m = _ALBUM_NUMBER_RE.match(album_name or "")
    return m.group(1) if m else None

def normalize_album_number(number: str) -> str:
    parts = (number or "").strip().split(".")
    while len(parts) > 1 and parts[-1] == "000":
        parts.pop()
    return ".".join(parts)

def album_title(album_name: str) -> str:
    """The part of an album name after its number: "002.000 - Lydia" -> "Lydia"."""
    m = _ALBUM_TITLE_RE.match(album_name or "")
    return m.group(1) if m else (album_name or "").strip()

_missing_album_warned: set = set()

def album_id_for(key: str) -> Optional[str]:
    number = ROUTING_ALBUM_NUMBERS.get(key)
    if number:
        want = normalize_album_number(number)
        for a in immich_list_albums():
            n = album_number(a.get("albumName") or "")
            if n and normalize_album_number(n) == want:
                return a.get("id")
    fallback = _ROUTING_ALBUM_FALLBACK_IDS.get(key)
    if fallback:
        return fallback
    if key not in _missing_album_warned:
        _missing_album_warned.add(key)
        print(f"[route] WARNING: no album numbered {number} for '{key}' -- not filing there", flush=True)
    return None

def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())

def match_studio(text: str) -> Optional[str]:
    t = _compact_text(text)
    for rx, key in STUDIO_RULES:
        if rx.search(t):
            return key
    return None

def is_camspy_upload(exif_make: Optional[str], filename: Optional[str]) -> bool:
    if (exif_make or "").strip().lower() == CAMSPY_EXIF_MAKE:
        return True
    return bool(CAMSPY_FILENAME_KEYWORD) and CAMSPY_FILENAME_KEYWORD in (filename or "").lower()

# ---- "Please Categorize" prefix and creampie-count field editing ----
def has_uncategorized_prefix(caption: str) -> bool:
    c = (caption or "").strip().lower()
    p = UNCATEGORIZED_CAPTION.lower()
    return c == p or c.startswith(p + " |")

def strip_uncategorized_prefix(caption: str) -> str:
    c = (caption or "").strip()
    if not has_uncategorized_prefix(c):
        return c
    return c[len(UNCATEGORIZED_CAPTION):].lstrip().lstrip("|").strip()

def with_uncategorized_prefix(caption: str) -> str:
    c = strip_uncategorized_prefix(caption)
    return f"{UNCATEGORIZED_CAPTION} | {c}" if c else UNCATEGORIZED_CAPTION

def with_creampie_count(caption: str, count: int, event_times: List[str]) -> str:
    """Replace (or add) the "Separate Creampies" field, leaving every other field as is."""
    parts = [p.strip() for p in (caption or "").split(" | ")]
    if len(parts) >= 2 and parts[0].lower() == "separate creampies":
        parts = parts[2:]
    rest = " | ".join(p for p in parts if p)
    if count < 1:
        return rest
    field = f"Separate Creampies | {count} (~{', '.join(event_times)})"
    return f"{field} | {rest}" if rest else field

# ---- Classifier prompts for routing ----
_TRIAGE_PROMPT = (
    "Look at this image. Respond with ONLY these four labeled lines, nothing else -- no other "
    "sentences or explanations.\n"
    "ANTHRO: Y or N -- Y if any anthropomorphic animal character is shown: a humanoid "
    "character with an animal head, muzzle or snout, fur covering the body, paws, or an animal "
    "tail (furry / anthro art, whether drawn, painted, or 3D-rendered). N for real animals. N "
    "for an ordinary human, including a human wearing animal ears, a tail, horns, or cow-print "
    "clothing as a costume or accessory.\n"
    "COW: Y or N -- Y only if ANTHRO is Y and at least one anthro character is a cow, bull, or "
    "other bovine.\n"
    "SEX: one of NONE, SOLO, HUMAN-HUMAN, HUMAN-ANTHRO, ANTHRO-ANTHRO -- whether a sex act is "
    "happening right now and between whom. HUMAN-ANTHRO when a human and an anthro character "
    "are having sex with each other. ANTHRO-ANTHRO when only anthro characters are. HUMAN-HUMAN "
    "when only humans are. SOLO when one character is masturbating alone. NONE if no sex act is "
    "happening (posing nude is NONE).\n"
    "NUDITY: Y or N -- Y if any bare breast, bare nipple, bare butt, or genitals are visible, "
    "even partially (e.g. a nipple peeking out of clothing). Otherwise N.\n"
    "Example answer:\nANTHRO: N\nCOW: N\nSEX: NONE\nNUDITY: N"
)

def _parse_triage(text: str) -> dict:
    t = (text or "").upper()
    m = re.search(r"SEX\s*:\s*(NONE|SOLO|HUMAN-HUMAN|HUMAN-ANTHRO|ANTHRO-HUMAN|ANTHRO-ANTHRO)\b", t)
    sex = m.group(1) if m else "NONE"
    if sex == "ANTHRO-HUMAN":
        sex = "HUMAN-ANTHRO"
    anthro = bool(re.search(r"ANTHRO\s*:\s*Y", t))
    return {
        "anthro": anthro,
        "cow": anthro and bool(re.search(r"COW\s*:\s*Y", t)),
        "sex_human_anthro": sex == "HUMAN-ANTHRO",
        "sex_anthro_anthro": sex == "ANTHRO-ANTHRO",
        "nudity": bool(re.search(r"NUDITY\s*:\s*Y", t)),
    }

_PORN_CATEGORY_PROMPT = (
    "Look at this image. Respond with ONLY these eight labeled lines, nothing else -- no other "
    "sentences or explanations.\n"
    "WOMEN: the number of adult women clearly visible, as a digit (0, 1, 2, 3 ...).\n"
    "MEN: Y or N -- Y if any man, or any part of a man (body, hand, cock), is visible.\n"
    "MASTURBATING: Y or N -- Y if a woman is rubbing, fingering, or penetrating her own pussy, "
    "clit, or ass with her hand, fingers, or a toy.\n"
    "COWPRINT: Y or N -- Y if a woman is wearing black-and-white cow-print clothing or "
    "accessories, or cow horns.\n"
    "LACTATING: Y or N -- Y if milk is visibly leaking, dripping, or spraying from a woman's "
    "bare nipples/breasts.\n"
    "GLORYWALL: Y or N -- Y if a woman is stuck through a wall or partition, with her upper "
    "body on one side and her ass and legs on the other side, so she can be used from the "
    "other side.\n"
    "TENTACLES: Y or N -- Y if tentacles are wrapped around or penetrating a woman.\n"
    "CUMLOC: where a visible load of cum/jizz is, exactly one of VAGINA (in, on, or leaking out "
    "of her vagina), FACE, TITS, ASS, BODY, NONE. Lubricant clinging to a shaft, her own clear "
    "wetness, and milk are all NONE.\n"
    "Example answer:\nWOMEN: 1\nMEN: N\nMASTURBATING: N\nCOWPRINT: N\nLACTATING: N\n"
    "GLORYWALL: N\nTENTACLES: N\nCUMLOC: NONE"
)

def _parse_porn_categories(text: str) -> dict:
    t = (text or "").upper()
    m = re.search(r"WOMEN\s*:\s*(\d+)", t)
    m2 = re.search(r"CUMLOC\s*:\s*(VAGINA|FACE|TITS|ASS|BODY|NONE)\b", t)
    return {
        "women": int(m.group(1)) if m else 0,
        "men": bool(re.search(r"(?<!WO)MEN\s*:\s*Y", t)),
        "masturbating": bool(re.search(r"MASTURBATING\s*:\s*Y", t)),
        "cowprint": bool(re.search(r"COWPRINT\s*:\s*Y", t)),
        "lactating": bool(re.search(r"LACTATING\s*:\s*Y", t)),
        "glorywall": bool(re.search(r"GLORYWALL\s*:\s*Y", t)),
        "tentacles": bool(re.search(r"TENTACLES\s*:\s*Y", t)),
        "vaginal_cum": bool(m2) and m2.group(1) == "VAGINA",
    }

def _classify_with_prompt(images: List[Image.Image], prompt: str, parser, caption_detailed,
                          max_new_tokens: int = 80) -> List[dict]:
    if not images:
        return []
    batch_fn = getattr(caption_detailed, "batch", None)
    if batch_fn:
        raw = batch_fn([
            {"pil_image": img, "prompt_override": prompt, "max_new_tokens": max_new_tokens, "greedy": True}
            for img in images
        ])
    else:
        raw = [caption_detailed(img, prompt_override=prompt, max_new_tokens=max_new_tokens, greedy=True)
               for img in images]
    return [parser(text) for text in raw]

def frames_agree(rows: List[dict], key: str) -> bool:
    """Whether enough of the per-frame answers say `key` -- see VIDEO_FLAG_MIN_FRAMES."""
    if not rows:
        return False
    need = 1 if len(rows) < 3 else min(VIDEO_FLAG_MIN_FRAMES, len(rows))
    return sum(1 for r in rows if r.get(key)) >= need

def _evenly(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]

# ---- LydiaDog face match ----
_lydiadog_refs: Dict[str, object] = {"at": 0.0, "embeddings": []}

def _cosine(a: List[float], b: List[float]) -> float:
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def _ml_detect_faces(image_bytes: bytes) -> dict:
    """Low-threshold face detection via Immich's ML service.
    Returns {"w", "h", "boxes": [{x1, y1, x2, y2, score, embedding}]}."""
    entries = json.dumps({
        "facial-recognition": {
            "detection": {"modelName": ML_FACE_MODEL, "options": {"minScore": LYDIADOG_DETECT_MIN_SCORE}},
            "recognition": {"modelName": ML_FACE_MODEL},
        }
    })
    r = requests.post(ML_URL, data={"entries": entries},
                      files={"image": ("image.jpg", io.BytesIO(image_bytes), "application/octet-stream")},
                      timeout=300)
    r.raise_for_status()
    data = r.json()
    boxes = []
    for f in data.get("facial-recognition") or []:
        emb = f.get("embedding")
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except ValueError:
                emb = None
        b = f.get("boundingBox") or {}
        boxes.append({"x1": b.get("x1"), "y1": b.get("y1"), "x2": b.get("x2"), "y2": b.get("y2"),
                      "score": float(f.get("score") or 0.0), "embedding": emb})
    return {"w": int(data.get("imageWidth") or 0), "h": int(data.get("imageHeight") or 0),
            "boxes": boxes}

def _lydiadog_reference_embeddings(state: "RoutingState") -> List[List[float]]:
    if time.time() - float(_lydiadog_refs["at"]) > 3600:
        _lydiadog_refs["embeddings"] = state.person_face_embeddings(LYDIADOG_PERSON_NAME)
        _lydiadog_refs["at"] = time.time()
    return _lydiadog_refs["embeddings"]

def matches_lydiadog(asset_id: str, gen_info: Optional[str], raw_people: List[str],
                     state: "RoutingState") -> bool:
    if gen_info and any(kw in gen_info.lower() for kw in LYDIADOG_GEN_INFO_KEYWORDS):
        print(f"[route] {asset_id} LydiaDog by generation info", flush=True)
        return True
    if any(n.casefold() == LYDIADOG_PERSON_NAME.casefold() for n in raw_people):
        print(f"[route] {asset_id} LydiaDog by Immich person tag", flush=True)
        return True
    refs = _lydiadog_reference_embeddings(state)
    if not refs:
        return False
    try:
        r = requests.get(f"{IMMICH_URL}/api/assets/{asset_id}/thumbnail", headers=immich_headers(),
                         params={"size": "preview"}, timeout=120)
        r.raise_for_status()
        faces = [b for b in _ml_detect_faces(r.content)["boxes"] if b.get("embedding")]
    except Exception as e:
        print(f"[route] {asset_id} LydiaDog face check failed: {e}", flush=True)
        return False
    best = max((max(_cosine(f["embedding"], ref) for ref in refs) for f in faces), default=0.0)
    if best >= LYDIADOG_MIN_SIMILARITY:
        print(f"[route] {asset_id} LydiaDog by face similarity {best:.2f}", flush=True)
        return True
    return False

_person_id_cache: Dict[str, str] = {}

def immich_person_id(name: str) -> Optional[str]:
    if name in _person_id_cache:
        return _person_id_cache[name]
    r = requests.get(f"{IMMICH_URL}/api/people", headers=immich_headers(),
                     params={"withHidden": "true"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    people = data.get("people", []) if isinstance(data, dict) else data
    matches = [p for p in people if p.get("name") == name]
    if len(matches) != 1:
        print(f"[tag] {len(matches)} Immich people named {name!r} -- can't tag", flush=True)
        return None
    _person_id_cache[name] = matches[0]["id"]
    return _person_id_cache[name]

def _clamp_box(box: dict, w: int, h: int) -> Optional[Dict[str, int]]:
    try:
        x1, y1 = max(0.0, float(box["x1"])), max(0.0, float(box["y1"]))
        x2, y2 = min(float(w), float(box["x2"])), min(float(h), float(box["y2"]))
    except (KeyError, TypeError, ValueError):
        return None
    # Specks and edge artefacts: the same 3%-of-the-frame floor the album script uses.
    if x2 - x1 < max(20, 0.03 * w) or y2 - y1 < max(20, 0.03 * h):
        return None
    return {"x": round(x1), "y": round(y1), "width": round(x2 - x1), "height": round(y2 - y1)}

def tag_lydiadog(asset_id: str, state: "RoutingState") -> None:
    """Give an asset filed under LydiaDog's album the LydiaDog Person tag.

    Immich can only tag a person through a face row, and the stock detector finds a face on
    only a few percent of her renders. So, in order: keep an existing LydiaDog face; adopt an
    unnamed face Immich already found; re-detect at a low threshold and take the best box
    (confident score, or similar enough to her tagged faces); and if nothing qualifies, tag
    the whole frame, because album membership already says it's her. A face belonging to a
    different named person is never taken over -- a new LydiaDog face is added beside it."""
    person_id = immich_person_id(LYDIADOG_PERSON_NAME)
    if not person_id:
        return
    r = requests.get(f"{IMMICH_URL}/api/faces", headers=immich_headers(),
                     params={"id": asset_id}, timeout=60)
    r.raise_for_status()
    faces = r.json() or []
    if any((f.get("person") or {}).get("id") == person_id for f in faces):
        return
    adoptable = [f for f in faces if not (f.get("person") or {}).get("name")]
    if adoptable:
        face = max(adoptable, key=lambda f: (
            max(0, f.get("boundingBoxX2", 0) - f.get("boundingBoxX1", 0))
            * max(0, f.get("boundingBoxY2", 0) - f.get("boundingBoxY1", 0))))
        if not DRY_RUN:
            # The PATH carries the target person and the BODY the face being moved.
            requests.put(f"{IMMICH_URL}/api/faces/{person_id}",
                         headers={**immich_headers(), "Content-Type": "application/json"},
                         data=json.dumps({"id": face["id"]}), timeout=60).raise_for_status()
        print(f"[tag] {asset_id} LydiaDog: adopted existing face {face['id']}", flush=True)
        return

    r = requests.get(f"{IMMICH_URL}/api/assets/{asset_id}/original", headers=immich_headers(), timeout=180)
    r.raise_for_status()
    det = _ml_detect_faces(r.content)
    w, h = det["w"], det["h"]
    if not w or not h:
        print(f"[tag] {asset_id} LydiaDog: ML service returned no image size -- skipped", flush=True)
        return
    refs = _lydiadog_reference_embeddings(state)
    candidates = []
    for b in det["boxes"]:
        geom = _clamp_box(b, w, h)
        if not geom:
            continue
        sim = max((_cosine(b["embedding"], ref) for ref in refs), default=None) if b.get("embedding") else None
        if b["score"] >= LYDIADOG_SCORE_CONFIDENT:
            tier = 2
        elif sim is not None and sim >= LYDIADOG_TAG_MIN_SIMILARITY:
            tier = 1
        else:
            continue
        candidates.append((tier, geom["width"] * geom["height"], geom))
    if candidates:
        geom = max(candidates, key=lambda c: (c[0], c[1]))[2]
        how = "detected face"
    else:
        geom = {"x": 0, "y": 0, "width": w, "height": h}
        how = "whole frame (no face found)"
    if not DRY_RUN:
        requests.post(f"{IMMICH_URL}/api/faces",
                      headers={**immich_headers(), "Content-Type": "application/json"},
                      data=json.dumps({"personId": person_id, "assetId": asset_id,
                                       "imageWidth": w, "imageHeight": h, **geom}),
                      timeout=60).raise_for_status()
    print(f"[tag] {asset_id} LydiaDog: tagged {how} {geom}", flush=True)

# ---- Person albums (step 2) ----
def person_albums(raw_people: List[str]) -> List[Tuple[str, str, str]]:
    """(person, album name, album id) for each tagged person who has an album of their own --
    one whose title after the number is exactly their Immich name, or their name after
    PEOPLE_NAME_OVERRIDES. A person with several such albums gets the lowest-numbered one."""
    albums = immich_list_albums()
    out: List[Tuple[str, str, str]] = []
    for raw in raw_people:
        canon = canonical_people_name(raw)
        if raw.casefold() in ROUTING_PERSON_EXCLUDE or canon.casefold() in ROUTING_PERSON_EXCLUDE:
            continue
        wanted = {raw.casefold(), canon.casefold()}
        hits = sorted(
            (a.get("albumName") or "", a.get("id"))
            for a in albums
            if album_title(a.get("albumName") or "").casefold() in wanted and a.get("id")
        )
        if hits:
            out.append((raw, hits[0][0], hits[0][1]))
    return out

# ---- Persistent state (Postgres) ----
class RoutingState:
    def __init__(self, conn):
        self.conn = conn
        self._ensure_tables()
        self.has_job_status = pg_column_exists(conn, "asset_job_status", "facesRecognizedAt")
        if not self.has_job_status:
            print("[route] asset_job_status.facesRecognizedAt not found -- face wait falls back to "
                  "FACE_WAIT_GRACE_SECONDS after upload", flush=True)
        self.face_has_deleted_at = pg_column_exists(conn, "asset_face", "deletedAt")
        # Immich v3.2 links faces to people through personGroupId; asset_face has no personId.
        self.face_person_join = (
            'p.id = af."personId"' if pg_column_exists(conn, "asset_face", "personId")
            else 'p."personGroupId" = af."personGroupId"')

    def _exec(self, sql: str, params=None):
        with self.conn.cursor() as cur:
            if params is None:
                cur.execute(sql)
            else:
                cur.execute(sql, params)
            return cur.fetchall() if cur.description else None

    def _meta_get(self, key: str) -> Optional[str]:
        rows = self._exec("SELECT value FROM captioner_meta WHERE key = %s", (key,))
        return rows[0][0] if rows else None

    def _meta_set(self, key: str, value: str) -> None:
        self._exec("INSERT INTO captioner_meta(key, value) VALUES (%s, %s) "
                   "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (key, value))

    def _ensure_tables(self) -> None:
        self._exec("""
            CREATE TABLE IF NOT EXISTS captioner_meta (key text PRIMARY KEY, value text NOT NULL);
            CREATE TABLE IF NOT EXISTS captioner_routed (
              asset_id uuid PRIMARY KEY, outcome text NOT NULL,
              routed_at timestamptz NOT NULL DEFAULT now());
            CREATE TABLE IF NOT EXISTS captioner_face_wait (
              asset_id uuid PRIMARY KEY, next_check timestamptz NOT NULL);
            CREATE TABLE IF NOT EXISTS captioner_album_member (
              album_id uuid NOT NULL, asset_id uuid NOT NULL, PRIMARY KEY (album_id, asset_id));
        """)
        # First run only: everything that already has a real caption was processed before
        # routing existed, so clearing its description later must not make it look new.
        # Descriptions holding nothing but generation info are NOT seeded -- those are fresh
        # renders from the local pipeline that were never captioned. (No bound parameters on
        # this statement, so the LIKE wildcards are single percent signs.)
        if self._meta_get("routed_seeded") is None:
            self._exec("""
                INSERT INTO captioner_routed(asset_id, outcome)
                SELECT a.id, 'preexisting' FROM asset a
                JOIN asset_exif ae ON ae."assetId" = a.id
                WHERE ae.description IS NOT NULL AND btrim(ae.description) <> ''
                  AND NOT (btrim(ae.description) LIKE '{%' AND btrim(ae.description) LIKE '%}')
                ON CONFLICT DO NOTHING
            """)
            self._meta_set("routed_seeded", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            print("[route] seeded captioner_routed with already-captioned assets", flush=True)
        # Likewise, only album additions made after this first run count as moves.
        if self._meta_get("album_member_seeded") is None:
            self._exec("""
                INSERT INTO captioner_album_member(album_id, asset_id)
                SELECT "albumId", "assetId" FROM album_asset ON CONFLICT DO NOTHING
            """)
            self._meta_set("album_member_seeded", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            print("[route] seeded album membership snapshot", flush=True)

    def is_routed(self, asset_id: str) -> bool:
        return bool(self._exec("SELECT 1 FROM captioner_routed WHERE asset_id = %s", (asset_id,)))

    def mark_routed(self, asset_id: str, outcome: str) -> None:
        self._exec("INSERT INTO captioner_routed(asset_id, outcome) VALUES (%s, %s) "
                   "ON CONFLICT (asset_id) DO UPDATE SET outcome = EXCLUDED.outcome, routed_at = now()",
                   (asset_id, outcome))
        self._exec("DELETE FROM captioner_face_wait WHERE asset_id = %s", (asset_id,))

    def face_status(self, asset_id: str) -> str:
        """"ready", "timeout" (waited long enough, go ahead), or "wait"."""
        if self.has_job_status:
            rows = self._exec("""
                SELECT (ajs."facesRecognizedAt" IS NOT NULL
                        AND ajs."facesRecognizedAt" <= now() - make_interval(secs => %s)),
                       (a."createdAt" <= now() - make_interval(secs => %s))
                FROM asset a LEFT JOIN asset_job_status ajs ON ajs."assetId" = a.id
                WHERE a.id = %s
            """, (FACE_WAIT_GRACE_SECONDS, FACE_WAIT_MAX_SECONDS, asset_id))
        else:
            rows = self._exec("""
                SELECT false, (a."createdAt" <= now() - make_interval(secs => %s))
                FROM asset a WHERE a.id = %s
            """, (FACE_WAIT_GRACE_SECONDS, asset_id))
        if not rows:
            return "timeout"
        recognized, timed_out = rows[0]
        if recognized:
            return "ready"
        return "timeout" if timed_out else "wait"

    def defer(self, asset_id: str) -> None:
        self._exec("INSERT INTO captioner_face_wait(asset_id, next_check) "
                   "VALUES (%s, now() + make_interval(secs => %s)) "
                   "ON CONFLICT (asset_id) DO UPDATE SET next_check = EXCLUDED.next_check",
                   (asset_id, FACE_WAIT_RECHECK_SECONDS))

    def deferred_ids(self) -> set:
        rows = self._exec("SELECT asset_id::text FROM captioner_face_wait WHERE next_check > now()")
        return {r[0] for r in rows or []}

    def record_membership(self, album_id: str, asset_id: str) -> None:
        self._exec("INSERT INTO captioner_album_member(album_id, asset_id) VALUES (%s, %s) "
                   "ON CONFLICT DO NOTHING", (album_id, asset_id))

    def new_memberships(self) -> List[dict]:
        """Album memberships that appeared since the last poll (and forget removed ones, so
        re-adding an asset later counts as a fresh move)."""
        self._exec("""
            DELETE FROM captioner_album_member m WHERE NOT EXISTS (
              SELECT 1 FROM album_asset aa WHERE aa."albumId" = m.album_id AND aa."assetId" = m.asset_id)
        """)
        import psycopg2.extras
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT aa."albumId"::text AS album_id, aa."assetId"::text AS asset_id,
                       al."albumName" AS album_name, a."type"::text AS type,
                       ae.description AS description
                FROM album_asset aa
                JOIN album al ON al.id = aa."albumId"
                JOIN asset a ON a.id = aa."assetId"
                LEFT JOIN asset_exif ae ON ae."assetId" = a.id
                LEFT JOIN captioner_album_member m
                  ON m.album_id = aa."albumId" AND m.asset_id = aa."assetId"
                WHERE m.album_id IS NULL
            """)
            return list(cur.fetchall())

    def person_face_embeddings(self, person_name: str) -> List[List[float]]:
        deleted = 'AND af."deletedAt" IS NULL' if self.face_has_deleted_at else ""
        try:
            rows = self._exec(f"""
                SELECT fs.embedding::text FROM face_search fs
                JOIN asset_face af ON af.id = fs."faceId"
                JOIN person p ON {self.face_person_join}
                WHERE p.name = %s {deleted}
            """, (person_name,))
        except Exception as e:
            print(f"[route] could not read reference faces for {person_name}: {e}", flush=True)
            return []
        out = []
        for (txt,) in rows or []:
            try:
                out.append([float(x) for x in json.loads(txt)])
            except (ValueError, TypeError):
                continue
        print(f"[route] {len(out)} reference face(s) for {person_name}", flush=True)
        return out

# ----------------------------
# Main loop
# ----------------------------
def main():
    must_env("IMMICH_URL", IMMICH_URL)
    must_env("IMMICH_API_KEY", IMMICH_API_KEY)

    ocr_fn = load_florence_ocr()
    caption_detailed = load_joycaption()

    def caption_image(pil_image: Image.Image, person_names: Optional[List[str]] = None) -> Tuple[str, str]:
        ocr = ocr_fn(pil_image)
        detailed = caption_detailed(pil_image, person_names=person_names)
        if ocr_is_meaningful(ocr):
            return f"{ocr.strip()} | {detailed.strip()}", "OCR+DETAILED"
        return detailed, "DETAILED"

    conn = None
    if not USE_API_ONLY:
        print(f"[pg] Connecting to {PGHOST}:{PGPORT} db={PGDATABASE} user={PGUSER}", flush=True)
        conn = pg_connect()
        pg_ensure_skip_table(conn)
    else:
        print("[mode] Running in safe API-only mode (no DB access, no custom tables)", flush=True)

    # Routing state lives in Postgres whether or not candidate discovery uses it -- see the
    # "Upload routing" section. No credentials means no routing, not a crash.
    state: Optional[RoutingState] = None
    if ROUTING_ENABLED:
        try:
            state_conn = conn if conn is not None else (pg_connect() if PGPASSWORD else None)
            if state_conn is not None:
                state = RoutingState(state_conn)
                print("[route] upload routing and album-move handling enabled", flush=True)
        except Exception as e:
            print(f"[route] could not set up routing state: {e}", flush=True)
        if state is None:
            print("[route] no Postgres access -- routing disabled, captioning by album only", flush=True)

    total_done = 0

    def names_for(albums: List[str], people: List[str]) -> List[str]:
        """Album identities first (that system predates People tags and its prompts were
        tuned around it), then any other named People."""
        names = list(extract_identities_from_albums(albums))
        seen = {n.lower() for n in names}
        for name in people:
            if name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
        return names

    def finalize_caption(raw_caption: str, mode: str, albums: List[str]) -> Tuple[str, List[str], List[str]]:
        """Cleanup + identity handling shared by every path. Returns (caption, implied_tags,
        misfiled_identities); an empty caption means the model produced nothing usable."""
        caption = clean_caption(raw_caption)
        if not caption.strip():
            return "", [], []
        if mode != "VIDEO-PORN-COMPACT":
            caption = strip_false_nudity_leaks(caption, albums)
        if mode == "VIDEO-PORN-COMPACT":
            # The compact field format has no "the woman"/"she" prose to substitute a
            # name into, and it deliberately omits any person-reference wording when
            # the only identified person is Lydia (that's the whole point of skipping
            # her description) -- running this through the narrative-caption identity
            # logic would misread that as "nobody depicted" and incorrectly flag her as
            # misfiled.
            implied_tags, misfiled_identities = [], []
        else:
            caption, implied_tags, misfiled_identities = apply_identity_overrides(caption, albums)
        caption = " ".join(caption.split()).strip()[:MAX_CAPTION_CHARS]
        return caption, implied_tags, misfiled_identities

    def mark_empty_caption(asset_id: str) -> None:
        if not USE_API_ONLY:
            pg_mark_skip(conn, asset_id, "EMPTY_OR_JUNK_CAPTION")
        print(f"[skip] {asset_id} produced empty/junk caption (marked skip)", flush=True)

    def recaption(
        asset_id: str,
        asset_type: str,
        albums: List[str],
        prefetched_thumbnail: Optional[Image.Image] = None,
        prefetched_thumbnail_error: Optional[Exception] = None,
        exif_make: Optional[str] = None,
        gen_info: Optional[str] = None,
        auto_file: bool = True,
        misfile_cleanup: bool = True,
    ) -> None:
        """Caption an asset by the albums it's already in -- the pre-routing behavior, used
        for everything that isn't a brand-new upload. auto_file=False skips the caption-
        keyword album filing (furry/lactation/hucow/Camspy); misfile_cleanup=False keeps the
        "no person in the caption" heuristic from undoing a face-recognition filing."""
        nonlocal total_done
        person_names = names_for(albums, get_asset_people_names(asset_id))

        # Computed for every asset type, not just video -- the auto-filing rules below
        # apply to images too, and non-human content must be excluded from them there
        # as well.
        feral = is_feral_album(albums)
        nonhuman = is_nonhuman_album(albums)

        if asset_type == "VIDEO":
            dense = is_dense_sampling_album(albums)
            compilation = is_compilation_album(albums)
            multiple = is_multiple_creampie_album(albums)
            raw_caption, mode = caption_video(
                asset_id, caption_detailed, person_names=person_names, dense=dense,
                compilation=compilation, feral=feral, multiple=multiple,
                guaranteed_multi=is_guaranteed_multi_album(albums),
                single=is_single_creampie_album(albums), nonhuman=nonhuman,
                full_caption=is_full_caption_album(albums),
                masturbation=is_masturbation_album(albums),
                categorized=is_categorized_album(albums),
            )
        else:
            if prefetched_thumbnail_error is not None:
                raise prefetched_thumbnail_error
            img = prefetched_thumbnail if prefetched_thumbnail is not None else immich_get_thumbnail(asset_id)
            raw_caption, mode = caption_image(img, person_names=person_names)
            generate_and_apply_e621_tags(asset_id, img, caption_detailed)

        caption, implied_tags, misfiled_identities = finalize_caption(raw_caption, mode, albums)
        if not caption:
            mark_empty_caption(asset_id)
            return

        # Truncation above applies to the caption alone -- the generation info is
        # re-attached afterwards so MAX_CAPTION_CHARS can never clip the JSON.
        ok = immich_update_description(asset_id, compose_description(caption, gen_info))
        if not ok:
            print(f"[fail] {asset_id} update failed", flush=True)
            return
        total_done += 1
        alb = ", ".join(albums[:3]) + ("..." if len(albums) > 3 else "")
        print(f"[ok] {asset_id} [{mode}] albums=[{alb}] => {caption}", flush=True)

        if implied_tags:
            immich_apply_tags(asset_id, implied_tags)

        if misfiled_identities and misfile_cleanup:
            for name in misfiled_identities:
                for album_name in find_albums_matching_identity(albums, name):
                    album_id = immich_album_id_by_name(album_name)
                    if album_id:
                        immich_remove_from_album(asset_id, album_id)
            immich_unarchive(asset_id)
            print(f"[misfile] {asset_id} not actually {'/'.join(misfiled_identities)} -- removed from identity album(s), unarchived", flush=True)

        # "Please categorize" assets are deliberately left exactly as they are --
        # unfiled, unarchived, and sitting in the main timeline waiting for the human.
        # Auto-filing or archiving them here would defeat the entire point.
        if mode == "VIDEO-UNCATEGORIZED":
            print(f"[parked] {asset_id} awaiting manual categorization", flush=True)
        elif auto_file:
            # Single AND Multiple Creampie membership are both purely manual now: the
            # captioner reports what it detected in the caption ("Separate Creampies |
            # 1 (~12:02)") but never files the asset into either album, and never
            # removes it from either. The human reads the caption and sorts.
            #
            # Auto-filing into Single Creampie is what put 80 wrongly-classified
            # videos there -- 56 of them from Bondage Creampie -- because any asset
            # already sitting in some album skipped the "Please categorize" park and
            # went straight through detection into filing. Detection is good enough to
            # inform a decision, not good enough to make one unattended.

            # Feral is the one category that belongs in no additional album at all:
            # it's a real, non-anthropomorphic animal, the opposite of furry, and its
            # own feral albums are the whole classification. Everything below is
            # skipped for it -- anthro, by contrast, is expected to live in Furry Stuff.
            if not feral:
                # Membership in an identity album is a deliberate human filing
                # decision, so don't second-guess it off a caption keyword. Lydia
                # stylized as a dog captions as "anthropomorphic dog", which would
                # otherwise sweep that entire album into Furry Stuff and archive it
                # out of the timeline.
                if _FURRY_TRIGGER_RE.search(caption) and not extract_identities_from_albums(albums):
                    immich_add_to_album(asset_id, FURRY_ALBUM_ID)
                    immich_archive(asset_id)

                if _LACTATION_TRIGGER_RE.search(caption):
                    immich_add_to_album(asset_id, LACTATION_ALBUM_ID)

                if _HUCOW_TRIGGER_RE.search(caption):
                    immich_add_to_album(asset_id, HUCOW_ALBUM_ID)

            # Ray-Ban Meta glasses capture belongs in Camspy regardless of what else
            # it is -- keyed off EXIF make, which the DB/API candidate fetch supplies.
            if (exif_make or "").strip().lower() == CAMSPY_EXIF_MAKE:
                immich_add_to_album(asset_id, CAMSPY_ALBUM_ID)

        if STAMP_PORN_CAPTION_DATE and mode == "VIDEO-PORN-COMPACT":
            immich_set_date_taken_now(asset_id)

    def route_new_asset(
        asset_id: str,
        asset_type: str,
        filename: Optional[str],
        exif_make: Optional[str],
        gen_info: Optional[str],
        thumb: Optional[Image.Image],
    ) -> Optional[str]:
        """Walk a brand-new upload through the routing order. Returns the outcome, or None
        when it's waiting on face recognition and will be picked up again later."""
        is_video = asset_type == "VIDEO"
        added_keys: List[str] = []

        def add(key: str) -> None:
            album_id = album_id_for(key)
            if not album_id or key in added_keys:
                return
            immich_add_to_album(asset_id, album_id)
            state.record_membership(album_id, asset_id)
            added_keys.append(key)

        def finish(raw_caption: str, mode: str, outcome: str, archive: bool,
                   caption_override: Optional[str] = None) -> str:
            nonlocal total_done
            albums = refresh_asset_albums(asset_id, [])
            if caption_override is not None:
                caption = caption_override
            else:
                # Misfile cleanup is deliberately ignored here: every album this path adds
                # was chosen by looking at the asset, not by a human who might have slipped.
                caption, _, _ = finalize_caption(raw_caption, mode, albums)
            if not caption:
                mark_empty_caption(asset_id)
            elif immich_update_description(asset_id, compose_description(caption, gen_info)):
                total_done += 1
                print(f"[ok] {asset_id} [{mode}] routed={outcome} albums=[{', '.join(albums)}] "
                      f"archived={archive} => {caption}", flush=True)
                if STAMP_PORN_CAPTION_DATE and mode == "VIDEO-PORN-COMPACT":
                    immich_set_date_taken_now(asset_id)
            else:
                print(f"[fail] {asset_id} update failed", flush=True)
                return outcome
            if archive:
                immich_archive(asset_id)
            state.mark_routed(asset_id, outcome)
            return outcome

        if not is_video and thumb is None:
            thumb = immich_get_thumbnail(asset_id)

        # 1. CamSpy: Ray-Ban Meta capture, or a SpyPhoto file. Full narrative caption.
        if is_camspy_upload(exif_make, filename):
            people = get_asset_people_names(asset_id)
            add("camspy")
            if is_video:
                raw, mode = caption_video(asset_id, caption_detailed, person_names=people, full_caption=True)
            else:
                raw, mode = caption_image(thumb, person_names=people)
            return finish(raw, mode, "camspy", archive=True)

        # 2. One of us? Nothing else happens until Immich's face recognition has had its go.
        status = state.face_status(asset_id)
        if status == "wait":
            state.defer(asset_id)
            print(f"[route] {asset_id} waiting on face recognition", flush=True)
            return None
        if status == "timeout":
            print(f"[route] {asset_id} face recognition didn't finish in time -- continuing", flush=True)
        raw_people = get_asset_people_names(asset_id, raw=True)
        people = [canonical_people_name(n) for n in raw_people]
        matches = person_albums(raw_people)
        if matches:
            for person, album_name, album_id in matches:
                immich_add_to_album(asset_id, album_id)
                state.record_membership(album_id, asset_id)
                print(f"[route] {asset_id} is {person} -> {album_name}", flush=True)
            albums = refresh_asset_albums(asset_id, [m[1] for m in matches])
            recaption(asset_id, asset_type, albums, prefetched_thumbnail=thumb,
                      exif_make=exif_make, gen_info=gen_info, auto_file=False,
                      misfile_cleanup=False)
            state.mark_routed(asset_id, "person")
            return "person"

        video_path = None
        try:
            if is_video:
                fd, video_path = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)
                immich_download_original(asset_id, video_path)
                frames = extract_video_frames(video_path, dense=False)
                if not frames:
                    raise RuntimeError("no frames extracted")
            else:
                frames = [(0.0, thumb)]
            triage = _classify_with_prompt([img for _, img in frames], _TRIAGE_PROMPT,
                                           _parse_triage, caption_detailed, max_new_tokens=40)

            # 3. Anthro. Everything here ends in its regular caption and the archive.
            if frames_agree(triage, "anthro"):
                add("furry")
                if is_video:
                    add("anthro_video")
                    if frames_agree(triage, "sex_human_anthro"):
                        add("human_anthro_video")
                    if frames_agree(triage, "sex_anthro_anthro"):
                        add("anthro_sex_video")
                    albums = refresh_asset_albums(asset_id, [])
                    raw, mode = caption_video(
                        asset_id, caption_detailed, person_names=names_for(albums, people),
                        dense=is_dense_sampling_album(albums), nonhuman=True, categorized=True,
                        video_path=video_path,
                    )
                else:
                    if frames_agree(triage, "sex_human_anthro"):
                        add("human_anthro_still")
                    if frames_agree(triage, "cow"):
                        add("cow_anthro")
                    if matches_lydiadog(asset_id, gen_info, raw_people, state):
                        add("lydia_dog")
                        try:
                            tag_lydiadog(asset_id, state)
                        except Exception as e:
                            print(f"[tag] {asset_id} LydiaDog tagging failed: {e}", flush=True)
                    albums = refresh_asset_albums(asset_id, [])
                    raw, mode = caption_image(thumb, person_names=names_for(albums, people))
                return finish(raw, mode, "anthro", archive=True)

            # 4. Nudity. Without it: the regular caption, left in the timeline, done.
            nude_rows = [(ts, img) for (ts, img), t in zip(frames, triage) if t["nudity"]]
            if not nude_rows:
                if is_video:
                    raw, mode = caption_video(asset_id, caption_detailed, person_names=people,
                                              full_caption=True, video_path=video_path)
                else:
                    raw, mode = caption_image(thumb, person_names=people)
                return finish(raw, mode, "clean", archive=False)

            # 5. Porn categories.
            cats = _classify_with_prompt([img for _, img in _evenly(nude_rows, PORN_PROMPT_MAX_FRAMES)],
                                         _PORN_CATEGORY_PROMPT, _parse_porn_categories,
                                         caption_detailed, max_new_tokens=80)
            solo = (max((c["women"] for c in cats), default=0) == 1
                    and not frames_agree(cats, "men"))
            if solo:
                add("internet_titties")
                if frames_agree(cats, "masturbating"):
                    add("masturbation")
            if frames_agree(cats, "cowprint"):
                add("hucow")
            if frames_agree(cats, "lactating"):
                add("lactation")
            if frames_agree(cats, "glorywall"):
                add("glorywall")

            text = filename or ""
            if is_video:
                card_frames = extract_frames_at(video_path, TITLECARD_TIMESTAMPS)
                text += " " + " ".join(ocr_fn(img) for _, img in card_frames + frames)
            else:
                text += " " + ocr_fn(thumb)
            hentaied = frames_agree(cats, "tentacles") or bool(_HENTAIED_TEXT_RE.search(_compact_text(text)))

            if not is_video:
                if hentaied:
                    add("hentaied")
                raw, mode = caption_image(thumb, person_names=people)
                # A creampie keeps it in the timeline until the human has sorted it.
                archive = hentaied or (any(k in _PORN_ALBUM_KEYS for k in added_keys)
                                       and not frames_agree(cats, "vaginal_cum"))
                return finish(raw, mode, "hentaied" if hentaied else "porn", archive=archive)

            dense_frames = extract_video_frames(video_path, dense=True)
            signals = _classify_video_frames(dense_frames, caption_detailed)
            if not any(k == "lactation" for k in added_keys) and frames_agree(signals, "lactating"):
                add("lactation")
            masturbation = "masturbation" in added_keys

            def porn_id(count: int = 0, event_times: Optional[List[str]] = None) -> str:
                return compact_porn_caption(signals, dense_frames, caption_detailed,
                                            person_names=people, masturbation=masturbation,
                                            count=count, event_times=event_times)

            if hentaied:
                add("hentaied")
                return finish("", "VIDEO-PORN-COMPACT", "hentaied", archive=True,
                              caption_override=clean_caption(porn_id()))

            studio = match_studio(text)
            if studio:
                add(studio)
                add("multi")
                count, event_times = count_creampie_events(
                    [(s["ts"], s["state"], s["partner_visible"]) for s in signals],
                    min_count=MULTI_CREAMPIE_MIN_COUNT)
                print(f"[route] {asset_id} studio {studio}: {count} creampie(s)", flush=True)
                return finish("", "VIDEO-PORN-COMPACT", "studio", archive=True,
                              caption_override=clean_caption(porn_id(count, event_times)))

            # No cum classification in the caption at upload -- but a creampie parks the
            # video at "Please Categorize" in the timeline for the human to sort.
            caption = clean_caption(porn_id())
            if detect_single_creampie(signals) is not None:
                if any(s["bound"] for s in signals):
                    add("bondage_creampie")
                return finish("", "VIDEO-PORN-COMPACT", "please-categorize", archive=False,
                              caption_override=with_uncategorized_prefix(caption))
            archive = any(k in _PORN_ALBUM_KEYS for k in added_keys)
            return finish("", "VIDEO-PORN-COMPACT", "porn", archive=archive, caption_override=caption)
        finally:
            if video_path:
                try:
                    os.remove(video_path)
                except OSError:
                    pass

    def run_cum_counter(asset_id: str, min_count: int = 0) -> Tuple[int, List[str]]:
        fd, video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            immich_download_original(asset_id, video_path)
            frames = extract_video_frames(video_path, dense=True)
            if not frames:
                raise RuntimeError("no frames extracted")
            signals = _classify_video_frames(frames, caption_detailed)
            return count_creampie_events(
                [(s["ts"], s["state"], s["partner_visible"]) for s in signals], min_count=min_count)
        finally:
            try:
                os.remove(video_path)
            except OSError:
                pass

    move_attempts: Dict[str, int] = {}
    last_move_poll = [0.0]

    def handle_album_moves() -> None:
        """Act on albums the human just filed things into. See the "Upload routing" section."""
        if state is None or time.time() - last_move_poll[0] < MOVE_POLL_SECONDS:
            return
        last_move_poll[0] = time.time()
        try:
            rows = state.new_memberships()
        except Exception as e:
            print(f"[move] membership poll failed: {e}", flush=True)
            return
        if not rows:
            return
        multi_id, single_id = album_id_for("multi"), album_id_for("single")
        by_asset: Dict[str, List[dict]] = {}
        for row in rows:
            by_asset.setdefault(row["asset_id"], []).append(row)

        for asset_id, adds in by_asset.items():
            names = [r["album_name"] or "" for r in adds]
            to_multi = any(r["album_id"] == multi_id for r in adds)
            to_single = any(r["album_id"] == single_id for r in adds)
            to_counted = any(n.strip().startswith(MULTI_EVENT_COUNT_ALBUM_PREFIXES) for n in names)
            to_porn = any((album_number(n) or "").startswith("200.") for n in names)
            to_lydiadog = bool(LYDIADOG_ALBUM_PREFIX) and any(
                n.strip().startswith(LYDIADOG_ALBUM_PREFIX) for n in names)
            is_video = (adds[0]["type"] or "").upper() == "VIDEO"
            caption, gen = split_description(adds[0]["description"])
            try:
                if to_lydiadog and not is_video:
                    tag_lydiadog(asset_id, state)
                if not caption:
                    # Still waiting for its caption: the normal pass captions it by album
                    # (and counts creampies for Multiple Creampie), so only the archive
                    # decision is made here.
                    if to_multi or to_single:
                        immich_archive(asset_id)
                else:
                    was_parked = has_uncategorized_prefix(caption)
                    new = strip_uncategorized_prefix(caption)
                    if (to_multi or to_counted) and is_video:
                        count, event_times = run_cum_counter(
                            asset_id, min_count=MULTI_CREAMPIE_MIN_COUNT if to_multi else 0)
                        new = with_creampie_count(new, count, event_times)
                        print(f"[move] {asset_id} CumCounter: {count} creampie(s)", flush=True)
                    if new != caption:
                        # An emptied caption (it was nothing but "Please Categorize") puts the
                        # asset back in the queue, to be captioned by its new album.
                        immich_update_description(asset_id, compose_description(new, gen))
                    if to_multi or to_single or (was_parked and to_porn):
                        immich_archive(asset_id)
                print(f"[move] {asset_id} added to [{', '.join(names)}] handled", flush=True)
            except Exception as e:
                move_attempts[asset_id] = move_attempts.get(asset_id, 0) + 1
                print(f"[move] {asset_id} failed (attempt {move_attempts[asset_id]}): {e}", flush=True)
                if move_attempts[asset_id] < MOVE_MAX_ATTEMPTS:
                    continue
                print(f"[move] {asset_id} giving up after {MOVE_MAX_ATTEMPTS} attempts", flush=True)
            move_attempts.pop(asset_id, None)
            for r in adds:
                state.record_membership(r["album_id"], asset_id)

    def process_candidate(
        asset_id: str,
        asset_type: str,
        albums: List[str],
        prefetched_thumbnail: Optional[Image.Image] = None,
        prefetched_thumbnail_error: Optional[Exception] = None,
        exif_make: Optional[str] = None,
        gen_info: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        try:
            if asset_type == "VIDEO" and not CAPTION_VIDEOS:
                if not USE_API_ONLY:
                    pg_mark_skip(conn, asset_id, "SKIP_VIDEO")
                print(f"[skip] {asset_id} is VIDEO (skipping)", flush=True)
                return

            # Re-read album membership as late as possible -- see refresh_asset_albums().
            albums = refresh_asset_albums(asset_id, albums)

            if state is not None and not albums and not state.is_routed(asset_id):
                if prefetched_thumbnail_error is not None:
                    raise prefetched_thumbnail_error
                route_new_asset(asset_id, asset_type, filename, exif_make, gen_info,
                                prefetched_thumbnail)
            else:
                recaption(asset_id, asset_type, albums, prefetched_thumbnail,
                          prefetched_thumbnail_error, exif_make, gen_info)

            time.sleep(SLEEP_SECONDS)

        except ThumbnailNotFound as e:
            if not USE_API_ONLY:
                pg_mark_skip(conn, asset_id, "THUMBNAIL_404")
            print(f"[skip] {asset_id}: {e} (marked skip)", flush=True)
            time.sleep(0.2)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if not USE_API_ONLY:
                pg_mark_skip(conn, asset_id, f"HTTP_ERROR_{status}")
            print(f"[skip] {asset_id}: HTTP {status} fetching asset (marked skip)", flush=True)
            time.sleep(0.2)

        except RuntimeError as e:
            # caption_video raises this specific message for a video with no
            # decodable frames (e.g. a truncated/failed upload). Unlike other
            # RuntimeErrors it's not transient -- retrying never helps -- but
            # without this it retried forever with no backoff (asset
            # 7e610ef6-7054-426f-a5af-f69c439d0d7d: 44k+ retries, ~51/min,
            # 13.5h straight, GPU pinned at 12.5GB idle -- see 2026-09-14
            # weekly log review).
            if str(e) == "no frames extracted":
                if not USE_API_ONLY:
                    pg_mark_skip(conn, asset_id, "NO_FRAMES_EXTRACTED")
                print(f"[skip] {asset_id}: no frames extracted (marked skip)", flush=True)
                time.sleep(0.2)
            else:
                print(f"[error] {asset_id}: {e}", flush=True)
                time.sleep(1.0)

        except Exception as e:
            print(f"[error] {asset_id}: {e}", flush=True)
            time.sleep(1.0)

    if USE_API_ONLY:
        while True:
            handle_album_moves()
            candidates = get_uncaptioned_candidates_api()
            if not candidates:
                print(f"[done] No more blank assets. Sleeping {IDLE_SLEEP_SECONDS}s and rechecking...", flush=True)
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            print(f"[batch] {len(candidates)} candidates", flush=True)
            deferred = state.deferred_ids() if state is not None else set()
            for row in candidates:
                handle_album_moves()
                asset_id = row.get("id")
                if asset_id in deferred:
                    continue
                asset_type = row.get("type", "UNKNOWN").upper()  # API may not have type; fallback
                # Albums are read inside process_candidate (refresh_asset_albums), as late
                # as possible, so there's nothing to fetch here.
                exif_info = row.get("exifInfo") or {}
                process_candidate(
                    asset_id, asset_type, [],
                    exif_make=exif_info.get("make"),
                    gen_info=split_description(exif_info.get("description"))[1],
                    filename=row.get("originalFileName"),
                )

            print(f"[progress] total updated this run: {total_done}", flush=True)

    # DB-direct mode: a background thread stays one candidate ahead of the GPU -- while
    # process_candidate() is busy running generation on the current asset, this thread
    # fetches the next DB candidate and (for images) downloads its thumbnail, so that
    # network round-trip happens off the GPU's critical path instead of stalling it every
    # single item. The queue is deliberately maxsize=1: only ever one item prefetched
    # ahead, so priority reordering (freshly-cleared images jumping the queue) stays just
    # as fresh as the old synchronous DB_REPRIORITIZE_BATCH=1 behavior.
    #
    # _in_flight_ids guards against the prefetch thread re-fetching the same candidate
    # twice before its caption has actually been written back (the DB row still looks
    # like a valid candidate -- empty description -- right up until process_candidate()
    # finishes and calls immich_update_description).
    prefetch_q: "queue.Queue" = queue.Queue(maxsize=1)
    in_flight_lock = threading.Lock()
    in_flight_ids: set = set()

    def _prefetch_worker():
        worker_conn = pg_connect()
        try:
            while True:
                try:
                    rows = pg_fetch_candidates(worker_conn, DB_REPRIORITIZE_BATCH,
                                               exclude_face_wait=state is not None)
                except Exception as e:
                    prefetch_q.put(("fetch_error", e))
                    time.sleep(1.0)
                    continue

                if not rows:
                    prefetch_q.put(("idle", None))
                    time.sleep(IDLE_SLEEP_SECONDS)
                    continue

                for row in rows:
                    asset_id = str(row["id"])
                    # pg_fetch_candidates() matches any JSON-looking description so Postgres
                    # doesn't have to reason about generation-info keys; reject the ones that
                    # turned out to be an unrelated blob (a real, already-written caption)
                    # before they cost a GPU pass.
                    if not description_is_captionable(row.get("description")):
                        continue
                    with in_flight_lock:
                        if asset_id in in_flight_ids:
                            continue
                        in_flight_ids.add(asset_id)

                    asset_type = (row.get("type") or "").upper()
                    albums = row.get("albums") or []
                    thumb, thumb_err = None, None
                    if asset_type != "VIDEO":
                        try:
                            thumb = immich_get_thumbnail(asset_id)
                        except Exception as e:
                            thumb_err = e

                    prefetch_q.put(("row", (asset_id, asset_type, albums, thumb, thumb_err,
                                            row.get("exif_make"),
                                            split_description(row.get("description"))[1],
                                            row.get("original_file_name"))))
        finally:
            worker_conn.close()

    threading.Thread(target=_prefetch_worker, daemon=True, name="prefetch").start()
    print("[prefetch] background lookahead thread started", flush=True)

    while True:
        handle_album_moves()
        kind, payload = prefetch_q.get()

        if kind == "fetch_error":
            print(f"[error] prefetch DB fetch failed: {payload}", flush=True)
            time.sleep(1.0)
            continue

        if kind == "idle":
            print(f"[done] No more blank assets. Sleeping {IDLE_SLEEP_SECONDS}s and rechecking...", flush=True)
            continue

        asset_id, asset_type, albums, thumb, thumb_err, exif_make, gen_info, filename = payload
        try:
            process_candidate(asset_id, asset_type, albums, prefetched_thumbnail=thumb,
                              prefetched_thumbnail_error=thumb_err, exif_make=exif_make,
                              gen_info=gen_info, filename=filename)
        finally:
            with in_flight_lock:
                in_flight_ids.discard(asset_id)

        print(f"[progress] total updated this run: {total_done}", flush=True)


if __name__ == "__main__":
    main()
