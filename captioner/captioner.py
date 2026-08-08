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

# Auto-filing: any video with a detected creampie, regardless of which album it's manually
# sorted into, gets added here (in addition to its existing albums, never removing it from
# anywhere) -- unless it's already been manually placed in Multiple Creampie, which the
# captioner never adds to or removes from (see is_multiple_creampie_album).
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
# Every porn-section prefix: 100 (people I know IRL / Lydia), 200 (creampie & friends),
# 300 (furry/feral), 400 (camspy / game porn), 500 (hotwife / hookers). Membership in any of
# them means the human has already filed it, so it gets captioned rather than parked.
CATEGORIZED_ALBUM_PREFIXES = tuple(
    p.strip()
    for p in os.environ.get("CATEGORIZED_ALBUM_PREFIXES", "100.,200.,300.,400.,500.").split(",")
    if p.strip()
)
FULL_CAPTION_ALBUM_KEYWORDS = [
    k.strip().lower()
    for k in os.environ.get("FULL_CAPTION_ALBUMS", "camspy,lv hookers").split(",")
    if k.strip()
]
UNCATEGORIZED_CAPTION = os.environ.get("UNCATEGORIZED_CAPTION", "Please categorize")

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

def find_albums_matching_identity(albums: List[str], identity_name: str) -> List[str]:
    matches: List[str] = []
    for album in albums or []:
        if not album:
            continue
        for album_kw, rx in _IDENTITY_ALBUM_REGEXES.items():
            if _IDENTITY_MAP.get(album_kw) == identity_name and rx.search(album):
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

def apply_identity_overrides(caption: str, albums: List[str]) -> Tuple[str, List[str], List[str]]:
    """Returns (updated_caption, implied_tags, misfiled_identities)."""
    if not caption:
        return caption, [], []
    identities = extract_identities_from_albums(albums)
    if not identities:
        return caption, [], []

    if not _PERSON_WORD_RE.search(caption):
        # Nobody appears to be depicted at all -- don't force any identity name into the
        # caption. Flag every expected identity so the caller can clean up the misfile.
        return caption, [], identities

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
    result = subprocess.run(
        [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, TypeError):
        return 0.0

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
def is_multiple_creampie_album(albums: List[str]) -> bool:
    # The whole 200.000.xxx range is multiple-creampie content: the base album plus its
    # studio/kink sub-albums (Puta Locura, Creampie Squad, Gangbang, Slutwife, Czech,
    # Orgy, Hentaied...). Being filed into any of them is the human saying "this is a
    # multi", so event counting runs for all of them.
    return any((album or "").strip().startswith(MULTI_CREAMPIE_PREFIX) for album in (albums or []))

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
    return any(
        (album or "").strip().startswith(CATEGORIZED_ALBUM_PREFIXES)
        for album in (albums or [])
    )

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

def extract_video_frames(video_path: str, dense: bool = False) -> List[Tuple[float, Image.Image]]:
    duration = probe_duration_seconds(video_path)
    timestamps = compute_dense_timestamps(duration) if dense else compute_video_timestamps(duration)

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
    min_gap_seconds: float = 60.0,
) -> Tuple[int, List[str]]:
    # frame_states is already in chronological sample order: (timestamp, STATE, partner_visible)
    # per frame, where STATE is one of INSERTED/CUM/NONE and partner_visible says whether a
    # male partner is anywhere in frame with her right now. A single still frame can't show
    # the *instant* of pulling out -- that's motion, not a static visual state -- so we don't
    # ask the model to recognize that moment directly, and counting every fresh "cum visible"
    # sighting as its own event doesn't work either: cum can go in and out of view purely from
    # a camera angle or position change with no new ejaculation involved.
    #
    # Only reached for content manually placed in Multiple Creampie and not also in a
    # compilation album (compilation membership skips creampie detection entirely -- see
    # caption_video) -- i.e. genuinely "one woman, multiple creampies in one continuous
    # encounter," never a compilation of different women cut together.
    #
    # "Multiple creampies" specifically means multiple DIFFERENT men, not just multiple times
    # cum became visible -- ordinary continued sex after the real creampie (same man, still in
    # frame) is completely normal and must not count as a second round, no matter how many
    # times the existing load becomes visible again during it. A single "partner absent" frame
    # alone isn't reliable evidence he actually left, though -- porn cinematography constantly
    # crops the man's body out of frame for close-ups even mid-scene with the same guy the
    # whole time (confirmed in testing: using that signal alone made a known-bad case *worse*,
    # 12 events up to 18). So a new event requires BOTH real elapsed time (min_gap_seconds)
    # since the last counted event AND at least one sampled frame where she was genuinely alone
    # during that gap -- either condition alone was too permissive on its own.
    event_starts_raw: List[float] = []
    was_cum = False
    seen_alone_since_last_event = False
    for ts, state, partner_visible in frame_states:
        if not partner_visible:
            seen_alone_since_last_event = True
        if state == "INSERTED":
            was_cum = False
        elif state == "CUM":
            if not was_cum:
                had_scene_break = (
                    not event_starts_raw
                    or (
                        seen_alone_since_last_event
                        and (ts - event_starts_raw[-1]) >= min_gap_seconds
                    )
                )
                if had_scene_break:
                    event_starts_raw.append(ts)
                    seen_alone_since_last_event = False
            was_cum = True
        else:
            was_cum = False
    return len(event_starts_raw), [format_ts(ts) for ts in event_starts_raw]

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
    "STATE: one of INSERTED, CUM, or NONE. If GENITALS is N, STATE must always be NONE --  "
    "you cannot see a creampie in a frame that doesn't show her vagina at all, no matter what "
    "else is visible. A creampie is ONLY a penis or toy ejaculating inside her VAGINA -- cum "
    "in her mouth, on her face, or on/in her ass/anus does NOT count, no matter how it looks. "
    "Milk leaking/dripping/spraying from her nipples is NEVER cum, even though both are thin "
    "white liquid -- milk on her breasts/chest is never STATE, it's the separate LACTATING "
    "field below. INSERTED if a cock/toy is actively penetrating her vagina right now. CUM if "
    "an actual visible load of cum/jizz is on or dripping from her vagina right now AND "
    "nothing is currently penetrating it. Do NOT confuse lubricant with cum -- this is a "
    "common mistake. Real cum is thick and white, is actively dripping or pooling out of her, "
    "and does NOT cling to the penis/toy shaft. Lube is ALSO often thick and white, but it "
    "stays smeared/coating the penis or toy shaft itself (visible on the shaft when it's "
    "pulled out or during thrusting), and is typically present continuously throughout the "
    "scene rather than only appearing after a period of sex. If the white fluid you see is "
    "coating the shaft, or has been visible the whole time rather than only showing up after "
    "real thrusting, that's lube -- answer NONE, not CUM. Her own natural arousal wetness "
    "(thin, clear, glossy) also doesn't count as CUM. NONE otherwise -- this also includes "
    "oral sex, a facial/cum on her face or in her mouth, cum on/in her ass, and anal "
    "penetration; none of those are INSERTED or CUM, they're always NONE for this field.\n"
    "BOUND: Y or N -- Y if she is visibly tied up, chained, cuffed, or otherwise physically "
    "restrained right now. Otherwise N.\n"
    "SPECIES: NONE, or a single animal name (e.g. dog, horse) if a REAL (photographic, "
    "non-illustrated, non-anthropomorphic) animal is sexually engaging with her right now. "
    "Illustrated/anime/furry/anthro humanoid-animal characters do NOT count -- answer NONE "
    "for those, and NONE if only humans are involved.\n"
    "LACTATING: Y or N -- Y if milk is visibly leaking, dripping, or spraying from her bare "
    "nipples/breasts right now. Otherwise N.\n"
    "Example full answer:\nNUDITY: Y\nPARTNER: Y\nGENITALS: Y\nSTATE: CUM\nBOUND: N\nSPECIES: NONE\nLACTATING: N"
)

def _parse_video_signal(text: str) -> dict:
    t = text.upper()
    if "TITLECARD" in t:
        return {
            "titlecard": True, "nudity": False, "partner_visible": False, "state": "NONE",
            "bound": False, "species": None, "lactating": False,
        }
    nudity = bool(re.search(r"NUDITY\s*:\s*Y", t))
    partner_visible = bool(re.search(r"PARTNER\s*:\s*Y", t))
    genitals_visible = bool(re.search(r"GENITALS\s*:\s*Y", t))
    if re.search(r"STATE\s*:\s*INSERTED", t):
        state = "INSERTED"
    elif re.search(r"STATE\s*:\s*CUM", t):
        state = "CUM"
    else:
        state = "NONE"
    # Defense in depth: even though the prompt tells the model STATE must be NONE when
    # GENITALS is N, it doesn't reliably comply -- confirmed in testing, a frame showing
    # milk streaming from a nipple with no genitals anywhere in view was still read as
    # STATE: CUM. Overriding here doesn't depend on the model getting that instruction right.
    if not genitals_visible:
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
        "bound": bound, "species": species, "lactating": lactating,
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
COMPACT_DESC_SKIP_NAMES = {"Lydia"}

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

def caption_video(
    asset_id: str,
    caption_detailed,
    person_names: Optional[List[str]] = None,
    dense: bool = False,
    compilation: bool = False,
    feral: bool = False,
    multiple: bool = False,
    nonhuman: bool = False,
    full_caption: bool = False,
    masturbation: bool = False,
    categorized: bool = True,
) -> Tuple[str, str]:
    fd, video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
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

        if nonhuman or compilation or feral:
            # No creampie count at all for non-human content (anthro/furry/feral) -- the
            # detection was built around live-action photography and doesn't reliably apply
            # to illustrated art.
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
            cum_ts = None
            seen_insertion = False
            seen_partner = False
            for s in signals:
                if s["partner_visible"]:
                    seen_partner = True
                if s["state"] == "INSERTED":
                    seen_insertion = True
                elif s["state"] == "CUM" and seen_insertion and seen_partner:
                    cum_ts = s["ts"]
            count, event_times = (1, [format_ts(cum_ts)]) if cum_ts is not None else (0, [])
        else:
            count, event_times = count_creampie_events(
                [(s["ts"], s["state"], s["partner_visible"]) for s in signals]
            )
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
        return caption, "VIDEO-PORN-COMPACT"
    finally:
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

def immich_list_albums() -> List[dict]:
    global _album_list_cache
    if _album_list_cache is not None:
        return _album_list_cache
    try:
        r = requests.get(f"{IMMICH_URL}/api/albums", headers=immich_headers(), timeout=60)
        r.raise_for_status()
        _album_list_cache = r.json()
    except Exception as e:
        print(f"[album] list failed: {e}", flush=True)
        return []
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
                if not desc or desc.strip() == "":
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

def get_asset_albums(asset_id: str) -> List[str]:
    """Fetch album names for an asset via API (needed in API-only mode).

    GET /api/assets/{id} does NOT include album membership on this Immich version
    (verified empirically -- AssetResponseDto has no "albums" field). The correct
    endpoint is GET /api/albums?assetId={id}, which returns the list of albums
    directly (each with an "albumName" field).
    """
    try:
        url = f"{IMMICH_URL}/api/albums"
        r = requests.get(url, headers=immich_headers(), params={"assetId": asset_id}, timeout=30)
        r.raise_for_status()
        data = r.json()
        return [album.get("albumName") for album in data if album.get("albumName")]
    except Exception as e:
        print(f"[api] Failed to fetch albums for {asset_id}: {e}", flush=True)
        return []

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

def pg_fetch_candidates(conn, limit: int) -> List[dict]:
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

    sql = f"""
    SELECT
      a.id as id,
      {select_type}
      ae.description as description,
      ae.make as exif_make,
      COALESCE(array_remove(array_agg(al."albumName"), NULL), '{{}}'::text[]) as albums
    FROM asset a
    JOIN asset_exif ae ON ae."assetId" = a.id
    LEFT JOIN captioner_skip cs ON cs.asset_id = a.id
    LEFT JOIN album_asset aa ON aa."assetId" = a.id
    LEFT JOIN album al ON al.id = aa."albumId"
    WHERE
      cs.asset_id IS NULL
      AND (ae.description IS NULL OR btrim(ae.description) = '')
    GROUP BY a.id, ae.description, ae.make {', a."type"' if has_type else ''}
    {order_clause}
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

    total_done = 0

    def process_candidate(
        asset_id: str,
        asset_type: str,
        albums: List[str],
        prefetched_thumbnail: Optional[Image.Image] = None,
        prefetched_thumbnail_error: Optional[Exception] = None,
        exif_make: Optional[str] = None,
    ) -> None:
        nonlocal total_done
        try:
            if asset_type == "VIDEO" and not CAPTION_VIDEOS:
                if not USE_API_ONLY:
                    pg_mark_skip(conn, asset_id, "SKIP_VIDEO")
                print(f"[skip] {asset_id} is VIDEO (skipping)", flush=True)
                return

            person_names = extract_identities_from_albums(albums)

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
                    compilation=compilation, feral=feral, multiple=multiple, nonhuman=nonhuman,
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

            caption = clean_caption(raw_caption)
            if not caption.strip():
                if not USE_API_ONLY:
                    pg_mark_skip(conn, asset_id, "EMPTY_OR_JUNK_CAPTION")
                print(f"[skip] {asset_id} produced empty/junk caption (marked skip)", flush=True)
                return

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

            ok = immich_update_description(asset_id, caption)
            if ok:
                total_done += 1
                alb = ", ".join(albums[:3]) + ("..." if len(albums) > 3 else "")
                print(f"[ok] {asset_id} [{mode}] albums=[{alb}] => {caption}", flush=True)

                if implied_tags:
                    immich_apply_tags(asset_id, implied_tags)

                if misfiled_identities:
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
                else:
                    if asset_type == "VIDEO":
                        # Multiple Creampie membership is purely manual (see
                        # is_multiple_creampie_album) -- the captioner never adds an asset to
                        # it and never removes one, only files newly-detected creampies into
                        # Single Creampie for anything not already placed by hand.
                        #
                        # Single Creampie is a 100%-human album, so non-human content never
                        # gets filed into it (caption_video already emits no count for those,
                        # but the guard is explicit so the two can't drift apart).
                        count_match = re.search(r"Separate Creampies\s*\|\s*(\d+)", caption, re.IGNORECASE)
                        if count_match and not multiple and not nonhuman:
                            n = int(count_match.group(1))
                            if n >= 1:
                                immich_add_to_album(asset_id, SINGLE_CREAMPIE_ALBUM_ID)
                                immich_archive(asset_id)

                    # Feral is the one category that belongs in no additional album at all:
                    # it's a real, non-anthropomorphic animal, the opposite of furry, and its
                    # own feral albums are the whole classification. Everything below is
                    # skipped for it -- anthro, by contrast, is expected to live in Furry Stuff.
                    if not feral:
                        if _FURRY_TRIGGER_RE.search(caption):
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
            else:
                print(f"[fail] {asset_id} update failed", flush=True)

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

        except Exception as e:
            print(f"[error] {asset_id}: {e}", flush=True)
            time.sleep(1.0)

    if USE_API_ONLY:
        while True:
            candidates = get_uncaptioned_candidates_api()
            if not candidates:
                print(f"[done] No more blank assets. Sleeping {IDLE_SLEEP_SECONDS}s and rechecking...", flush=True)
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            print(f"[batch] {len(candidates)} candidates", flush=True)
            for row in candidates:
                asset_id = row.get("id")
                asset_type = row.get("type", "UNKNOWN").upper()  # API may not have type; fallback
                albums = get_asset_albums(asset_id)  # Fetch separately
                process_candidate(
                    asset_id, asset_type, albums,
                    exif_make=(row.get("exifInfo") or {}).get("make"),
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
                    rows = pg_fetch_candidates(worker_conn, DB_REPRIORITIZE_BATCH)
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
                                            row.get("exif_make"))))
        finally:
            worker_conn.close()

    threading.Thread(target=_prefetch_worker, daemon=True, name="prefetch").start()
    print("[prefetch] background lookahead thread started", flush=True)

    while True:
        kind, payload = prefetch_q.get()

        if kind == "fetch_error":
            print(f"[error] prefetch DB fetch failed: {payload}", flush=True)
            time.sleep(1.0)
            continue

        if kind == "idle":
            print(f"[done] No more blank assets. Sleeping {IDLE_SLEEP_SECONDS}s and rechecking...", flush=True)
            continue

        asset_id, asset_type, albums, thumb, thumb_err, exif_make = payload
        try:
            process_candidate(asset_id, asset_type, albums, prefetched_thumbnail=thumb,
                              prefetched_thumbnail_error=thumb_err, exif_make=exif_make)
        finally:
            with in_flight_lock:
                in_flight_ids.discard(asset_id)

        print(f"[progress] total updated this run: {total_done}", flush=True)


if __name__ == "__main__":
    main()
