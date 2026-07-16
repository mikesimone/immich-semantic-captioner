#!/usr/bin/env python3
import io
import os
import sys
import time
import json
import re
import subprocess
import tempfile
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

# Albums whose videos get dense, uniform-interval frame sampling across the whole
# clip instead of head-sparse/tail-dense -- for compilation-style videos where multiple
# distinct events (e.g. creampies) can occur anywhere, not just near the end, sometimes
# back to back with little gap between them.
DENSE_SAMPLING_ALBUM_KEYWORDS = os.environ.get("DENSE_SAMPLING_ALBUM_KEYWORDS", "creampie")
DENSE_INTERVAL_SECONDS = float(os.environ.get("DENSE_INTERVAL_SECONDS", "5"))
DENSE_MAX_VIDEO_FRAMES = int(os.environ.get("DENSE_MAX_VIDEO_FRAMES", "120"))

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

# Bare ASCII hyphens only count as a separator when surrounded by whitespace (" - "),
# so hyphenated compound words ("close-up", "dark-skinned") survive intact. Pipes/bullets/
# dashes always split, since those are never legitimately part of a normal word.
_SEP_REGEX = re.compile(r"\s*[\|\u2022•·–—]+\s*|\s+-\s+")
_JUNK_FULLCAPTION_REGEXES = [re.compile(r"^watch and share .* gifs on gfycat$", re.IGNORECASE)]
_TRAILING_HANDLE_RE = re.compile(r"\s*@[\w.]+\s*$", re.IGNORECASE)

# Backstop for when the model narrates a watermark/site-name/URL despite being told not
# to (e.g. "The watermark 'Princess69.com' is in the top right corner", "OnlyFans URL is
# visible at the bottom"). Split on real sentence boundaries (period-followed-by-space)
# rather than every period, since site domains like "OnlyFans.com" contain a period with
# no following space -- a naive per-period split would chop the sentence there and leave
# a dangling ".com/whatever" fragment behind.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_JUNK_SENTENCE_RE = re.compile(
    r"\b(?:watermarks?|logos?|onlyfans|url)\b"
    r"|\b[a-z0-9][a-z0-9-]*\.(?:com|net|org|co|xyz|vip|me|tv)\b",
    re.IGNORECASE,
)

def _strip_watermark_sentences(s: str) -> str:
    sentences = _SENTENCE_SPLIT_RE.split(s)
    kept = [sent for sent in sentences if not _JUNK_SENTENCE_RE.search(sent)]
    return " ".join(kept).strip()

# Backstop for the model opening with meta-commentary about the caption itself instead of
# describing the image (e.g. "A smutty, degrading caption for the image: ...").
_META_PREAMBLE_RE = re.compile(
    r"^(a|an)\s+[\w,\s]{0,40}\bcaption\b[\w\s]{0,20}\b(for|of)\s+(this|the)\s+"
    r"(image|video|photo|frame)[^:]{0,10}:\s*",
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
    "- State the apparent age range of any people (e.g. \"young adult\", \"college-age\", "
    "\"middle-aged\", \"milf\") when it's visually apparent.\n"
    "- Describe clothing specifically: what garment type it is and exactly what it covers vs. "
    "leaves bare. If an item is visually distinctive enough to identify the brand (e.g. red-"
    "soled heels are Christian Louboutin), name it.\n"
    "- Always describe the face and facial expression (eye contact, mouth open/closed, biting "
    "lip, eyes closed, surprised, blissed out, etc.).\n"
    "- Completely ignore watermarks, logos, and site names -- never mention or quote them. If "
    "there's other text baked into the image itself (a meme caption, speech bubble, etc.) that "
    "is NOT a watermark or site name, transcribe it.\n"
    "- If this is illustrated/animated art rather than a photo of a real person, and you "
    "recognize the character as a specific fictional or franchise character, name them.\n"
    "- Where it fits, use the same terms e621/Rule34 taggers use for acts, kinks, species, or "
    "fetish elements (e.g. \"paizuri\", \"gangbang\", \"bukkake\", \"futanari\") instead of "
    "vaguer plain-English phrasing.\n"
    "- Never use vague or ambiguous language -- say exactly what's happening.\n"
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
        "fluids at all in that case, even in passing or as a comparison.\n"
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
        "(fingers, a toy/object, a tongue, a cock). Mention any visible cum/jizz.\n"
        "Only describe what is actually visible in this specific frame/image -- never invent "
        "sexual content, fluids, or acts that aren't really there."
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
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()

    def caption_detailed(
        pil_image: Image.Image,
        video_note: str = "",
        person_names: Optional[List[str]] = None,
        prompt_override: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        prompt = prompt_override if prompt_override is not None else build_caption_prompt(video_note, person_names)
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]
        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[convo_string], images=[pil_image], return_tensors="pt").to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=JOYCAPTION_TEMPERATURE,
                top_p=0.9,
                repetition_penalty=JOYCAPTION_REPETITION_PENALTY,
            )[0]
        generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
        txt = processor.tokenizer.decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return " ".join(txt.strip().split())[:MAX_CAPTION_CHARS]

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

_CREAMPIE_RE = re.compile(r"creampie", re.IGNORECASE)

def count_creampie_events(frame_captions: List[Tuple[float, str]]) -> Tuple[int, List[str]]:
    # frame_captions is already in chronological sample order. Each frame is captioned
    # independently, so a "creampie" mention marks the pull-out-with-fresh-cum moment for
    # that single frame -- there's no cross-frame memory for the model to count events
    # itself. Count a new event on every transition from "not currently a creampie moment"
    # to "creampie moment": a run of consecutive flagged frames is one event, and as long
    # as at least one frame in between shows a fresh insertion (not flagged) rather than
    # lingering residual cum, back-to-back events stay separated instead of merging.
    event_starts = []
    was_flagged = False
    for ts, cap in frame_captions:
        is_flagged = bool(_CREAMPIE_RE.search(cap))
        if is_flagged and not was_flagged:
            event_starts.append(format_ts(ts))
        was_flagged = is_flagged
    return len(event_starts), event_starts

# For dense (creampie-count) videos we don't want a full narrative per frame -- just a
# cheap per-frame signal for counting, plus one detailed description of the woman from a
# single representative frame. This short classification prompt drives that per-frame pass.
_DENSE_SIGNAL_PROMPT = (
    "Look at this single video frame and answer with ONLY ONE short line, no extra words.\n"
    "First, if this frame is just a text card, logo, watermark screen, loading screen, or "
    "otherwise doesn't show a person at all, say \"TITLECARD\" and stop there.\n"
    "Otherwise say exactly one of:\n"
    "\"CREAMPIE\" -- a cock/toy has just been pulled out and fresh cum is visibly dumping "
    "out right now.\n"
    "\"INSERTED\" -- a cock/toy is currently inside her (ongoing penetration, no pull-out "
    "happening right now).\n"
    "\"CUM-VISIBLE\" -- cum is visible but nothing is currently being inserted or pulled "
    "out (a lull between events).\n"
    "\"NONE\" -- none of the above apply to this frame.\n"
    "Then, on the same line, add \" BONDAGE\" if rope, cuffs, a collar/leash, or other "
    "restraints are visible on her in this frame, otherwise add \" NO-BONDAGE\"."
)

def _parse_dense_signal(text: str) -> Tuple[bool, bool, bool]:
    t = text.upper()
    is_titlecard = "TITLECARD" in t
    is_creampie = "CREAMPIE" in t
    is_bondage = "BONDAGE" in t and "NO-BONDAGE" not in t and "NO BONDAGE" not in t
    return is_titlecard, is_creampie, is_bondage

def _caption_video_dense(
    frames: List[Tuple[float, Image.Image]],
    caption_detailed,
    person_names: Optional[List[str]],
) -> Tuple[str, str]:
    creampie_flags: List[Tuple[float, str]] = []
    bondage_any = False
    person_frames: List[Tuple[float, Image.Image]] = []

    for ts, img in frames:
        signal = caption_detailed(img, prompt_override=_DENSE_SIGNAL_PROMPT, max_new_tokens=12)
        is_titlecard, is_creampie, is_bondage = _parse_dense_signal(signal)
        creampie_flags.append((ts, "CREAMPIE" if is_creampie else ""))
        bondage_any = bondage_any or is_bondage
        if not is_titlecard:
            person_frames.append((ts, img))

    count, event_times = count_creampie_events(creampie_flags)

    # Pick a representative frame for the one detailed description -- the middle of
    # whichever frames actually show a person, skipping title cards/intro screens, since
    # a fixed "just take the middle frame" would sometimes land on an intro card instead.
    candidates = person_frames or frames
    _, desc_img = candidates[len(candidates) // 2]
    woman_desc = caption_detailed(desc_img, video_note="This is one frame from a video.", person_names=person_names)

    plural = "creampie" if count == 1 else "creampies"
    summary = f"SUMMARY: {count} separate {plural} visible (~{', '.join(event_times)})" if count else "SUMMARY: 0 creampies detected"

    parts = [summary, woman_desc]
    if bondage_any:
        parts.append("Bondage/restraints are visible at some point in this video.")

    return " || ".join(parts), "VIDEO-FRAMES-DENSE"

def caption_video(
    asset_id: str,
    caption_detailed,
    person_names: Optional[List[str]] = None,
    dense: bool = False,
) -> Tuple[str, str]:
    fd, video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        immich_download_original(asset_id, video_path)
        frames = extract_video_frames(video_path, dense=dense)
        if not frames:
            raise RuntimeError("no frames extracted")

        if dense:
            return _caption_video_dense(frames, caption_detailed, person_names)

        parts = []
        for ts, img in frames:
            cap = caption_detailed(img, video_note="This is one frame from a video.", person_names=person_names)
            parts.append(f"[{format_ts(ts)}] {cap}")
        return " || ".join(parts), "VIDEO-FRAMES"
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

    # Same priority as the DB path: images before videos, newest first within each type.
    # Two stable sorts: createdAt descending first, then type-priority ascending -- the
    # second sort preserves the createdAt ordering within each type group.
    candidates.sort(key=lambda item: item.get("createdAt", "") or "", reverse=True)
    candidates.sort(key=lambda item: 0 if str(item.get("type", "")).upper() == "IMAGE" else 1)

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

                person_names = extract_identities_from_albums(albums)

                if asset_type == "VIDEO":
                    dense = is_dense_sampling_album(albums)
                    raw_caption, mode = caption_video(
                        asset_id, caption_detailed, person_names=person_names, dense=dense
                    )
                else:
                    img = immich_get_thumbnail(asset_id)
                    raw_caption, mode = caption_image(img, person_names=person_names)

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

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                if not USE_API_ONLY:
                    pg_mark_skip(conn, asset_id, f"HTTP_ERROR_{status}")
                print(f"[skip] {asset_id}: HTTP {status} fetching asset (marked skip)", flush=True)
                time.sleep(0.2)

            except Exception as e:
                print(f"[error] {asset_id}: {e}", flush=True)
                time.sleep(1.0)

        print(f"[progress] total updated this run: {total_done}", flush=True)


if __name__ == "__main__":
    main()
