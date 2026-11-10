#!/usr/bin/env python3
"""Runtime patch for immich_server: search treats archived assets like timeline assets.

Why: the mobile app sends every search with an explicit visibility -- `timeline` normally, and
`archive` (archive ONLY) when "Search archives" is ticked (mobile/lib/infrastructure/repositories/
search_api.repository.dart, both smart and metadata search). So anything archived is invisible to
a normal phone search, and ticking the box swaps to an archive-only result set, which is why a
blank archive search dumps every archived picture. For a library where "archive" just means "keep
it off my timeline", that makes search useless.

The patch changes ONE thing in the shared legacy search builder (searchAssetBuilderLegacy in
dist/utils/database.js, used by /search/smart, /search/metadata, random, large-assets and search
statistics): a request for visibility = timeline matches timeline OR archive. Explicit `archive`,
`locked`, `hidden` and `not-locked` behave exactly as upstream, and the timeline itself (a different
query) is untouched, so archive still keeps things off the timeline on web and phone.

Same rules as the sibling patches: exact text replacement against compiled output, every pattern
must match its exact count or nothing is written (loud FAIL), a marker line makes it idempotent,
and reapply-archive-people-patch-on-restart.sh re-runs it after every immich_server start.

    python3 scripts/patch_immich_archive_search.py [--container immich_server] [--no-restart] [--dry-run]

Exits 0 if already patched or newly patched, 1 if upstream code changed or on error.
"""
import argparse
import os
import subprocess
import tempfile

TARGET = "/usr/src/app/server/dist/utils/database.js"
MARKER = "// [archive-search-patch] search visibility=timeline also matches archive\n"

# (description, old, new, expected_count)
PATCHES_V34 = [
    (
        "searchAssetBuilderLegacy visibility filter",
        "        : qb.where('asset.visibility', '=', options.visibility))\n",
        "        : options.visibility === AssetVisibility.Timeline\n"
        "            ? qb.where('asset.visibility', 'in', [AssetVisibility.Timeline, AssetVisibility.Archive])\n"
        "            : qb.where('asset.visibility', '=', options.visibility))\n",
        1,
    ),
]
VARIANTS = [("3.4.x", PATCHES_V34)]


def log(msg: str) -> None:
    print(f"[patch-search] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="immich_server")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "database.js")
        try:
            subprocess.run(["docker", "cp", f"{a.container}:{TARGET}", local], check=True)
        except subprocess.CalledProcessError as e:
            log(f"FAIL: couldn't copy {TARGET} out of {a.container}: {e}")
            return 1
        content = open(local, encoding="utf-8").read()
        if content.startswith(MARKER):
            log(f"already patched ({a.container} needs no change)")
            return 0
        chosen = None
        for version, patches in VARIANTS:
            counts = [content.count(old) for _, old, _, _ in patches]
            if all(c == exp for c, (*_, exp) in zip(counts, patches)):
                chosen = (version, patches)
                break
            log(f"  {version} patterns don't fit: " + ", ".join(
                f"{d!r} matched {c}, expected {e}" for c, (d, _, _, e) in zip(counts, patches) if c != e))
        if not chosen:
            log("FAIL: no known patch variant matches -- upstream code likely changed. Not touching the file. "
                "Needs a human to re-derive the patch against dist/utils/database.js.")
            return 1
        version, patches = chosen
        log(f"using {version} patterns")
        for desc, old, new, exp in patches:
            content = content.replace(old, new)
            log(f"  {desc}: {exp} occurrence(s) patched")
        if a.dry_run:
            log(f"[dry-run] would write the patch and restart={not a.no_restart}")
            return 0
        with open(local, "w", encoding="utf-8") as f:
            f.write(MARKER + content)
        try:
            subprocess.run(["docker", "cp", local, f"{a.container}:{TARGET}"], check=True)
        except subprocess.CalledProcessError as e:
            log(f"FAIL: couldn't copy the patched file back into {a.container}: {e}")
            return 1
        log(f"patched {a.container}")
        if a.no_restart:
            log("--no-restart: the running process keeps the old code until restarted")
        else:
            subprocess.run(["docker", "restart", a.container], check=True, stdout=subprocess.DEVNULL)
            log(f"restarted {a.container}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
