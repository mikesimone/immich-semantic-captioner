#!/usr/bin/env python3
"""Runtime patch for immich_server's BUILT FRONTEND: removes the hardcoded
"visibility: Timeline" filter the person-detail page (/people/{id}) applies to its own asset
query, so a person's page actually shows every photo they're tagged in -- archived included --
instead of silently dropping anyone whose tagged photos are all archived. Without this, the
backend fix in patch_immich_archive_people.py only gets you halfway: the person now shows up
in the People list, but clicking into them can still show zero photos.

Root cause (found by grepping the built SvelteKit output, since this is compiled/minified
frontend, not the NestJS backend patched by patch_immich_archive_people.py): the person page's
route module (.../people/[personId]/.../+page.svelte, compiled into a hashed file under
/build/www/_app/immutable/nodes/) computes its asset-search options as roughly

    { visibility: AssetVisibility.Timeline, personId: <this person's id> }

and sends that straight to GET /api/timeline/buckets. That endpoint's `visibility` param is
optional and, when omitted, returns assets of every visibility (verified empirically: the same
request with the visibility param entirely removed returns archived assets that vanish the
moment "timeline" is sent) -- so the fix is to stop sending it, not to find some other value
that means "everything".

CACHING: /_app/immutable/ is served with `Cache-Control: public,max-age=31536000,immutable` --
a year, and browsers skip revalidation entirely for `immutable` resources even on a normal
reload. Editing a chunk's *content* while keeping its *filename* (the first version of this
script did exactly that) fixes the server but leaves every client that already loaded that
exact URL stuck on the stale cached copy indefinitely, with no ordinary reload able to fix it.
The correct approach -- and what this version does -- is to actually rename the patched file
(a new URL is a guaranteed cache miss for everyone) and cascade that rename upward through
whatever references it, one hop at a time, until reaching a file that ISN'T immutably cached.
In practice that's exactly two hops: the node chunk is imported by one SvelteKit "entry" chunk,
which is referenced only by index.html -- and index.html is served `Cache-Control: no-store`,
so once its content points at the new (unpatched-cache) entry filename, EVERY client picks up
the whole fixed chain on their very next normal page load. No hard refresh, no manual cache
clearing, no user action at all.

This also regenerates precompressed .br/.gz siblings at each renamed file (there's no `brotli`
CLI in the image, so this uses Node's built-in zlib, which is guaranteed present) -- the static
file server prefers these over the plain file when present, so skipping this step would leave
old content being served even from a brand-new URL.

The node/entry chunk renames take effect immediately (plain static files, read from disk per
request). index.html does NOT -- Immich serves it from an in-memory copy read once at startup
(the usual SPA-fallback pattern), so the on-disk edit needs one container restart to actually
be served, which this script issues itself. Verify this assumption still holds against any
future Immich version before trusting the "no restart" framing anywhere in this file.

Idempotency: rather than a marker, this checks whether the *currently referenced* chunk (the
one index.html's content actually points at right now) already has the fix applied, and does
nothing if so. Fails loudly (clear error, nothing written) if the reference chain doesn't look
like the 2-hop shape above, or if more than one file references a given filename (ambiguous --
needs a human to check what changed upstream in Immich).

Run directly (one-shot, idempotent):

    python3 scripts/patch_immich_archive_people_frontend.py [--container immich_server]

Exits 0 if patched (or nothing needed patching), 1 on any real failure.
"""
import argparse
import re
import subprocess
import sys
import tempfile
import os
import time

WWW_ROOT = "/build/www"
NODES_DIR = f"{WWW_ROOT}/_app/immutable/nodes"
INDEX_HTML = f"{WWW_ROOT}/index.html"
MAX_HOPS = 5

# Matches the compiled `{visibility: <enum>.Timeline, personId: ...}` object literal
# regardless of the minified identifier used for the AssetVisibility enum (verified against a
# real build as `ae.Timeline`, but that name is not stable across builds). Deliberately avoids
# \w -- the container's grep doesn't support it (busybox), so the Python and shell versions of
# this pattern must both stick to POSIX-safe character classes.
TARGET_RE = re.compile(r"visibility:[A-Za-z_$][A-Za-z0-9_$]*\.Timeline,personId:")
REPLACEMENT = "personId:"


def log(msg: str) -> None:
    print(f"[patch-fe] {msg}", flush=True)


def sh(cmd: list, **kwargs):
    return subprocess.run(cmd, check=True, **kwargs)


def read_file(container: str, container_path: str, tmpdir: str) -> str:
    local_path = os.path.join(tmpdir, os.path.basename(container_path) + f".{time.time_ns()}")
    sh(["docker", "cp", f"{container}:{container_path}", local_path])
    with open(local_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(container: str, container_path: str, content: str, tmpdir: str) -> None:
    local_path = os.path.join(tmpdir, os.path.basename(container_path) + f".{time.time_ns()}")
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(content)
    sh(["docker", "cp", local_path, f"{container}:{container_path}"])


def recompress(container: str, container_path: str) -> None:
    script = (
        "const fs=require('fs'),zlib=require('zlib');"
        f"const p={container_path!r};"
        "const buf=fs.readFileSync(p);"
        "fs.writeFileSync(p+'.br', zlib.brotliCompressSync(buf));"
        "fs.writeFileSync(p+'.gz', zlib.gzipSync(buf));"
    )
    sh(["docker", "exec", container, "node", "-e", script])


def find_matching_node_chunks(container: str) -> list:
    result = subprocess.run(
        ["docker", "exec", container, "sh", "-c",
         f"grep -lE '{TARGET_RE.pattern}' {NODES_DIR}/*.js 2>/dev/null || true"],
        check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def find_referencing_files(container: str, basename: str) -> list:
    result = subprocess.run(
        ["docker", "exec", container, "sh", "-c",
         f"grep -rl {basename!r} {WWW_ROOT} --include='*.js' --include='*.html' 2>/dev/null || true"],
        check=True, capture_output=True, text=True,
    )
    # Exclude the file itself if grep matched its own filename appearing in its own content.
    return [line for line in result.stdout.splitlines() if line.strip() and os.path.basename(line) != basename]


def new_path_for(container_path: str, suffix: str) -> str:
    root, ext = os.path.splitext(container_path)
    return f"{root}-{suffix}{ext}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="immich_server")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    container = args.container
    suffix = f"archpatch{int(time.time())}"

    try:
        matches = find_matching_node_chunks(container)
    except subprocess.CalledProcessError as e:
        log(f"FAIL: couldn't search {NODES_DIR} in {container}: {e}")
        return 1

    if not matches:
        log("no node chunk currently has the unpatched filter -- already patched, or this Immich build changed the code (nothing to do either way)")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for node_path in matches:
            content = read_file(container, node_path, tmpdir)
            new_content, count = TARGET_RE.subn(REPLACEMENT, content)
            if count == 0:
                log(f"  {node_path}: matched by grep but not by the precise regex -- skipping, needs a human look")
                continue

            log(f"  {node_path}: {count} occurrence(s) to patch")
            if args.dry_run:
                continue

            current_old_path = node_path
            current_new_path = new_path_for(node_path, suffix)
            current_new_content = new_content
            old_paths_to_remove = []

            for hop in range(MAX_HOPS):
                write_file(container, current_new_path, current_new_content, tmpdir)
                recompress(container, current_new_path)
                log(f"  wrote {current_new_path} (hop {hop})")

                old_paths_to_remove.append(current_old_path)
                old_basename = os.path.basename(current_old_path)
                new_basename = os.path.basename(current_new_path)

                refs = find_referencing_files(container, old_basename)
                if not refs:
                    log(f"FAIL: nothing references {old_basename!r} -- can't propagate the rename up to an uncached entry point (expected to eventually reach index.html). Aborting without touching index.html or removing any old file.")
                    return 1
                if INDEX_HTML in refs and len(refs) == 1:
                    index_content = read_file(container, INDEX_HTML, tmpdir)
                    new_index_content = index_content.replace(old_basename, new_basename)
                    if new_index_content == index_content:
                        log(f"FAIL: index.html matched by grep for {old_basename!r} but string replace made no change -- unexpected, aborting.")
                        return 1
                    write_file(container, INDEX_HTML, new_index_content, tmpdir)
                    recompress(container, INDEX_HTML)
                    log(f"  updated {INDEX_HTML} to reference {new_basename}")

                    # Remove the now-orphaned pre-rename files -- otherwise they'd sit there
                    # still matching TARGET_RE (the node chunk) forever, and the NEXT run would
                    # find them, try to redo this whole cascade from a stale starting point
                    # (the entry chunk they'd try to update no longer matches what index.html
                    # actually references), and fail loudly on every single future restart.
                    for old_path in old_paths_to_remove:
                        sh(["docker", "exec", container, "sh", "-c",
                            f"rm -f {old_path} {old_path}.br {old_path}.gz"])
                    log(f"  removed {len(old_paths_to_remove)} orphaned pre-rename file(s)")

                    log(f"  restarting {container} -- index.html is served from an in-memory copy read at startup, so the on-disk edit alone doesn't take effect until the process restarts (unlike the plain-static /_app/immutable/ files, which are read fresh from disk per request and needed no restart)")
                    if not args.dry_run:
                        sh(["docker", "restart", container])
                    break
                if len(refs) != 1:
                    log(f"FAIL: {old_basename!r} is referenced by {len(refs)} files ({refs}) -- expected exactly one (the parent chunk) or index.html alone. This Immich build's structure differs from what this script assumes; needs a human to check.")
                    return 1

                parent_path = refs[0]
                parent_content = read_file(container, parent_path, tmpdir)
                new_parent_content, parent_count = (
                    parent_content.replace(old_basename, new_basename),
                    parent_content.count(old_basename),
                )
                if parent_count == 0:
                    log(f"FAIL: {parent_path} referenced {old_basename!r} per grep but count() found zero -- aborting.")
                    return 1
                log(f"  {parent_path}: updating {parent_count} reference(s) to {old_basename} -> {new_basename}")

                current_old_path = parent_path
                current_new_path = new_path_for(parent_path, suffix)
                current_new_content = new_parent_content
            else:
                log(f"FAIL: didn't reach index.html within {MAX_HOPS} hops -- aborting rather than guessing further.")
                return 1

    if args.dry_run:
        log(f"[dry-run] would patch {len(matches)} node chunk(s) and cascade the rename up to index.html")
    else:
        log("patch complete; every client gets the fix on their next normal page load (index.html is never cached, so no hard refresh needed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
