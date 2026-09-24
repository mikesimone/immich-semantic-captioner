#!/usr/bin/env python3
"""Runtime patch for immich_server: removes the "must have a timeline-visibility face"
restriction from the People feature, so people tagged only in archived albums still show up
in the People page, the face-tagging name-suggestion dropdown, and per-person asset counts.

Root cause (verified by reading the compiled source directly): GET /api/people is backed by
PersonRepository.getAllForUser() in dist/repositories/person.repository.js, which INNER JOINs
asset_face to asset with `.on('asset.visibility', '=', AssetVisibility.Timeline)` -- a person
whose only tagged faces are on archive-visibility assets produces zero rows and is silently
excluded, even though the Person row, the face, and the name are all completely intact and
fully functional otherwise (confirmed: GET /api/people/{id} and search-by-personId both work
fine regardless of visibility). The same restriction also appears in getStatistics() (a
person's "X photos" count) and getNumberOfPeople() (the total/hidden counts on the /api/people
response). This patches all three call sites in place, inside the already-running container's
compiled JS -- no image rebuild, no source fork, so it survives normal container restarts and
keeps working after immich-auto-upgrade.service pulls a new upstream image, AS LONG AS this
script is re-run after every immich_server start (see the companion watcher service/systemd
unit, immich-archive-people-patch-watcher.service, which does exactly that).

This is a text patch against compiled output, not a proper Immich fork -- deliberately, so it
doesn't require a Node/pnpm toolchain or a custom Docker build. If a future Immich release
changes this query's exact text, the patch will fail LOUDLY (a clear error, unmodified file)
rather than silently mis-patching or corrupting the file -- see FAIL below.

Run directly (one-shot, idempotent):

    python3 scripts/patch_immich_archive_people.py [--container immich_server] [--no-restart]

Exits 0 if already patched or newly patched successfully, 1 if the expected source patterns
weren't found (upstream code changed -- needs the patch updated) or on any other error.
"""
import argparse
import subprocess
import sys
import tempfile
import os

TARGET_PATH_IN_CONTAINER = "/usr/src/app/server/dist/repositories/person.repository.js"

MARKER = "// [archive-people-patch] visibility restriction removed from People queries\n"

# (description, old, new, expected_count) -- applied in order. `old` must appear EXACTLY
# expected_count times; any other number aborts the whole patch without writing anything
# (see main()).
#
# The exact count matters, and is not paranoia. Through v3.1.0 the first entry below matched
# two call sites at once -- getAllForUser() and getStatistics() shared a byte-identical join
# tail -- and the check was merely "at least one". v3.2.0's cluster-groups rework added an
# owner/shared-album predicate to getStatistics() only, so its tail stopped ending at
# `null))`, the shared pattern silently fell from 2 matches to 1, and the patch reported
# success having left getStatistics() fully restricted: every person's photo count silently
# dropped to timeline-visibility assets only. An exact count turns that class of upstream
# drift back into the loud failure this script promises in its docstring.
PATCHES = [
    (
        "getAllForUser() join tail",
        "            .on('asset.visibility', '=', kysely_1.sql.lit(enum_1.AssetVisibility.Timeline))\n"
        "            .on('asset.deletedAt', 'is', null))",
        "            .on('asset.deletedAt', 'is', null))",
        1,
    ),
    (
        # Split from the entry above in v3.2.0 -- see the note on drift. The trailing
        # `.on((eb) => eb.or(` is what distinguishes this site from getAllForUser()'s, so it
        # is matched (and re-emitted) verbatim rather than trimmed to the deletedAt line.
        "getStatistics() join tail",
        "            .on('asset.visibility', '=', kysely_1.sql.lit(enum_1.AssetVisibility.Timeline))\n"
        "            .on('asset.deletedAt', 'is', null)\n"
        "            .on((eb) => eb.or(",
        "            .on('asset.deletedAt', 'is', null)\n"
        "            .on((eb) => eb.or(",
        1,
    ),
    (
        "getNumberOfPeople() where-clause tail",
        "            .where('asset.visibility', '=', kysely_1.sql.lit(enum_1.AssetVisibility.Timeline))\n"
        "            .where('asset.deletedAt', 'is', null)))))",
        "            .where('asset.deletedAt', 'is', null)))))",
        1,
    ),
]


def log(msg: str) -> None:
    print(f"[patch] {msg}", flush=True)


def docker_cp_out(container: str, container_path: str, local_path: str) -> None:
    subprocess.run(["docker", "cp", f"{container}:{container_path}", local_path], check=True)


def docker_cp_in(local_path: str, container: str, container_path: str) -> None:
    subprocess.run(["docker", "cp", local_path, f"{container}:{container_path}"], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="immich_server")
    ap.add_argument("--no-restart", action="store_true", help="Patch the file but don't restart the container")
    ap.add_argument("--dry-run", action="store_true", help="Report what would happen without writing/restarting")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "person.repository.js")
        try:
            docker_cp_out(args.container, TARGET_PATH_IN_CONTAINER, local_path)
        except subprocess.CalledProcessError as e:
            log(f"FAIL: couldn't copy {TARGET_PATH_IN_CONTAINER} out of {args.container}: {e}")
            return 1

        with open(local_path, "r", encoding="utf-8") as f:
            content = f.read()

        if content.startswith(MARKER):
            log(f"already patched ({args.container} needs no change)")
            return 0

        new_content = content
        total_replacements = 0
        for desc, old, new, expected in PATCHES:
            count = new_content.count(old)
            if count != expected:
                log(
                    f"FAIL: pattern for {desc!r} matched {count} time(s), expected {expected} "
                    f"-- upstream code likely changed since this patch was written. Not touching "
                    f"the file. Needs a human to re-derive the patch against the current "
                    f"dist/repositories/person.repository.js."
                )
                return 1
            new_content = new_content.replace(old, new)
            total_replacements += count
            log(f"  {desc}: {count} occurrence(s) patched")

        new_content = MARKER + new_content

        if args.dry_run:
            log(f"[dry-run] would patch {total_replacements} occurrence(s) and restart={not args.no_restart}")
            return 0

        with open(local_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        try:
            docker_cp_in(local_path, args.container, TARGET_PATH_IN_CONTAINER)
        except subprocess.CalledProcessError as e:
            log(f"FAIL: couldn't copy patched file back into {args.container}: {e}")
            return 1

        log(f"patched {total_replacements} occurrence(s) in {args.container}")

        if args.no_restart:
            log("--no-restart set: file is patched on disk but the running Node process still has the old code loaded until restarted")
            return 0

        log(f"restarting {args.container} to load the patched file...")
        subprocess.run(["docker", "restart", args.container], check=True)
        log("restart issued")
        return 0


if __name__ == "__main__":
    sys.exit(main())
