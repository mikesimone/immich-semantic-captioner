#!/usr/bin/env python3
"""Runtime patch for immich_server: lets a timeline query that asks for BOTH your archived
assets and your partners' assets actually work, so a person's page (/people/{id}) shows every
photo of them -- your archived ones included -- plus whatever your partners share.

Why it's needed (Immich 3.2.x, verified against the compiled source): the person page asks
GET /api/timeline/buckets with {personId, withPartners: true} and no `visibility`. No
visibility means "timeline + archive" (database.withDefaultVisibility), but
TimelineService.timeBucketChecks() rejects withPartners with 400 whenever visibility is Archive
OR undefined -- because the asset query applies one visibility filter to everyone, so allowing
it would expose partners' archived assets. Result: archived-only people show a count over an
empty grid.

The patch does the two halves together (never one without the other):

  1. timeline.service.js: withPartners is rejected only for an explicit visibility=archive
     (plus the existing locked / favorite / trash rules); undefined is allowed.
  2. asset.repository.js, getTimeBuckets() and getTimeBucket(): when withPartners is set and
     visibility is undefined, the owner filter becomes
         owner's assets (timeline + archive)  OR  partners' assets with visibility = timeline
     so a partner's archive is never shown, exactly as before. Every other query is unchanged.

Companion to patch_immich_archive_people.py (People list/counts) and
patch_immich_archive_people_frontend.py (person page drops its Timeline-only filter); all
three are reapplied after every immich_server start by
reapply-archive-people-patch-on-restart.sh.

Same rules as the other patch: text replacement against compiled output, each pattern must
match EXACTLY the expected number of times or nothing is written; a marker line makes it
idempotent.

    python3 scripts/patch_immich_archive_partners.py [--container immich_server] [--no-restart] [--dry-run]

Exits 0 if already patched or newly patched, 1 if upstream code changed or on error.
"""
import argparse
import os
import subprocess
import sys
import tempfile

MARKER = "// [archive-partners-patch] owner archive + partner timeline allowed together\n"
DIST = "/usr/src/app/server/dist"

def owner_patch(any_uuid: str, timeline_lit: str) -> tuple[str, str]:
    old = f"            const isOwner = eb('asset.ownerId', '=', {any_uuid}(options.userIds));\n"
    new = (
        "            const isOwner = options.withPartners && options.visibility === undefined && options.userIds.length > 1\n"
        "                ? eb.or([\n"
        "                    eb('asset.ownerId', '=', options.userIds[0]),\n"
        "                    eb.and([\n"
        f"                        eb('asset.ownerId', '=', {any_uuid}(options.userIds.slice(1))),\n"
        f"                        eb('asset.visibility', '=', {timeline_lit}),\n"
        "                    ]),\n"
        "                ])\n"
        f"                : eb('asset.ownerId', '=', {any_uuid}(options.userIds));\n"
    )
    return old, new


# Each pattern lists one (old, new) alternative per Immich build style; the first alternative
# that matches exactly `expected_count` times is used. v3.3.0 compiles to ES modules, so the
# CommonJS prefixes of 3.2.x (kysely_1., enum_1., database_1.) are gone; the logic is
# unchanged. (3.3.0 also ORs shared-album assets around isOwner when personId is set; that
# wraps isOwner from outside, so replacing isOwner still never admits a partner's archive.)
# Both files come from the same image, so they always resolve to the same build style.
V33 = (
    "const isRequestedArchived = dto.visibility === AssetVisibility.Archive || dto.visibility === undefined;",
    "const isRequestedArchived = dto.visibility === AssetVisibility.Archive;",
)
V32 = (
    "const isRequestedArchived = dto.visibility === enum_1.AssetVisibility.Archive || dto.visibility === undefined;",
    "const isRequestedArchived = dto.visibility === enum_1.AssetVisibility.Archive;",
)

# file -> [(description, [(old, new), ...], expected_count)]
FILES = {
    f"{DIST}/services/timeline.service.js": [
        ("withPartners check allows undefined visibility", [V33, V32], 1),
    ],
    f"{DIST}/repositories/asset.repository.js": [
        (
            "getTimeBuckets()/getTimeBucket() owner filter",
            [
                owner_patch("anyUuid", "sql.lit(AssetVisibility.Timeline)"),
                owner_patch("(0, database_1.anyUuid)", "kysely_1.sql.lit(enum_1.AssetVisibility.Timeline)"),
            ],
            2,
        ),
    ],
}


def log(msg: str) -> None:
    print(f"[patch-partners] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="immich_server")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    staged = []  # (local_path, container_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (path, patches) in enumerate(FILES.items()):
            local = os.path.join(tmpdir, f"{i}.js")
            try:
                subprocess.run(["docker", "cp", f"{args.container}:{path}", local], check=True)
            except subprocess.CalledProcessError as e:
                log(f"FAIL: couldn't copy {path} out of {args.container}: {e}")
                return 1
            content = open(local, encoding="utf-8").read()
            if content.startswith(MARKER):
                log(f"{os.path.basename(path)}: already patched")
                continue
            for desc, alternatives, expected in patches:
                counts = [content.count(old) for old, _ in alternatives]
                match = next((alt for alt, n in zip(alternatives, counts) if n == expected), None)
                if match is None:
                    log(f"FAIL: {desc!r} matched {counts} time(s) in {os.path.basename(path)} (one per known "
                        f"build style), expected {expected} -- upstream code changed. Nothing written.")
                    return 1
                content = content.replace(*match)
                log(f"  {os.path.basename(path)}: {desc}: {expected} occurrence(s)")
            open(local, "w", encoding="utf-8").write(MARKER + content)
            staged.append((local, path))

        if not staged:
            log("already patched (immich_server needs no change)")
            return 0
        if args.dry_run:
            log(f"[dry-run] would patch {len(staged)} file(s)")
            return 0
        # Every file was checked before any is written, so a pattern miss never leaves a
        # half-applied patch (service relaxed but repository not = partner archive leak).
        for local, path in staged:
            subprocess.run(["docker", "cp", local, f"{args.container}:{path}"], check=True)
        log(f"patched {len(staged)} file(s)")

    if args.no_restart:
        log("--no-restart: the running process keeps the old code until restarted")
        return 0
    log(f"restarting {args.container}...")
    subprocess.run(["docker", "restart", args.container], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
