#!/usr/bin/env bash
set -euo pipefail

IMG="${1:-immich-semantic-captioner:known-good}"
KNOWN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/KNOWN_GOOD.env"

exp_img="$(grep '^IMAGE_ID=' "$KNOWN" | cut -d= -f2-)"
exp_pip="$(grep '^PIP_FREEZE_SHA256=' "$KNOWN" | cut -d= -f2-)"

act_img="$(docker image inspect "$IMG" --format '{{.Id}}')"
act_pip="$(docker run --rm "$IMG" /opt/venv/bin/pip freeze | awk '/^[A-Za-z0-9_.-]+==[0-9]/{print}' | sort | sha256sum | awk '{print $1}')"

echo "Expected IMAGE_ID:       $exp_img"
echo "Actual   IMAGE_ID:       $act_img"
echo
echo "Expected PIP_FREEZE_SHA: $exp_pip"
echo "Actual   PIP_FREEZE_SHA: $act_pip"
echo

if [[ "$act_pip" != "$exp_pip" ]]; then
  echo "FAIL: pip dependency fingerprint does not match known-good."
  exit 1
fi

if [[ "$act_img" != "$exp_img" ]]; then
  echo "WARN: image ID differs, but dependencies match (this is usually fine)."
else
  echo "OK: image ID matches known-good."
fi

echo "OK: dependency fingerprint matches known-good."
