# Source me: `. scripts/vault-env.sh; vault_env NAME [NAME...]` exports each NAME from the
# fleet's Infisical vault (via Environment/bin/fleet-sync get) unless it is already set in
# the environment. Replaces sourcing ~/.api-keys. Values are never printed; a name the vault
# can't supply is reported by name only and makes vault_env return 1.
vault_env() {
  local fs n v rc=0
  for fs in "$(command -v fleet-sync 2>/dev/null)" "$HOME/Environment/bin/fleet-sync" /home/msimone/Environment/bin/fleet-sync; do
    [ -n "$fs" ] && [ -x "$fs" ] && break
  done
  for n in "$@"; do
    [ -n "${!n:-}" ] && continue
    if v=$("$fs" get "$n" 2>/dev/null) && [ -n "$v" ]; then
      export "$n=$v"
    else
      echo "vault-env: $n not available from the vault (is fleet-sync/Infisical reachable?)" >&2
      rc=1
    fi
  done
  return $rc
}
