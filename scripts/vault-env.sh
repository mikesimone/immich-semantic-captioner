# Source me: `. scripts/vault-env.sh; vault_env NAME [NAME...]` exports each NAME from the
# fleet's Infisical vault (via Environment/bin/vault-get: the Infisical CLI) unless it is already set in
# the environment. Replaces sourcing ~/.api-keys. Values are never printed; a name the vault
# can't supply is reported by name only and makes vault_env return 1.
vault_env() {
  local fs n v rc=0
  for fs in "$HOME/Environment/bin/vault-get" /home/msimone/Environment/bin/vault-get; do
    [ -n "$fs" ] && [ -x "$fs" ] && break
  done
  for n in "$@"; do
    [ -n "${!n:-}" ] && continue
    if v=$("$fs" "$n" 2>/dev/null) && [ -n "$v" ]; then
      export "$n=$v"
    else
      echo "vault-env: $n not available from the vault (is the Infisical vault reachable?)" >&2
      rc=1
    fi
  done
  return $rc
}
