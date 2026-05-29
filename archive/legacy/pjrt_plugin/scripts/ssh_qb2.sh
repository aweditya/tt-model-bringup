#!/usr/bin/env bash
# Wrapper around ssh qb2 that retries on transient connection failures
# (qb2 sshd intermittently returns "Connection refused" — often roughly
# 3-in-4 attempts fail). Usage: pjrt_plugin/scripts/ssh_qb2.sh '<remote command>'
#
# Exits non-zero only after MAX_ATTEMPTS consecutive failures.

set -u

REMOTE_CMD="${1:-echo ok}"
MAX_ATTEMPTS="${SSH_MAX_ATTEMPTS:-40}"
SLEEP_SEC="${SSH_RETRY_SLEEP:-4}"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  if ssh -o ConnectTimeout=8 -o BatchMode=yes -o StrictHostKeyChecking=no \
         qb2 "$REMOTE_CMD"; then
    exit 0
  fi
  rc=$?
  # rc=255 typically = ssh transport failure (e.g. connection refused).
  # Any other rc = remote command actually ran but exited non-zero -> bubble up.
  if [ "$rc" -ne 255 ]; then
    exit "$rc"
  fi
  attempt=$((attempt + 1))
  sleep "$SLEEP_SEC"
done

echo "ssh_qb2: gave up after $MAX_ATTEMPTS attempts" >&2
exit 255
