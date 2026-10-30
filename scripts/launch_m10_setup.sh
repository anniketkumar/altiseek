#!/usr/bin/env bash
# Detached system installation with a lock and persistent logs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/m10/setup
exec 9>/tmp/altiseek-m10-system.lock
flock -n 9 || { echo 'M10 system installation already running'; exit 1; }
echo $$ > data/m10/setup/launcher.pid
echo running > data/m10/setup/state
if bash scripts/setup_m10_system.sh > data/m10/setup/system.log 2>&1; then
  echo system_ready > data/m10/setup/state
else
  echo failed > data/m10/setup/state
  exit 1
fi
