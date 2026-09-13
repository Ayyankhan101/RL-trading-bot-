#!/usr/bin/env bash
#
# Wrapper for the gold paper-trading bot under launchd.
#
# launchd starts a job with almost no environment: no PATH, no working
# directory, no shell profile. Everything the bot needs is supplied here
# explicitly rather than inherited.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${GOLDBOT_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/bin/python3}"
POLL="${GOLDBOT_POLL:-900}"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/live_gold.log"
MAX_LOG_BYTES=$((10 * 1024 * 1024))

mkdir -p "$LOG_DIR"

# Rotate before appending. A month of 15-minute polling fills a disk quietly,
# and a bot that dies because the volume is full is a bot that silently stops
# producing the record it exists to produce.
if [[ -f "$LOG" ]]; then
  size=$(stat -f%z "$LOG" 2>/dev/null || stat -c%s "$LOG" 2>/dev/null || echo 0)
  if (( size > MAX_LOG_BYTES )); then
    mv -f "$LOG" "$LOG.1"
  fi
fi

cd "$REPO" || exit 1

if [[ ! -x "$PYTHON" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] python not found at $PYTHON" >> "$LOG"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting live_gold.py --poll $POLL" >> "$LOG"

# -u so the log reflects events as they happen rather than in 4KB blocks;
# watching a buffered log is indistinguishable from watching a hung process.
exec "$PYTHON" -u live_gold.py --poll "$POLL" >> "$LOG" 2>&1
