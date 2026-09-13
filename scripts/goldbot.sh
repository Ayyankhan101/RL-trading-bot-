#!/usr/bin/env bash
#
# Control the gold paper-trading bot as a launchd job.
#
#   ./scripts/goldbot.sh install     copy the LaunchAgent and load it
#   ./scripts/goldbot.sh start|stop|restart
#   ./scripts/goldbot.sh status      launchd state AND the account itself
#   ./scripts/goldbot.sh logs        follow the log
#   ./scripts/goldbot.sh uninstall   unload and remove
#
# Nothing is installed outside the repo except by `install`, and `uninstall`
# fully reverses it.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.goldbot.live"
AGENT_DIR="$HOME/Library/LaunchAgents"
AGENT="$AGENT_DIR/$LABEL.plist"
TEMPLATE="$REPO/scripts/com.goldbot.plist"
PYTHON="${GOLDBOT_PYTHON:-/opt/homebrew/Caskroom/miniforge/base/bin/python3}"
LOG="$REPO/logs/live_gold.log"

usage() { sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

case "${1:-}" in
  install)
    mkdir -p "$AGENT_DIR" "$REPO/logs"
    # The plist cannot use a relative path or ~, so the repo location is
    # substituted at install time rather than hardcoded in the committed file.
    sed "s|__REPO__|$REPO|g" "$TEMPLATE" > "$AGENT"
    launchctl unload "$AGENT" 2>/dev/null
    if launchctl load "$AGENT"; then
      echo "installed and loaded: $AGENT"
      echo "it will now start on login and restart if it crashes"
      echo "check it with: ./scripts/goldbot.sh status"
    else
      echo "load failed; check $REPO/logs/launchd.err.log" >&2
      exit 1
    fi
    ;;

  uninstall)
    launchctl unload "$AGENT" 2>/dev/null
    rm -f "$AGENT"
    echo "unloaded and removed $AGENT"
    echo "the paper account in results/live_gold/ is left untouched"
    ;;

  start)   launchctl start "$LABEL" && echo "started $LABEL" ;;
  stop)    launchctl stop  "$LABEL" && echo "stopped $LABEL" ;;
  restart) launchctl stop  "$LABEL" 2>/dev/null; sleep 2
           launchctl start "$LABEL" && echo "restarted $LABEL" ;;

  status)
    echo "=== launchd ==="
    if [[ ! -f "$AGENT" ]]; then
      echo "not installed - run: ./scripts/goldbot.sh install"
    else
      line=$(launchctl list | grep "$LABEL" || true)
      if [[ -z "$line" ]]; then
        echo "installed but not loaded"
      else
        pid=$(echo "$line" | awk '{print $1}')
        code=$(echo "$line" | awk '{print $2}')
        if [[ "$pid" == "-" ]]; then
          echo "loaded, not currently running (last exit code $code)"
        else
          echo "running, pid $pid, last exit code $code"
        fi
      fi
    fi

    # A job can be alive and still have processed nothing for days, which is the
    # failure worth catching. Only the account itself shows that.
    echo
    echo "=== account ==="
    cd "$REPO" && "$PYTHON" live_gold.py --status 2>&1 | tail -20
    ;;

  logs)
    [[ -f "$LOG" ]] || { echo "no log yet at $LOG"; exit 0; }
    tail -f "$LOG"
    ;;

  *) usage ;;
esac
