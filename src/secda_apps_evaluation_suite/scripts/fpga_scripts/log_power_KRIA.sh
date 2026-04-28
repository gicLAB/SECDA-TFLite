#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./log_power_KRIA.sh [--auto] [--interval SECONDS] [LOG_FILE]
# Defaults:
#   INTERVAL=0.05s, LOG_FILE=log.txt

AUTO=1
INTERVAL=0.05
LOG_FILE="log.txt"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto) AUTO=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)  LOG_FILE="$1"; shift ;;  # first non-option is log filename
  esac
done

# Find INA260
HWMON_PATH=""
for hwmon in /sys/class/hwmon/hwmon*; do
  if grep -iq "ina260" "$hwmon/name"; then
    HWMON_PATH="$hwmon"
    break
  fi
done
if [[ -z "$HWMON_PATH" ]]; then
  echo "❌ INA260 hwmon device not found."
  exit 1
fi

echo "✅ INA260 found at $HWMON_PATH"
echo "ℹ️  Sampling every ${INTERVAL}s"
echo "📄 Logging to $LOG_FILE"

# Start prompt or auto-start
if [[ "$AUTO" -eq 0 ]]; then
  echo "▶️  Press Enter to START logging..."
  read -r
else
  echo "⏩ Auto-start logging"
fi

# Graceful stop on SIGINT/SIGTERM
stop=0
trap 'stop=1' INT TERM

# Header
echo "power_uW" > "$LOG_FILE"

# Logging loop
while [[ "$stop" -eq 0 ]]; do
  POWER=$(cat "$HWMON_PATH/power1_input" 2>/dev/null || echo "")
  # If needed also capture voltage/current:
  # VOLT=$(cat "$HWMON_PATH/in1_input" 2>/dev/null || echo "")
  # CURR=$(cat "$HWMON_PATH/curr1_input" 2>/dev/null || echo "")
  # printf "%s,%s,%s\n" "$POWER" "$VOLT" "$CURR" >> "$LOG_FILE"
  [[ -n "$POWER" ]] && echo "$POWER" >> "$LOG_FILE"
  # sleep "$INTERVAL"
done

echo "✅ Logging stopped. Results saved in $LOG_FILE"
