#!/usr/bin/env bash
# Run the Mumble Voice Bot on sauron as a systemd user service
#
# Usage:
#   ./scripts/run-sauron.sh          # Start the bot
#   ./scripts/run-sauron.sh stop     # Stop the bot
#   ./scripts/run-sauron.sh status   # Check status
#   ./scripts/run-sauron.sh logs     # Follow logs
#   ./scripts/run-sauron.sh restart  # Restart the bot

set -euo pipefail

UNIT_NAME="mumble-voice-bot"
REMOTE="sam@sauron"
WORKDIR="/srv/share/sam/projects/microsoftsam"
CONFIG="config.sauron.yaml"

# Sync local changes to sauron first
sync_changes() {
    echo "Syncing changes to sauron..."
    rsync -av --exclude='.venv*' --exclude='__pycache__' --exclude='*.egg-info' \
        --exclude='.git' --exclude='*.pyc' --exclude='latency.jsonl' \
        . "${REMOTE}:${WORKDIR}/"
}

start_bot() {
    sync_changes
    echo "Starting ${UNIT_NAME}..."
    ssh "${REMOTE}" << EOF
pkill -f mumble_tts_bot || true
sleep 1
systemd-run --user --unit=${UNIT_NAME} --collect \
  --setenv=PATH=/run/current-system/sw/bin:/nix/var/nix/profiles/default/bin:/home/sam/.nix-profile/bin \
  /run/current-system/sw/bin/bash -c \
  'cd ${WORKDIR} && exec nix develop --command uv run python mumble_tts_bot.py --config ${CONFIG}'
sleep 2
systemctl --user status ${UNIT_NAME}
EOF
}

stop_bot() {
    echo "Stopping ${UNIT_NAME}..."
    ssh "${REMOTE}" "systemctl --user stop ${UNIT_NAME} 2>/dev/null || pkill -f mumble_tts_bot || true"
}

status_bot() {
    ssh "${REMOTE}" "systemctl --user status ${UNIT_NAME} 2>/dev/null || echo 'Service not running'"
}

logs_bot() {
    ssh "${REMOTE}" "journalctl --user -u ${UNIT_NAME} -f"
}

logs_recent() {
    ssh "${REMOTE}" "journalctl --user -u ${UNIT_NAME} -n 100 --no-pager"
}

case "${1:-start}" in
    start)
        start_bot
        ;;
    stop)
        stop_bot
        ;;
    restart)
        stop_bot
        sleep 2
        start_bot
        ;;
    status)
        status_bot
        ;;
    logs)
        logs_bot
        ;;
    logs-recent)
        logs_recent
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|logs-recent}"
        exit 1
        ;;
esac
