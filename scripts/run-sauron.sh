#!/usr/bin/env bash
# Run the Mumble Voice Bot on sauron as a systemd user service
#
# Usage:
#   ./scripts/run-sauron.sh          # Start the bot
#   ./scripts/run-sauron.sh stop     # Stop the bot
#   ./scripts/run-sauron.sh status   # Check status
#   ./scripts/run-sauron.sh logs     # Follow logs
#   ./scripts/run-sauron.sh restart  # Restart the bot
#   ./scripts/run-sauron.sh sync-voices  # Sync voice files (not in git)

set -euo pipefail

UNIT_NAME="mumble-voice-bot"
REMOTE="sam@sauron"
WORKDIR="/srv/share/sam/projects/microsoftsam"
CONFIG="config.sauron.yaml"
REPO_URL="git@github.com:alam0rt/microsoftsam.git"

# Sync code changes via git
sync_changes() {
    echo "Syncing changes to sauron via git..."
    ssh "${REMOTE}" << EOF
set -e
if [ ! -d "${WORKDIR}/.git" ]; then
    echo "Cloning repository..."
    mkdir -p "$(dirname ${WORKDIR})"
    git clone --recurse-submodules "${REPO_URL}" "${WORKDIR}"
else
    echo "Pulling latest changes..."
    cd "${WORKDIR}"
    git fetch origin
    git reset --hard origin/main
    git submodule update --init --recursive
fi
EOF
}

# Sync voice files and other non-git assets
sync_voices() {
    echo "Syncing voice files to sauron..."
    rsync -av --progress \
        --include='*.wav' \
        --include='voices/' \
        --include='voices/**' \
        --exclude='*' \
        . "${REMOTE}:${WORKDIR}/"
}

# Sync souls directory (personalities, voice references, weights)
sync_souls() {
    echo "Syncing souls to sauron..."
    rsync -av --progress \
        souls/ "${REMOTE}:${WORKDIR}/souls/"
}

# Sync secrets (API keys, etc.) - stored in .envrc locally
sync_secrets() {
    if [ ! -f ".envrc" ]; then
        echo "No .envrc file found. Create it with your API keys:"
        echo "  echo 'export OPENROUTER_API_KEY=sk-or-...' > .envrc"
        exit 1
    fi
    echo "Syncing .envrc to sauron..."
    rsync -av --chmod=F600 \
        .envrc "${REMOTE}:${WORKDIR}/.envrc"
    echo ".envrc synced to ${WORKDIR}/.envrc on sauron"
}

start_bot() {
    sync_changes
    echo "Starting ${UNIT_NAME}..."
    # Note: Using a script on remote to handle secrets properly
    ssh "${REMOTE}" bash -s "${WORKDIR}" "${CONFIG}" << 'EOFSCRIPT'
WORKDIR="$1"
CONFIG="$2"

pkill -f mumble_tts_bot || true
sleep 1

# Source .envrc if it exists (contains API keys)
if [ -f "${WORKDIR}/.envrc" ]; then
    source "${WORKDIR}/.envrc"
fi

systemd-run --user --unit=mumble-voice-bot --collect \
  --setenv=PATH=/run/current-system/sw/bin:/nix/var/nix/profiles/default/bin:/home/sam/.nix-profile/bin \
  --setenv=OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
  /run/current-system/sw/bin/bash -c \
  "cd ${WORKDIR} && exec nix develop --command uv run python mumble_tts_bot.py --config ${CONFIG}"
sleep 2
systemctl --user status mumble-voice-bot
EOFSCRIPT
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
    sync-voices)
        sync_voices
        ;;
    sync-souls)
        sync_souls
        ;;
    sync-secrets)
        sync_secrets
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|logs-recent|sync-voices|sync-souls|sync-secrets}"
        exit 1
        ;;
esac
