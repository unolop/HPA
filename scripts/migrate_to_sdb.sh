#!/bin/bash
# Migrate large dirs to /mnt/sdb and symlink back.
# Safe: rsync first, verify, then remove original and create symlink.

set -e
LOG=/home/david/Desktop/yuna/HPA/logs/migrate_to_sdb.log
exec > >(tee -a "$LOG") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }

# ── 1. HF model cache (~245G) ─────────────────────────────────────────────────
SRC_HF="/home/david/Desktop/yuna/.cache/hf"
DST_HF="/mnt/sdb/yuna/.cache/hf"

log "=== Migrating HF cache: $SRC_HF → $DST_HF ==="
mkdir -p "$DST_HF"
rsync -a --info=progress2 "$SRC_HF/" "$DST_HF/"
log "rsync done. Verifying..."

SRC_COUNT=$(find "$SRC_HF" -type f | wc -l)
DST_COUNT=$(find "$DST_HF" -type f | wc -l)
log "Source files: $SRC_COUNT  Dest files: $DST_COUNT"

if [ "$SRC_COUNT" -ne "$DST_COUNT" ]; then
    log "ERROR: file counts differ — aborting, original untouched"
    exit 1
fi

log "Removing original and creating symlink..."
rm -rf "$SRC_HF"
ln -s "$DST_HF" "$SRC_HF"
log "✓ HF cache → $DST_HF (symlinked)"

df -h /dev/nvme0n1p2 | tail -1

# ── 2. VQA data (~35G) ────────────────────────────────────────────────────────
SRC_DATA="/home/david/Desktop/yuna/data"
DST_DATA="/mnt/sdb/yuna/data"

log "=== Migrating data: $SRC_DATA → $DST_DATA ==="
mkdir -p "$DST_DATA"
rsync -a --info=progress2 "$SRC_DATA/" "$DST_DATA/"
log "rsync done. Verifying..."

SRC_COUNT=$(find "$SRC_DATA" -type f | wc -l)
DST_COUNT=$(find "$DST_DATA" -type f | wc -l)
log "Source files: $SRC_COUNT  Dest files: $DST_COUNT"

if [ "$SRC_COUNT" -ne "$DST_COUNT" ]; then
    log "ERROR: file counts differ — aborting, original untouched"
    exit 1
fi

log "Removing original and creating symlink..."
rm -rf "$SRC_DATA"
ln -s "$DST_DATA" "$SRC_DATA"
log "✓ data → $DST_DATA (symlinked)"

df -h /dev/nvme0n1p2 | tail -1
log "=== Migration complete ==="
