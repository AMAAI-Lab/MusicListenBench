#!/usr/bin/env bash
# Downloads the audio archives, checks their SHA-256 sums and unpacks them into data/
# (data/audio/<task>/... and data/audio_perturb/<task>/...).
#
# Usage:
#   bash scripts/fetch_audio.sh <base-url> [--test-only]
#
#   <base-url>   folder that holds the archives and SHA256SUMS
#   --test-only  skip the four training archives (about 3.1 GB) and fetch only the
#                1,000 clean and 2,000 flip/stay test files (about 0.9 GB)
#
# Alternatively, regenerate all audio with the generator (musiclistenbench/generator).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BASE_URL="${1:-}"
if [[ -z "$BASE_URL" || "$BASE_URL" == "-h" || "$BASE_URL" == "--help" ]]; then
    awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "${BASH_SOURCE[0]}"; exit 1
fi
shift
TEST_ONLY=0
[[ "${1:-}" == "--test-only" ]] && TEST_ONLY=1

DATA_DIR="${MLB_DATA_DIR:-data}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DATA_DIR"

ARCHIVES=(mlb_test_clean mlb_test_flip_stay)
if (( ! TEST_ONLY )); then ARCHIVES+=(mlb_train_melody mlb_train_harmony mlb_train_timbre mlb_train_rhythm); fi

curl -fsSL "${BASE_URL%/}/SHA256SUMS" -o "$TMP/SHA256SUMS"
for a in "${ARCHIVES[@]}"; do
    echo "== $a"
    curl -fL "${BASE_URL%/}/${a}.tar.gz" -o "$TMP/${a}.tar.gz"
    (cd "$TMP" && grep " ${a}.tar.gz\$" SHA256SUMS | sha256sum -c -)
    tar -xzf "$TMP/${a}.tar.gz" -C "$DATA_DIR"
    rm "$TMP/${a}.tar.gz"
done
echo "done. Check the files with: python scripts/verify_data.py --audio"
