#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/assets"
curl -L --fail --retry 2 \
  -o "$ROOT/assets/pexels_parking_pov.mp4" \
  'https://videos.pexels.com/video-files/18495959/18495959-uhd_2560_1440_25fps.mp4'

echo "Downloaded $ROOT/assets/pexels_parking_pov.mp4"
