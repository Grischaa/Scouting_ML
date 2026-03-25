#!/bin/sh
set -eu

cat > /usr/share/nginx/html/assets/runtime-config.js <<EOF
window.SCOUTING_API_BASE = ${SCOUTING_API_BASE:+\"$SCOUTING_API_BASE\"} || "";
EOF
