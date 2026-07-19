#!/usr/bin/env bash
# Regenerate README.md and README-1.svg from README.template.md.
#
# README.md is GENERATED — edit README.template.md and run this script
# (requires mermaid-cli: `npm install -g @mermaid-js/mermaid-cli`).
set -euo pipefail
cd "$(dirname "$0")/.."

TEMPLATE=README.template.md
SVG=README-1.svg
OUT=README.md

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# Extract the (single) mermaid block and render it.
awk '/^```mermaid$/{f=1;next} /^```$/{f=0} f' "$TEMPLATE" > "$tmpdir/diagram.mmd"
if [ ! -s "$tmpdir/diagram.mmd" ]; then
    echo "error: no mermaid block found in $TEMPLATE" >&2
    exit 1
fi
mmdc -i "$tmpdir/diagram.mmd" -o "$SVG" --quiet

# Emit README.md with the mermaid block replaced by the rendered image.
awk -v svg="$SVG" '
    /^```mermaid$/ { printf "![diagram](./%s)\n", svg; skip=1; next }
    skip && /^```$/ { skip=0; next }
    !skip
' "$TEMPLATE" > "$OUT"

echo "regenerated $OUT and $SVG from $TEMPLATE"
