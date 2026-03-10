#!/bin/bash

# Source and destination directories
SRC_DIR="../DecisionLens-broken"
DEST_DIR="../DecisionLens"

cd "$SRC_DIR" || exit 1

# Find files changed today (excluding .git)
FILES=$(find . -type f -newermt "$(date +%Y-%m-%d)" ! -path "./.git/*")

for file in $FILES; do
  # Create parent directories in DEST_DIR if needed
  mkdir -p "$DEST_DIR/$(dirname "$file")"
  # Copy the file
  cp "$file" "$DEST_DIR/$file"
  echo "Copied $file"
done

echo "All changed files copied!"