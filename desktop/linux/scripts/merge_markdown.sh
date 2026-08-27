#!/bin/bash

# Define the source directory and output file
SRC_DIR="markdown"
DST_DIR="markdown"
OUTPUT="$DST_DIR/algorithms.md"

# Define the files in the specific mandatory order
FILES=(
    "$SRC_DIR/mandatory_selection.md"
    "$SRC_DIR/route_construction.md"
    "$SRC_DIR/acceptance_criteria.md"
    "$SRC_DIR/route_improvement.md"
)

# Verify all files exist before proceeding
for file in "${FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Required file $file not found."
        exit 1
    fi
done

echo "Merging files into $OUTPUT..."

# Use pandoc to merge
# The --resource-path allows pandoc to find local images referenced in the subfolders
pandoc "${FILES[@]}" \
    -f markdown-yaml_metadata_block \
    -t markdown-smart \
    --wrap=none \
    --markdown-headings=atx \
    -s -o "$OUTPUT"

if [ $? -eq 0 ]; then
    echo "Success! Files merged in the following order:"
    printf "  - %s\n" "${FILES[@]}"
else
    echo "An error occurred during the pandoc conversion."
fi
