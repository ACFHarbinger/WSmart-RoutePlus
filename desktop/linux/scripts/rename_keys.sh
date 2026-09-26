#!/bin/bash

# Define the target directory (default is current directory)
TARGET_DIR=${1:-.}

echo "Starting replacement in directory: $TARGET_DIR"

# 1. Change 'must_go' to 'mandatory_selection'
echo "Updating: must_go -> mandatory_selection"
find "$TARGET_DIR" -type f -not -path '*/.*' -exec sed -i 's/must_go/mandatory_selection/g' {} +

# 2. Change 'post_processing' to 'route_improvement'
echo "Updating: post_processing -> route_improvement"
find "$TARGET_DIR" -type f -not -path '*/.*' -exec sed -i 's/post_processing/route_improvement/g' {} +

echo "Replacement complete."
