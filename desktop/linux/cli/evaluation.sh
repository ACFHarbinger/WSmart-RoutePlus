#!/bin/bash

# ==============================================================================
# EVALUATION SCRIPT (Config-File-Only Approach)
# ==============================================================================
# This script invokes main.py eval with config values loaded from YAML.
# Configuration is defined in:
#   - assets/configs/tasks/evaluation.yaml
#   - assets/configs/data/eval_data.yaml
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default to verbose mode
VERBOSE=true

# Configuration files
TASK_CONFIG="assets/configs/tasks/evaluation.yaml"
DATA_CONFIG="assets/configs/data/eval_data.yaml"

# Load config
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true

# Load environment config based on problem
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" "$ENV_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true
    fi
fi

# Parse CLI overrides
while getopts "qm:D:O:fw:C:np" flag; do
    case "${flag}" in
        q) VERBOSE=false;;
        m) MODEL_PATH=${OPTARG};;
        D) DATASETS=(${OPTARG});;
        O) OUTPUT_FILE=${OPTARG};;
        f) OVERWRITE_FLAG="-f";;
        w) WIDTH=(${OPTARG});;
        C) DECODE_STRATEGY=${OPTARG};;
        n) NO_CUDA_FLAG="--no_cuda";;
        p) NO_PROGRESS_BAR_FLAG="--no_progress_bar";;
        \?) echo -e "${RED}Invalid option: -${OPTARG}${NC}" >&2; exit 1;;
    esac
done
shift $((OPTIND-1))

# Use loaded or default values
DECODE_TYPE="${DECODE_TYPE:-greedy}"
DECODE_STRATEGY="${DECODE_STRATEGY:-greedy}"
VAL_SIZE="${VAL_SIZE:-10000}"
GRAPH_SIZE="${SIZE:-50}"
AREA="${AREA:-riomaior}"

# Display configuration
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       EVALUATION MODULE                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo -e "${CYAN}[PARAM]${NC} Model:            ${MAGENTA}${MODEL_PATH:-default}${NC}"
echo -e "${CYAN}[PARAM]${NC} Datasets:         ${MAGENTA}${DATASETS[*]:-default}${NC}"
echo -e "${CYAN}[PARAM]${NC} Decode Strategy:  ${MAGENTA}${DECODE_STRATEGY}${NC}"
echo ""

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2
    exec >/dev/null 2>&1
fi

# Build command array
PYTHON_CMD=(
    uv run python main.py eval
    --val_size "$VAL_SIZE"
    --strategy "$DECODE_TYPE"
    --decode_strategy "$DECODE_STRATEGY"
    --graph_size "$GRAPH_SIZE"
    --area "$AREA"
    $NO_CUDA_FLAG
    $NO_PROGRESS_BAR_FLAG
    $OVERWRITE_FLAG
)

# Add optional args
[ -n "$MODEL_PATH" ] && PYTHON_CMD+=(--model "$MODEL_PATH")
[ ${#DATASETS[@]} -gt 0 ] && PYTHON_CMD+=(--datasets "${DATASETS[@]}")
[ -n "$OUTPUT_FILE" ] && PYTHON_CMD+=(-o "$OUTPUT_FILE")
[ ${#WIDTH[@]} -gt 0 ] && PYTHON_CMD+=(--width "${WIDTH[@]}")

"${PYTHON_CMD[@]}" "$@"

# Restore output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
fi

echo ""
echo -e "${GREEN}✓ [DONE] Evaluation completed.${NC}"
