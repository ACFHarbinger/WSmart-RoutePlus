#!/bin/bash

# ==============================================================================
# META-RL TRAINING SCRIPT (Config-File-Only Approach)
# ==============================================================================
# This script invokes main.py train_lightning with Meta-RL experiment config.
# Configuration is defined in:
#   - assets/configs/tasks/meta_train.yaml
#   - assets/configs/data/meta_train.yaml
# ==============================================================================

set -e

# Handle memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
TASK_CONFIG="assets/configs/tasks/meta_train.yaml"

# Load config
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true

# Load environment config based on problem
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$ENV_CONFIG" "$TASK_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true
    fi
fi

# Parse CLI overrides
while getopts "q" flag; do
    case "${flag}" in
        q) VERBOSE=false;;
        \?) echo -e "${RED}Invalid option: -${OPTARG}${NC}" >&2; exit 1;;
    esac
done
shift $((OPTIND-1))

# Use loaded or default values
PROBLEM="${PROBLEM:-cwcvrp}"
AREA="${AREA:-riomaior}"
SIZE="${SIZE:-50}"
EPOCHS="${EPOCHS:-100}"
MODEL="${MODEL_NAMES[0]:-am}"
ENCODER="${MODEL_ENCODERS[0]:-gat}"
META_METHOD="${META_METHOD:-meta_rnn}"

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       META-RL TRAINING MODULE            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}${PROBLEM}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph size: ${MAGENTA}${SIZE}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}${AREA}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Epochs:     ${MAGENTA}${EPOCHS}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Method:     ${MAGENTA}${META_METHOD}${NC}"
echo ""

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2
    exec >/dev/null 2>&1
fi

# Execute Meta-RL training
uv run python main.py train \
    "experiment=meta_rl" \
    "env.name='${PROBLEM}'" \
    "env.num_loc=${SIZE}" \
    "env.area='${AREA}'" \
    "model.name='${MODEL}'" \
    "model.encoder_type='${ENCODER}'" \
    "train.n_epochs=${EPOCHS}" \
    "rl.meta_strategy='${META_METHOD}'" \
    "seed=${SEED:-42}" \
    "$@"

# Restore output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
fi

echo ""
echo -e "${GREEN}✓ [DONE] Meta-RL Training completed.${NC}"
