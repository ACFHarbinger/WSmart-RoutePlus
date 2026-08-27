#!/bin/bash

# ==============================================================================
# DATA GENERATION SCRIPT (Config-File-Only Approach)
# ==============================================================================
# This script invokes main.py gen_data with config values loaded from YAML.
# Configuration is defined in:
#   - assets/configs/tasks/gen_data.yaml
#   - assets/configs/data/gen_data.yaml
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
TASK_CONFIG="assets/configs/tasks/gen_data.yaml"

# Load config
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true

# Load environment config based on problem
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$ENV_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true
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
PROBLEM="${DATA_PROBLEM:-${PROBLEM:-cwcvrp}}"
AREA="${DATA_AREA:-${AREA:-riomaior}}"
WTYPE="${DATA_WASTE_TYPE:-${WTYPE:-plastic}}"
SIZES=(${DATA_NUM_LOCS[@]:-${SIZES[@]:-50}})
SEED="${DATA_SEED:-${SEED:-42}}"
DATA_DIR="${DATA_DATA_DIR:-${DATA_DIR:-data/datasets}}"
N_DATA="${DATA_DATASET_SIZE:-${N_DATA:-1280}}"

# Hydra list formating helper
format_hydra_list() {
    local arr=("$@")
    local first=true
    local result="["
    for item in "${arr[@]}"; do
        if [ "$first" = true ]; then
            result+="'$item'"
            first=false
        else
            result+=",'$item'"
        fi
    done
    result+="]"
    echo "$result"
}

# Derived values
FOCUS_GRAPHS=()
for size in ${SIZES[@]}; do
    FOCUS_GRAPHS+=("data/wsr_simulator/bins_selection/graphs_${size}V_1N_${WTYPE}.json")
done

SIZES_STR=$(format_hydra_list "${SIZES[@]}")
FOCUS_GRAPHS_STR=$(format_hydra_list "${FOCUS_GRAPHS[@]}")

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       DATA GENERATION MODULE             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:          ${MAGENTA}${PROBLEM}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:             ${MAGENTA}${AREA}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Sizes:            ${MAGENTA}${SIZES[*]}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Waste Type:       ${MAGENTA}${WTYPE}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Dataset Size:     ${MAGENTA}${N_DATA}${NC}"
echo ""

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2
    exec >/dev/null 2>&1
fi

GENERATE_DATASET=1
GENERATE_VAL_DATASET=1
GENERATE_TEST_DATASET=0

# Generate main dataset
if [ "${GENERATE_DATASET:-0}" -eq 0 ]; then
    echo -e "${BLUE}Generating main dataset...${NC}"
    uv run python main.py gen_data \
        "data.name=${DATA_FILENAME:-train}" \
        "data.problem=$PROBLEM" \
        "data.overwrite=true" \
        "data.waste_type=$WTYPE" \
        "data.num_locs=${SIZES_STR}" \
        "data.dataset_size=$N_DATA" \
        "data.dataset_type=train_time" \
        "data.n_epochs=${DATA_N_EPOCHS:-100}" \
        "data.epoch_start=${DATA_EPOCH_START:-${START:-0}}" \
        "data.focus_graphs=${FOCUS_GRAPHS_STR}" \
        "data.focus_size=${DATA_FOCUS_SIZE:-0}" \
        "data.vertex_method=${DATA_VERTEX_METHOD:-${VERTEX_METHOD:-mmn}}" \
        "data.data_distributions=$(format_hydra_list "${DATA_DATA_DISTRIBUTIONS[@]:-all}")" \
        "data.data_dir=$DATA_DIR" \
        "data.area=$AREA" \
        "data.seed=$SEED" \
        "$@"
fi

# Generate validation dataset
if [ "${GENERATE_VAL_DATASET:-0}" -eq 0 ]; then
    echo -e "${BLUE}Generating validation dataset...${NC}"
    uv run python main.py gen_data \
        "data.name=${DATA_FILENAME:-train}_val" \
        "data.problem=$PROBLEM" \
        "data.overwrite=true" \
        "data.waste_type=$WTYPE" \
        "data.num_locs=${SIZES_STR}" \
        "data.dataset_size=${DATA_VAL_SIZE:-1280}" \
        "data.dataset_type=train_time" \
        "data.n_epochs=${DATA_N_EPOCHS:-100}" \
        "data.epoch_start=${DATA_EPOCH_START:-${START:-0}}" \
        "data.focus_graphs=${FOCUS_GRAPHS_STR}" \
        "data.focus_size=${DATA_FOCUS_SIZE:-0}" \
        "data.vertex_method=${DATA_VERTEX_METHOD:-${VERTEX_METHOD:-mmn}}" \
        "data.data_distributions=$(format_hydra_list "${DATA_DATA_DISTRIBUTIONS[@]:-all}")" \
        "data.data_dir=$DATA_DIR" \
        "data.area=$AREA" \
        "data.seed=$SEED" \
        "$@"
fi

# Generate test dataset
if [ "${GENERATE_TEST_DATASET:-0}" -eq 0 ]; then
    echo -e "${BLUE}Generating test dataset...${NC}"
    uv run python main.py gen_data \
        "data.name=${DATA_FILENAME:-test}" \
        "data.problem=$PROBLEM" \
        "data.overwrite=true" \
        "data.area=$AREA" \
        "data.vertex_method=${VERTEX_METHOD:-mmn}" \
        "data.epoch_start=${START:-0}" \
        "data.seed=$SEED" \
        "data.n_epochs=${N_EPOCHS:-1}" \
        "data.data_distributions=$(format_hydra_list "${DATA_DATA_DISTRIBUTIONS[@]:-all}")" \
        "data.dataset_type=test_simulator" \
        "data.focus_graphs=${FOCUS_GRAPHS_STR}" \
        "data.focus_size=${TEST_FOCUS_SIZE:-0}" \
        "data.data_dir=${SIM_DATA_DIR:-$DATA_DIR}" \
        "data.waste_type=$WTYPE" \
        "data.num_locs=${SIZES_STR}" \
        "data.dataset_size=${DATA_TEST_SIZE:-1280}" \
        "$@"
fi

# Restore output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
fi

echo ""
echo -e "${GREEN}✓ [DONE] Data generation process completed.${NC}"
