#!/bin/bash

# ==============================================================================
# SIMULATION TEST SCRIPT (Config-File-Only Approach)
# ==============================================================================
# This script invokes main.py test_sim with config values loaded from YAML.
# It passes the essential sim.* overrides to Hydra, letting the rest default.
#
# Configuration is defined in:
#   - assets/configs/tasks/test_sim.yaml
#   - assets/configs/data/test_sim.yaml
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
TASK_CONFIG="assets/configs/tasks/test_sim.yaml"

# Load config values for both script display AND Python invocation
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true

# Parse CLI overrides
CLI_OVERRIDES=()
while getopts "qP:d:N:I:C:n:" flag; do
    case "${flag}" in
        q) VERBOSE=false;;
        P) SIM_AREA="${OPTARG}";;
        d) SIM_DATA_DISTRIBUTION="${OPTARG}";;
        N) SIM_SIZE="${OPTARG}";;
        n) SIM_N_SAMPLES="${OPTARG}";;
        I) SIM_POLICIES=(${OPTARG});;
        C) CLI_OVERRIDES+=("${OPTARG}");;
        \?) echo -e "${RED}Invalid option: -${OPTARG}${NC}" >&2; exit 1;;
    esac
done
shift $((OPTIND-1))

# Use loaded or default values
PROBLEM="${SIM_PROBLEM:-cwcvrp}"
AREA="${SIM_AREA:-riomaior}"
SIZE="${SIM_SIZE:-50}"
DAYS="${SIM_DAYS:-31}"
SAMPLES="${SIM_N_SAMPLES:-10}"
WASTE_TYPE="${SIM_WASTE_TYPE:-plastic}"
DATA_DIST="${SIM_DATA_DISTRIBUTION:-gamma1}"
POLICIES="${SIM_POLICIES[@]:-alns bcp}"

# Derived paths
IDX_PATH="graphs_${SIZE}V_1N_${WASTE_TYPE}.json"
DM_PATH="data/wsr_simulator/distance_matrix/gmaps_distmat_${WASTE_TYPE}[${AREA}].csv"
WASTE_PATH="daily_waste/${AREA}${SIZE}_${DATA_DIST}_wsr${DAYS}_N${SAMPLES}_seed${SEED:-42}.npz"

# Format policies as Hydra list
format_policies() {
    local first=true
    local result="["
    for p in ${POLICIES[@]}; do
        if [ "$first" = true ]; then
            result+="'$p'"
            first=false
        else
            result+=",'$p'"
        fi
    done
    result+="]"
    echo "$result"
}
POLICIES_STR=$(format_policies)

# Display configuration
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       SIMULATION TEST MODULE             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:             ${MAGENTA}${PROBLEM}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:                ${MAGENTA}${AREA}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Size:                ${MAGENTA}${SIZE}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Days:                ${MAGENTA}${DAYS}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Data Distribution:   ${MAGENTA}${DATA_DIST}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Samples:             ${MAGENTA}${SAMPLES}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Policies:            ${MAGENTA}${POLICIES[@]}${NC}"
echo ""

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2
    exec >/dev/null 2>&1
fi

# Execute with sim.* overrides from YAML config values
uv run python main.py test_sim \
    "sim.policies=${POLICIES_STR}" \
    "sim.problem='${PROBLEM}'" \
    "sim.graph.area='${AREA}'" \
    "sim.graph.num_loc=${SIZE}" \
    "sim.days=${DAYS}" \
    "sim.n_samples=${SAMPLES}" \
    "sim.graph.waste_type='${WASTE_TYPE}'" \
    "sim.data_distribution='${DATA_DIST}'" \
    "sim.graph.focus_graph='${IDX_PATH}'" \
    "sim.graph.dm_filepath='${DM_PATH}'" \
    "sim.checkpoint_days=30" \
    "sim.graph.edge_threshold='1.0'" \
    "sim.graph.edge_method='dist'" \
    "sim.log_level='WARNING'" \
    "${CLI_OVERRIDES[@]}" \
    "$@"

# Restore output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
fi

echo ""
echo -e "${GREEN}✓ [SUCCESS] Simulation test completed successfully.${NC}"
