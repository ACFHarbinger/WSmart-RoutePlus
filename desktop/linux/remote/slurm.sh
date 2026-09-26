#!/bin/bash

# Colors for output
RED='\033[0;31m'
NC='\033[0m' # No Color

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#SBATCH --job-name=train
#SBATCH --time=2-00:00:00

# Slurm parameters for Nexus cluster
# SBATCH --mem=32G # RAM
# SBATCH --mincpus=6
# SBATCH --output=/home/users/afernandes/Repositories/wsr_tmp/job_out_%A.out

# SBATCH --gres=shard:24 # GB of gpu VRAM
# SBATCH --exclude=nexus1, # we can exclude machines if we want

# Slurm parameters for INCD cluster
#SBATCH --partition=gpu
#SBATCH --output=/users5/vlabist/afonsofernandes/Repositories/wsr_tmp/job_out_%A.out
#SBATCH --gres=gpu:v100s

# Slurm parameters for HLT Ada clusters
# SBATCH --mem=64G # RAM
# SBATCH --mincpus=32
# SBATCH --output=/home/u024151/Repositories/wsr_tmp/job_out_%A.out

# Activate the Conda environment
eval "$(conda shell.bash hook)"
#conda activate /home/users/afernandes/anaconda3/envs/wsr
conda activate /users5/vlabist/afonsofernandes/anaconda3/envs/wsr
#conda activate /home/u024151/anaconda3/envs/wsr

# To execute this slurm script: sbatch slurm.sh

# Load Task Config first to get general settings and PROBLEM definition
TASK_CONFIG="assets/configs/tasks/slurm.yaml"
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG")

# Now load the specific environment config based on the problem defined in the task
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$ENV_CONFIG" "$TASK_CONFIG")
    fi
fi

# MAP ENVIRONMENT VARIABLES TO SCRIPT VARIABLES
if [ -n "$SIZE" ]; then GS="$SIZE"; fi
if [ -n "$EDGE_T" ]; then ET="$EDGE_T"; fi

# Use YAML variables as base defaults
NUMBER_OF_CORES=${NUMBER_OF_CORES} # Loaded from YAML

# Read arguments and execute program
while getopts c:m:p:g: flag
do
    case "${flag}" in
        c) command=${OPTARG};;
        m) model=${OPTARG};;
        p) problem=${OPTARG};;
        g) gs=${OPTARG};;
    esac
done

if [[ -z $command ]]; then
    command=${COMMAND}
fi

if [[ -z $model ]]; then
    model=(${MODEL[@]})
fi

if [[ -z $problem ]]; then
    problem=${PROBLEM}
fi

if [[ -z $gs ]]; then
    gs=${GS}
fi

if [[ "$command" == "train" ]]; then
    echo -e "${BLUE}Starting Slurm Job: [TRAIN]${NC}"
    echo -e "${CYAN}[SLURM]${NC} Model:   ${MAGENTA}${model[0]}${NC}"
    echo -e "${CYAN}[SLURM]${NC} Problem: ${MAGENTA}$problem${NC}"
    echo -e "${CYAN}[SLURM]${NC} Size:    ${MAGENTA}$gs${NC}"
    while getopts b:d:n:s:e:l:et: flag
    do
        case "${flag}" in
            b) bs=${OPTARG};;
            d) dd=${OPTARG};;
            n) ne=${OPTARG};;
            s) ebs=${OPTARG};;
            e) es=${OPTARG};;
            l) lp=${OPTARG};;
            et) et=${OPTARG};;
        esac
    done

    if [[ -z $bs ]]; then
        bs=${BS}
    fi

    if [[ -z $dd ]]; then
        dd=${DD_TRAIN}
    fi

    if [[ -z $ne ]]; then
        ne=${NE}
    fi

    if [[ -z $es ]]; then
        es=${ES}
    fi

    if [[ -z $ebs ]]; then
        ebs=${EBS}
    fi

    if [[ -z $et ]]; then
        et=${ET}
    fi

    uv run python main.py "$command" --model "${model[0]}" --baseline rollout --train_dataset virtual \
    --val_dataset data/datasets/wcvrp/wcvrp_unif20_val_seed1234.pkl --problem "$problem" \
    --batch_size "$bs" --data_distribution "$dd" --n_epochs "$ne" --eval_batch_size "$ebs" \
    --graph_size "$gs" --epoch_start "$es" --edge_threshold "$et" --n_other_layers 2;
elif [[ "$command" == "test" ]]; then
    echo -e "${BLUE}Starting Slurm Job: [TEST]${NC}"
    echo -e "${CYAN}[SLURM]${NC} Policies: ${MAGENTA}${model[*]}${NC}"
    echo -e "${CYAN}[SLURM]${NC} Problem:  ${MAGENTA}$problem${NC}"
    echo -e "${CYAN}[SLURM]${NC} Size:     ${MAGENTA}$gs${NC}"
    while getopts d:o:s:l:cf:nv:dd: flag
    do
        case "${flag}" in
            d) days=${OPTARG};;
            o) od=${OPTARG};;
            s) ns=${OPTARG};;
            l) lvl=${OPTARG};;
            cf) cf=${OPTARG};;
            nv) nv=${OPTARG};;
            dd) dd=${OPTARG};;
        esac
    done

    if [[ -z $days ]]; then
        days=${DAYS}
    fi

    if [[ -z $od ]]; then
        od=${OD}
    fi

    if [[ -z $ns ]]; then
        ns=${NS}
    fi

    if [[ -z $lvl ]]; then
        lvl=(${LVL[@]})
    fi

    if [[ -z $cf ]]; then
        cf=(${CF[@]})
    fi

    if [[ -z $dd ]]; then
         dd=${DD_TEST}
    fi

    if [[ -z $nv ]]; then
        nv=${NV}
    fi

    uv run python main.py "$command" --policies "${model[@]}" --dd "$dd" --problem "$problem" --size "$gs" \
    --days "$days" --output_dir "$od" --n_samples "$ns" --cpu_cores "$NUMBER_OF_CORES" --server_run --resume;
    #--lvl "${lvl[@]}" --cf "${cf[@]}" --n_vehicles "$nv"
else
    echo -e "${RED}Unknown command: $command${NC}"
fi
