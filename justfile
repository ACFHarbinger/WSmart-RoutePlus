# WSmart+ Route Framework - Justfile

set unstable := true

red := '\033[0;31m'
green := '\033[0;32m'
yellow := '\033[0;33m'
blue := '\033[0;34m'
purple := '\033[0;35m'
cyan := '\033[0;36m'
bold := '\033[1m'
reset := '\033[0m'

# Default variables (can be overridden: just train problem=wcvrp)

problem := "vrpp"
model := "am"
encoder := "gat"
decoder := "glimpse"
area := "figueiradafoz"
num_loc := "350"
sim_distribution := "emp"
batch_size := "64"
temporal_horizon := "0"
samples := "1"
seed := "42"
marker := "fast"
strategy := "greedy"
n_cores := "20"
hpo_policy := "bpc"
hpo_method := "nsgaii"
hpo_workers := "1"
hpo_selection := ""
hpo_acceptance := ""
hpo_improver := ""
hpo_trials := "10"
hpo_samples := "5"
hpo_policy_kw := "max_cg_iterations"
hpo_selection_kw := ""
hpo_acceptance_kw := ""
hpo_improver_kw := ""
policies := "aco_hh,alns,bpc,hgs,pg_clns,psoma,sans,swc_tcf"
results_dir := "assets/output/30_days/riomaior_100"
distribution := ""
constructor := ""
ms_strategy := ""
improver := ""
dry_run := "false"
quiet := "false"
batch_cfg := "logic/configs/batch.yaml"
fail_fast := "true"
resume := "true"

# --- Submodules ---

mod agent 'tools/agent'
mod app 'tools/app'
mod benchmark 'tools/benchmark'
mod ci 'tools/ci'
mod controller 'tools/controller'
mod database 'tools/database'
mod export 'tools/export'
mod helper 'tools/helper'
mod infrastructure 'tools/infrastructure'
mod reducer 'tools/reducer'
mod script 'tools/script'
mod test 'tools/test'
mod validation 'tools/validation'

# --- Help ---

# Print available commands with descriptions
help: helper::_print_header
    just helper::help

# --- Shorthands ---

# Initialize environment and install all dependencies
setup: helper::_print_header
    just infrastructure::setup

# Sync dependencies using uv
sync: helper::_print_header
    just infrastructure::sync

# Train a model with Hydra configs (problem=vrpp model=am)
train problem=problem model=model encoder=encoder decoder=decoder batch_size=batch_size temporal_horizon=temporal_horizon: helper::_print_header
    just controller::train '{{ problem }}' '{{ model }}' '{{ encoder }}' '{{ decoder }}' '{{ batch_size }}' '{{ temporal_horizon }}'

# Run model evaluation with Hydra configs
eval model_path="" dataset="" problem=problem strategy=strategy samples=samples: helper::_print_header
    just controller::eval '{{ model_path }}' '{{ dataset }}' '{{ problem }}' '{{ strategy }}' '{{ samples }}'

# Run multi-day simulator test
test-sim policies=policies area=area samples=samples n_cores=n_cores num_loc=num_loc sim_distribution=sim_distribution: helper::_print_header
    just controller::test-sim '{{ policies }}' '{{ area }}' '{{ samples }}' '{{ n_cores }}' '{{ num_loc }}' '{{ sim_distribution }}'

# Remove targeted simulation runs from output artefacts
clean-results results_dir=results_dir distribution=distribution constructor=constructor ms_strategy=ms_strategy improver=improver dry_run=dry_run quiet=quiet: helper::_print_header
    just reducer::clean-results '{{ results_dir }}' '{{ distribution }}' '{{ constructor }}' '{{ ms_strategy }}' '{{ improver }}' '{{ dry_run }}' '{{ quiet }}'

# Run simulation policy HPO
hpo-sim policy=hpo_policy trials=hpo_trials method=hpo_method workers=hpo_workers selection=hpo_selection acceptance=hpo_acceptance improver=hpo_improver policy_kw=hpo_policy_kw selection_kw=hpo_selection_kw acceptance_kw=hpo_acceptance_kw improver_kw=hpo_improver_kw area=area samples=hpo_samples: helper::_print_header
    just controller::hpo-sim '{{ policy }}' '{{ trials }}' '{{ method }}' '{{ workers }}' '{{ selection }}' '{{ acceptance }}' '{{ improver }}' '{{ policy_kw }}' '{{ selection_kw }}' '{{ acceptance_kw }}' '{{ improver_kw }}' '{{ area }}' '{{ samples }}'

# Generate a dataset
gen-data problem=problem: helper::_print_header
    just controller::gen-data '{{ problem }}'

# Generate Figueira da Foz plastic datasets
gen-data-figfoz-plastic: helper::_print_header
    just controller::gen-data-figfoz-plastic

# Launch WSmart-Route Studio (Tauri desktop app — native window, hot-reload)
studio: helper::_print_header
    just app::tauri-dev

# Build a WSmart-Route Studio release binary (installer in app/src-tauri/target/release/bundle/)
studio-build: helper::_print_header
    just app::build

# Install Studio JS/TS dependencies (run once after checkout)
studio-install: helper::_print_header
    just app::install

# Type-check the Studio TypeScript frontend
studio-check: helper::_print_header
    just app::check

# Lint the Studio Rust backend with Cargo Clippy
studio-clippy: helper::_print_header
    just app::clippy

# Delete persisted Studio settings (Tauri Store) — dev/testing only
studio-reset: helper::_print_header
    just app::reset-data

# Run fast unit tests (use `just test::test` for the full suite)
test-fast: helper::_print_header
    just test::test-fast

# Check code quality with ruff
lint: helper::_print_header
    just ci::lint

# Format code with ruff
format: helper::_print_header
    just ci::format

# Clean caches and build artifacts
clean: helper::_print_header
    just reducer::clean

# Remove simulation output files
clean-outputs: helper::_print_header
    just reducer::clean-outputs

# Remove large outputs (checkpoints and realtime logs)
clean-large-outputs: helper::_print_header
    just reducer::clean-large-outputs

# Build stripped-simulator executable (run after pruning)
package: helper::_print_header
    just infrastructure::package

# Run automated algorithm export (creates algo-export branch)
algo-export constructors="" selectors="" improvement="" acceptance="" joint="" models="" rl_algorithms="" imitation_policies="" drop_features="" envs="" sim_datasets="" distributions="" network="" skip_build="false" dry_run="false": helper::_print_header
    just export::algo-export '{{ constructors }}' '{{ selectors }}' '{{ improvement }}' '{{ acceptance }}' '{{ joint }}' '{{ models }}' '{{ rl_algorithms }}' '{{ imitation_policies }}' '{{ drop_features }}' '{{ envs }}' '{{ sim_datasets }}' '{{ distributions }}' '{{ network }}' '{{ skip_build }}' '{{ dry_run }}'

# Generic run command — pass any main.py arguments directly
run *args: helper::_print_header
    uv run python main.py {{ args }}

# Regenerate paper tables and figures, and compile paper PDF
paper force="false": helper::_print_header
    just script::paper '{{ force }}'


# Run a batch of experiments from a YAML config file
batch-run batch_cfg=batch_cfg dry_run=dry_run fail_fast=fail_fast n_cores=n_cores resume=resume: helper::_print_header
    just controller::batch-run '{{ batch_cfg }}' '{{ dry_run }}' '{{ fail_fast }}' '{{ n_cores }}' '{{ resume }}'

# Commit using the .gitmessage template
commit message: helper::_print_header
    just helper::commit '{{ message }}'

# Loop the Claude Code agent on a stateful context with live-streaming reasoning steps
loop-claude prompt="Continue implementing the studio, updating the ROADMAP and CHANGELOG, and commiting your work": helper::_print_header
    just agent::loop-claude '{{ prompt }}'

# Loop the Grok agent on a stateful context with live-streaming reasoning steps
loop-grok prompt="Continue implementing the studio, updating the ROADMAP and CHANGELOG, and commiting your work": helper::_print_header
    just agent::loop-grok '{{ prompt }}'

# Loop the Gemini agent on a stateful context with live-streaming reasoning steps
loop-gemini prompt="Continue implementing the studio, updating the ROADMAP and CHANGELOG, and commiting your work": helper::_print_header
    just agent::loop-gemini '{{ prompt }}'

# Loop the ChatGPT agent on a stateful context with live-streaming reasoning steps
loop-chatgpt prompt="Continue implementing the studio, updating the ROADMAP and CHANGELOG, and commiting your work": helper::_print_header
    just agent::loop-chatgpt '{{ prompt }}'
