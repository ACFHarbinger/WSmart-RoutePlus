# WSmart Route+ 1.0.2 (minimal package)

A lightweight version of the WSmart Route+ simulator: the multi-day waste-collection simulator
with four routing algorithms and three packing algorithms, run on existing datasets, and the
plain Attention Model trained with REINFORCE on existing `.td` datasets. It is built from the
WSmart-Route code by `logic/package/build_release.py --manifest logic/package/resources/minimal.yaml`.

Not included: the other routing and packing algorithms, route improvement and acceptance
criteria, time windows and time constraints (CPTPTC), experiment tracking, the Studio telemetry,
HPO, meta-learning, imitation learning, dataset generation and the `eval` and `gen_data` tasks.
Results differ from the full release's, which improves every route with local search.

## What is included

| Area | Components |
| --- | --- |
| Problems | `ptp` (Profitable Tour Problem), `cptp` (Capacitated PTP) |
| Routing algorithms | `na` (Attention Model), `swc_tcf` (Gurobi), `sans`, `pg_clns` |
| Packing (which bins must be collected) | `last_minute`, `lookahead`, `service_level` |
| Neural training | Attention Model (graph-attention encoder), REINFORCE with rollout baseline |
| Trained models | the paper's six models: Rio Maior 100 and 170 and Figueira da Foz 350, Empirical and Gamma-3 |

`docs/release.md` lists every component and the module that implements it.

## Files

```
__main__.py              entry point (`python . <task>`): test_sim | train
pyproject.toml           the package and its dependencies (extras: solvers, neural, test), pinned to the tested versions
.python-version          the Python version uv installs (3.12; the package supports Python 3.10 to 3.12)
logic/configs/           Hydra YAML (tasks, envs, models, policies); the policy files are grouped
                         like the code: policies/routing/<category>/policy_<key>.yaml and
                         policies/other/{packing,route_improvement,acceptance_criteria}/
logic/controllers/       task dispatch and job runners
logic/src/               configs, data, envs, interfaces, models, pipeline (learning, features,
                         simulations), policies (routing, packing, shared operators) and utils
                         (utils/results holds the simulator's result writers)
logic/release.yaml       the package manifest
logic/test/              the package tests (see Running the tests)
logic/submodules/wsmart_bin_analysis/
                         bin-analysis package (GridBase), used by the empirical distribution
docs/release.md          the included components
model_weights/           the paper's trained Attention Models (cptp<N>_<area>_plastic/<emp|gamma3>/amgat0-il-rl)
assets/diagrams/         the two simulator diagrams (PDF)
assets/files/FILE_LIST.txt
                         listing of the code files
scripts/linux/setup.sh     Linux setup: uv, code, virtual environment, data
scripts/windows/setup.bat  Windows setup (Command Prompt), same steps
scripts/windows/setup.ps1  Windows setup (PowerShell), same steps
data.zip                 runtime data, unpacked by the setup scripts into:
data/simulator/            distance and time matrices, coordinates, fill histories, bin selections
                           and the paper's pre-generated .npz simulation datasets
data/datasets/             small .td training datasets (20 bins, one day) for trying out training
```

## Setup

On Windows, setup installs the Microsoft Visual C++ x64 runtime when it is missing (torch needs
it), and Windows asks an administrator to approve that. Where nobody can approve the prompt (a
remote or unattended session), setup stops with an error after a couple of minutes; install the
runtime first from an administrator shell with `vc_redist.x64.exe /install /passive /norestart`
([download](https://aka.ms/vs/17/release/vc_redist.x64.exe)) and then run setup.

Run the setup script for your system from anywhere inside the package:

```bash
bash scripts/linux/setup.sh        # Linux
scripts\windows\setup.bat          # Windows, Command Prompt
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1   # Windows, PowerShell
```

Each script:

1. on Windows, installs the Microsoft Visual C++ x64 runtime if it is missing (see above);
2. installs [uv](https://docs.astral.sh/uv/) if it is not installed yet;
3. creates the virtual environment `.venv` with Python 3.12 (from `.python-version`; uv downloads
   it if needed, and no Python needs to be installed beforehand) and the package, its solvers,
   its neural training stack and the test tools, or updates it if it already exists (on Windows with an NVIDIA
   GPU it also installs the CUDA build of torch);
4. unpacks `data.zip` into `data/`.

Run it again at any time to update the environment or to unpack a new `data.zip`. Then
activate the environment:

```bash
source .venv/bin/activate          # Linux
.venv\Scripts\activate             # Windows (PowerShell: .venv\Scripts\Activate.ps1)
```

`swc_tcf` uses Gurobi and needs a Gurobi licence. Gurobi finds it at `C:\gurobi\gurobi.lic` on
Windows and `~/gurobi.lic` on Linux, or wherever `GRB_LICENSE_FILE` points.

## Running

```bash
# Simulate 10 days on Rio Maior (100 bins, the paper's Empirical dataset) with all four routing algorithms
python . test_sim sim.graph.area=riomaior sim.graph.num_loc=100 sim.graph.n_days=10 sim.data_distribution=emp \
  sim.graph.load_dataset=data/simulator/datasets/riomaior100_emp_wsr30_N1_seed42.npz \
  sim.graph.distance_method=ogd 'sim.graph.dm_filepath="gmaps_distmat_plastic[riomaior].csv"' \
  sim.graph.focus_graph=graphs_100V_1N_plastic.json "sim.policies=[sans,pg_clns,swc_tcf,na]"
```

The same simulation in PowerShell (backtick continues a line, and the quotes swap):

```powershell
python . test_sim sim.graph.area=riomaior sim.graph.num_loc=100 sim.graph.n_days=10 sim.data_distribution=emp `
  sim.graph.load_dataset=data/simulator/datasets/riomaior100_emp_wsr30_N1_seed42.npz `
  sim.graph.distance_method=ogd "sim.graph.dm_filepath='gmaps_distmat_plastic[riomaior].csv'" `
  sim.graph.focus_graph=graphs_100V_1N_plastic.json "sim.policies=[sans,pg_clns,swc_tcf,na]"
```

```bash
# Train the Attention Model on the 20-bin CPTP sample dataset (REINFORCE, one epoch)
python . train envs@train.env=cptp train.batch_size=16 train.num_workers=0 'train.env.eval_graphs=[]' \
  train.env.curriculum_graphs.0.num_loc=20 train.env.curriculum_graphs.0.area=riomaior \
  train.env.curriculum_graphs.0.distance_method=file \
  'train.env.curriculum_graphs.0.dm_filepath="gmaps_distmat_plastic[riomaior].csv"' \
  train.env.curriculum_graphs.0.focus_graph=graphs_20V_1N_plastic.json \
  train.env.curriculum_graphs.0.focus_size=64 train.env.curriculum_graphs.0.n_samples=64 \
  train.env.curriculum_graphs.0.n_days=1 \
  train.env.curriculum_graphs.0.load_dataset=data/datasets/cptp/cptp20_emp_smoke_seed42.td \
  train.model_weights_path=weights/ckpt train.final_model_path=weights/am/epoch-1.pt
```

Each routing algorithm runs with the packing variants in its
`logic/configs/policies/routing/<category>/policy_<key>.yaml` (e.g.
`routing/meta_heuristics/policy_sans.yaml`); `sim.policies` picks the algorithms. The packing
strategies are in `logic/configs/policies/other/packing/pk_<key>.yaml`. The other scenarios of
the paper use `area=riomaior num_loc=170` and `area=figueiradafoz num_loc=350`
(`distance_method=file`, `dm_filepath="osm_distmat_plastic[figueiradafoz].csv"`), with the
matching dataset in `data/simulator/datasets/` and `data_distribution=gamma3` for Gamma-3.
`na` reads its model from `model_path` in `logic/configs/policies/routing/learning_algorithms/policy_na.yaml` (the Rio Maior
100 Empirical model by default). To simulate another scenario, edit that line to the scenario's
model, e.g. `model_weights/cptp350_figueiradafoz_plastic/emp/amgat0-il-rl`. The simulator reads the
policy files themselves, so a command-line override of `model_path` does not reach it.

Limit simulator parallelism on small machines with `sim.cpu_cores=4`; the default `0` starts one
worker per spare core. Results are written under
`outputs/simulations/<n>days/<area><n>_<waste>/<distribution>/<run_name>/`
(`sim.output_dir`, relative to the package root) as one `log_<policy>_<N>N.json` per policy
plus a real-time `.jsonl` stream, with the run's resolved configuration in its `hydra/` folder.
Ctrl+C stops a simulation within about a second (exit code 1); the interrupted run writes no
summary.

## Running the tests

The package carries the tests that pass in this tree (the excluded ones need code the package
leaves out; `logic/test/release_tests.excluded.txt` lists them with their first failure). The
setup script installs the test tools and unpacks the data; run the tests from the package root:

```bash
.venv/bin/python -m pytest logic/test        # Windows: .venv\Scripts\python -m pytest logic\test
```

The tests clean up only what they create; they do not touch existing `outputs/` or `model_weights/`.

## Notes

- Each run also writes Hydra's own log folder, `outputs/<date>/<time>/`, in the package root.
- The heuristic routing algorithms stop on an iteration budget (`max_iterations` in their policy
  files), so a run gives the same result on any machine and under any load; `time_limit` is only a
  600 s safety guard. `swc_tcf` keeps its solver time limit.
- `tracking.*` options other than the log directory do nothing (no MLflow, ZenML or profiling).
- Simulations need a dataset in `sim.graph.load_dataset` and training a `.td` file in each
  graph's `load_dataset`; the package does not generate data.
- Checkpoints and datasets that record the former problem names (`vrpp`, `cvrpp`, `mvptp`) are
  rejected; retrain or regenerate them.
- Training reads the `.td` file in each graph's `load_dataset`; without a validation dataset it
  validates on the first training instances. The paper's training datasets (1.7 GB) are not
  included.
