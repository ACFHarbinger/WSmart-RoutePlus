#!/usr/bin/env bash
# Acceptance smoke for the minimal-ptp export (run from the package root, or point PY at a wrapper for the executable).
# train (tiny AM, empirical fills; PTP and MVPTP, from pre-generated .td datasets) -> test_sim of the four route constructors.
# Fails unless every policy log routes (km > 0) on some day.
#
# Needs data/simulator and two small training sets, made with the main repository (the package cannot generate data):
#   python main.py gen_data data.problem=<ptp|mvptp> data.dataset_type=train 'data.data_distributions=[emp]' \
#     data.filename=<package>/data/datasets/<problem>/<problem>20_emp_smoke_seed42.td \
#     'data.graphs=[{num_loc: 20, n_samples: 64, area: riomaior, waste_type: plastic, focus_size: 64, focus_graph: graphs_20V_1N_plastic.json, start_day: 0, n_days: 1, vertex_method: mmn, distance_method: file, dm_filepath: "gmaps_distmat_plastic[riomaior].csv"}]'
set -euo pipefail
PY="${PY:-.venv/bin/python}"
OUT="${1:-$(mktemp -d)}"
DAYS="${DAYS:-10}"
# SIM_CORES caps simulator worker processes (each loads torch/CUDA/Gurobi); the yaml default 0 means one per spare core.
mkdir -p "$OUT"
STAMP="$OUT/.stamp"; touch "$STAMP"
echo "[smoke] output: $OUT"

DS="data/simulator/datasets/riomaior100_emp_wsr30_N1_seed42.npz"; test -f "$DS"; echo "[smoke] dataset: $DS"
TD_PTP="${TD_PTP:-data/datasets/ptp/ptp20_emp_smoke_seed42.td}"; TD_MVPTP="${TD_MVPTP:-data/datasets/mvptp/mvptp20_emp_smoke_seed42.td}"
test -f "$TD_PTP" && test -f "$TD_MVPTP"

train() {  # <env> <n_samples> <dataset> <tag>
  "$PY" . train envs@train.env="$1" train.env.name="$1" \
    'train.env.curriculum_graphs=[{num_loc: 20, n_samples: '"$2"', area: riomaior, waste_type: plastic, distance_method: file, dm_filepath: "gmaps_distmat_plastic[riomaior].csv", focus_size: '"$2"', focus_graph: graphs_20V_1N_plastic.json, n_days: 1, load_dataset: '"$3"'}]' \
    'train.env.eval_graphs=[]' train.batch_size=16 train.num_workers=0 train.policy.model.encoder.embed_dim=32 \
    train.policy.model.encoder.hidden_dim=64 train.policy.model.encoder.n_layers=1 train.policy.model.encoder.n_heads=2 \
    train.policy.model.decoder.n_heads=2 train.policy.mandatory_selection=null train.data_distribution=emp \
    train.train_time=false train.model_weights_path="$OUT/ckpt$4" train.final_model_path="$OUT/am$4/epoch-1.pt" \
    > "$OUT/train$4.log" 2>&1 || { tail -20 "$OUT/train$4.log"; exit 1; }
  test -f "$OUT/am$4/epoch-1.pt"
}
train ptp 64 "$TD_PTP" "";         echo "[smoke] train ok"
train mvptp 32 "$TD_MVPTP" _mvptp; echo "[smoke] train (mvptp) ok"

"$PY" . test_sim sim.graph.area=riomaior sim.graph.num_loc=100 sim.graph.n_days="$DAYS" \
  'sim.graph.dm_filepath="gmaps_distmat_plastic[riomaior].csv"' sim.graph.focus_graph=graphs_100V_1N_plastic.json \
  sim.graph.load_dataset="$DS" sim.data_distribution=emp sim.cpu_cores="${SIM_CORES:-4}" p.na.na.amgat.0.model_path="$OUT/am" \
  > "$OUT/sim.log" 2>&1 || { tail -30 "$OUT/sim.log"; exit 1; }
"${CHECK_PY:-$PY}" - "$STAMP" "$DAYS" <<'PYEOF'
import glob, json, os, sys
stamp, days = os.path.getmtime(sys.argv[1]), sys.argv[2]
logs = [f for f in glob.glob(f"assets/output/{days}days/**/log_*.json", recursive=True) if os.path.getmtime(f) > stamp]
pols = ["_na_", "pg_clns", "sans", "swc_tcf"]
bad = []
for f in sorted(logs):
    d = json.load(open(f))["daily"]["0"]
    km = max(d["km"])
    print(f"[smoke]   {os.path.basename(f):60s} max km/day {km:8.2f}")
    if km <= 0: bad.append(f)
missing = [p for p in pols if not any(p in os.path.basename(f) for f in logs)]
assert not missing, f"no log for: {missing}"
assert not bad, f"logs that never route: {bad}"
print(f"[smoke] test_sim ok: {len(logs)} logs, all route")
PYEOF
echo "[smoke] PASS"
