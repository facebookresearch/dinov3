#!/bin/bash
# Runs the whole judge reliability evaluation on Slurm, from the login node:
#   1. (optional) downloads the models into $HF_HOME
#   2. submits one array job for the models that fit one GPU (one replica per GPU)
#   3. submits one array job for the models spread over a node's GPUs (--device_map auto in MODELS)
#   4. submits the merge, which starts when both jobs have ended and writes $OUT_DIR/report.md
# The model list is MODELS in judge_reliability.sbatch. Re-running this script resumes unfinished models.
#
# Usage, from the repository root with the geo environment active:
#   bash geo/eval/slurm/run_all.sh [--download] [--models 0,2,8] [--dry-run]
#
# Environment variables:
#   INPUT, OUT_DIR      input json and output folder (default geo/eval/todos.json, geo/eval/reliability)
#   MAX_PARALLEL        models running at the same time (default 4)
#   MERGE_ARGS          extra sbatch options for the CPU-only merge job, e.g. "--partition=cpu"
#   SBATCH_PARTITION, SBATCH_ACCOUNT   read by sbatch itself, applied to every job
set -euo pipefail

SBATCH_FILE=geo/eval/slurm/judge_reliability.sbatch
DOWNLOAD=0
DRY=0
SELECT=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --download) DOWNLOAD=1 ;;
    --models) SELECT=$2; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 1 ;;
  esac
  shift
done

[[ -f $SBATCH_FILE ]] || { echo "run this script from the repository root"; exit 1; }
export INPUT=${INPUT:-geo/eval/todos.json}
export OUT_DIR=${OUT_DIR:-geo/eval/reliability}
MAX_PARALLEL=${MAX_PARALLEL:-4}
MERGE_ARGS=${MERGE_ARGS:-}
[[ -f $INPUT ]] || { echo "input not found: $INPUT"; exit 1; }
python -c "import importlib.util as u, sys; sys.exit(any(u.find_spec(m) is None for m in ('torch', 'transformers', 'sklearn')))" \
  || { echo "torch/transformers/scikit-learn missing: activate the geo environment first"; exit 1; }
if (( ! DRY )); then
  command -v sbatch > /dev/null || { echo "sbatch not found: run this on the cluster login node"; exit 1; }
fi

# model list from the sbatch file
eval "$(sed -n '/^MODELS=(/,/^)/p' "$SBATCH_FILE")"
if [[ -n $SELECT ]]; then
  INDICES=${SELECT//,/ }
else
  INDICES=$(seq 0 $(( ${#MODELS[@]} - 1 )))
fi
SINGLE=()
NODE=()
echo "models:"
for i in $INDICES; do
  [[ -n ${MODELS[$i]:-} ]] || { echo "no model at index $i (0-$(( ${#MODELS[@]} - 1 )))"; exit 1; }
  IFS='|' read -r model flags _ <<< "${MODELS[$i]}"
  if [[ $flags == *--device_map* ]]; then NODE+=("$i"); where="whole node"; else SINGLE+=("$i"); where="one GPU per replica"; fi
  printf '  %2s  %-36s %-22s %s\n' "$i" "$model" "$flags" "($where)"
done

join() { local IFS=,; echo "$*"; }
run() {  # prints the job id; in dry-run mode only shows the command
  if (( DRY )); then echo "[dry-run] sbatch $*" >&2; echo "DRY"; else sbatch --parsable "$@" | cut -d';' -f1; fi
}

if (( DOWNLOAD )); then
  names=()
  for i in $INDICES; do names+=("$(cut -d'|' -f1 <<< "${MODELS[$i]}")"); done
  echo "downloading ${#names[@]} models into ${HF_HOME:-~/.cache/huggingface}"
  if (( DRY )); then echo "[dry-run] python geo/eval/judge_reliability.py download ${names[*]}"
  else python geo/eval/judge_reliability.py download "${names[@]}"; fi
fi

mkdir -p logs "$OUT_DIR"
JOBS=()
if (( ${#SINGLE[@]} )); then
  JOBS+=("$(run --array="$(join "${SINGLE[@]}")%$MAX_PARALLEL" "$SBATCH_FILE")")
  echo "submitted single-GPU models (${SINGLE[*]}): job ${JOBS[-1]}"
fi
if (( ${#NODE[@]} )); then
  JOBS+=("$(run --ntasks-per-node=1 --array="$(join "${NODE[@]}")%$MAX_PARALLEL" "$SBATCH_FILE")")
  echo "submitted whole-node models (${NODE[*]}): job ${JOBS[-1]}"
fi

# afterany: merge even if some model failed; the report flags models with missing questions
DEPENDENCY=$(IFS=:; echo "${JOBS[*]}")
# shellcheck disable=SC2086
MERGE=$(run --dependency=afterany:"$DEPENDENCY" --job-name=judge-rel-merge --ntasks=1 --cpus-per-task=2 \
  --mem=16G --time=00:30:00 --output=logs/judge-rel-merge-%j.out $MERGE_ARGS \
  --wrap "python geo/eval/judge_reliability.py merge --out_dir $OUT_DIR")
echo "submitted merge: job $MERGE (starts after ${JOBS[*]})"

cat << EOF

follow:  squeue -u \$USER
logs:    logs/judge-rel-<job>_<index>.out, logs/judge-rel-merge-$MERGE.out
report:  $OUT_DIR/report.md (after the merge job)
If a model runs out of time, run this script again with --models <index>. Finished questions are skipped.
EOF
