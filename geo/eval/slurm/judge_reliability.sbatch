#!/bin/bash
# Judge reliability on the ground-truth texts, one array task per judge model.
# Each array task runs one process per GPU (srun), and each process judges its own shard of the questions.
# Submit from the repository root (see geo/eval/slurm/README.md):
#   sbatch geo/eval/slurm/judge_reliability.sbatch
#
# Adjust the #SBATCH lines to your cluster (partition, account, GPU type, time).
#SBATCH --job-name=judge-rel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one process per GPU; keep equal to the number of GPUs below
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --array=0-7%4                # one task per entry of MODELS; %4 = at most 4 models at once
#SBATCH --output=logs/judge-rel-%A_%a.out
##SBATCH --partition=gpu
##SBATCH --account=my_account

set -euo pipefail

# model | extra flags | starting batch size (halved automatically on out-of-memory)
# Sizes assume 40-80 GB GPUs. For models that do not fit one GPU (70B+) use "--device_map auto" and
# submit that index with one process per node, e.g.:
#   sbatch --ntasks-per-node=1 --array=8 geo/eval/slurm/judge_reliability.sbatch
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|--load_in_4bit|32"    # same setup as the local 8 GB tests
  "microsoft/Phi-3.5-mini-instruct|--load_in_4bit|32"
  "Qwen/Qwen2.5-7B-Instruct||64"
  "Qwen/Qwen3-8B||64"
  "Qwen/Qwen2.5-14B-Instruct||32"
  "Qwen/Qwen3-14B||32"
  "Qwen/Qwen2.5-32B-Instruct||16"                 # ~65 GB in bf16: needs an 80 GB GPU, else add --load_in_4bit
  "Qwen/Qwen3-32B||16"
  "Qwen/Qwen2.5-72B-Instruct|--device_map auto|16"  # ~145 GB: spread over the node's GPUs
)

INPUT=${INPUT:-geo/eval/todos.json}
OUT_DIR=${OUT_DIR:-geo/eval/reliability}

# ---- environment: edit for your cluster ------------------------------------------------------------
# module load cuda/12.6
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate geo
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # put it on a shared filesystem
# export HF_HUB_OFFLINE=1                              # compute nodes without internet: pre-download first
# export HF_TOKEN=...                                  # only for gated models (Llama, Gemma)
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# ------------------------------------------------------------------------------------------------------

IFS='|' read -r MODEL FLAGS BATCH <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "array task $SLURM_ARRAY_TASK_ID: $MODEL flags='$FLAGS' batch=$BATCH on $SLURM_NTASKS processes ($(hostname))"
mkdir -p logs

# srun starts SLURM_NTASKS copies; each reads SLURM_PROCID/SLURM_NTASKS to pick its shard and
# SLURM_LOCALID to pick its GPU. Re-submitting the same index resumes where it stopped.
# shellcheck disable=SC2086
srun --kill-on-bad-exit=1 python geo/eval/judge_reliability.py run \
  --model "$MODEL" $FLAGS --batch_size "$BATCH" --input "$INPUT" --out_dir "$OUT_DIR"
