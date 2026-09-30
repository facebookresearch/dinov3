#!/bin/bash
# GRPO fine-tuning of FossilVL with an LLM judge as reward (geo/train_fossilvl_grpo.py).
# One torchrun per node, 4 processes: local rank 0 trains the captioning model, local ranks 1-3 each hold a
# replica of the judge. More nodes add both (use --nodes=N). Submit from the repository root:
#   SFT=/path/to/sft_output OUT=/path/to/rl_output sbatch geo/slurm/train_grpo.sbatch
# Re-submitting with the same OUT resumes from the last checkpoint.
#SBATCH --job-name=fossilvl-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one torchrun per node; it starts the 4 GPU processes
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/grpo-%j.out
##SBATCH --partition=tc

export HTTP_PROXY=http://$USER:$PASSWORD@inet-sys.petrobras.com.br:804
export HTTPS_PROXY=http://$USER:$PASSWORD@inet-sys.petrobras.com.br:804
export NO_PROXY="127.0.0.1, localhost, petrobras.com.br, petrobras.biz"


set -eo pipefail
. ~/.bashrc
conda activate cap
# The judge must already be in the HF cache (the reliability runs downloaded it). If the node has to download
# it, copy the HTTP(S)_PROXY lines from geo/eval/slurm/judge_reliability.sbatch and drop HF_HUB_OFFLINE.
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
set -u

SFT=${SFT:?set SFT to the SFT output folder (config.yaml + best_checkpoint.pt)}
OUT=${OUT:?set OUT to the RL output folder}
JUDGE=${JUDGE:-Qwen/Qwen2.5-32B-Instruct}   # lowest false-positive rate in the reliability study
LABELS=${LABELS:-geo/eval/todos.json}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
mkdir -p logs "$OUT"
echo "GRPO: sft=$SFT out=$OUT judge=$JUDGE nodes=$SLURM_NNODES master=$MASTER_ADDR"

srun torchrun \
  --nnodes="$SLURM_NNODES" --nproc_per_node=4 \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:29500" \
  geo/train_fossilvl_grpo.py \
    --model "$SFT" --ckpt best --labels "$LABELS" --out_dir "$OUT" --resume \
    --judge "$JUDGE" --policy_gpus_per_node 1 --judge_batch_size 32 \
    --batch_size 4 --num_generations 8 --epochs 1 \
    --lr 1e-6 --beta 0.04 --max_new_tokens 160 \
    --neg_ratio 1.0 --false_weight 1.0 \
    --save_every 50 --log_samples_every 10
