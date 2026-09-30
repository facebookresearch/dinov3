# Judge reliability on a Slurm cluster

`geo/eval/judge_reliability.py` measures how reliable each LLM judge of `eval_labels_llm.py` is. It uses the
ground-truth texts in `geo/eval/todos.json`, so no predictions are needed. A judge is shown each reference
text with a claim and must answer:

- **Yes** for the sample's own labels (positives), which gives the **TPR**;
- **No** for labels the sample does not have, which gives the **FPR**. These negatives are
  - `absent`: a label borrowed from another sample;
  - `role_swap`: one of the sample's main constituents claimed as a secondary one, or the reverse.

The 17,948 samples contain 4,844 distinct (text, labels) pairs. Each pair is judged once and weighted by how
often it occurs, which gives about 38k questions per model.

## 1. Setup (once, on the login node)

```bash
git clone <repo> && cd dinov3          # or copy the repo, including geo/eval/todos.json
conda create -n geo python=3.12 -y && conda activate geo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # match the cluster's CUDA
pip install -r geo/requirements.txt
mkdir -p logs                          # Slurm writes the job logs here and does not create the folder
```

Set `HF_HOME` to a shared filesystem with enough space (about 560 GB for every model in the list), for
example in `~/.bashrc`:

```bash
export HF_HOME=/scratch/$USER/hf
```

**Llama models (indices 9-11) are gated.** Before running them:
1. Accept the license on each model page while logged in to huggingface.co:
   `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.3-70B-Instruct`.
   Approval usually takes minutes to a few hours.
2. On the login node, with the same `HF_HOME` the jobs use, run `hf auth login` (older versions:
   `huggingface-cli login`) and paste a read token. The token is saved in `$HF_HOME` and the jobs pick it
   up from there. `run_all.sh` refuses to submit Llama models when no token is found.

Meta has no 14B or 32B text Llama, so the Llama judges are 3B (compare with Phi-3.5-mini), 8B (compare with
Qwen2.5-7B and Qwen3-8B) and 70B (compare with Qwen2.5-72B). The 70B needs about 141 GB in bf16, so it is
spread over a node's GPUs, like the 72B.

If the compute nodes have no internet access, download the models on the login node first, and uncomment
`export HF_HUB_OFFLINE=1` in the sbatch file:

```bash
python geo/eval/judge_reliability.py download \
  Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct Qwen/Qwen3-8B Qwen/Qwen2.5-14B-Instruct \
  Qwen/Qwen3-14B Qwen/Qwen2.5-32B-Instruct Qwen/Qwen3-32B Qwen/Qwen2.5-72B-Instruct \
  meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.3-70B-Instruct
```

## 2. Edit `geo/eval/slurm/judge_reliability.sbatch`

- `#SBATCH` lines: partition, account, time, and the number of GPUs per node. Keep `--ntasks-per-node`
  equal to `--gres=gpu:N`.
- The environment block: `module load ...` and `conda activate geo`.
- `MODELS`: the judges to compare. Each array index is one model.

## 3. Submit (from the repository root)

**All in one command.** `run_all.sh` submits every model in `MODELS` with the right layout, then a merge
job that writes the report when they finish:

```bash
bash geo/eval/slurm/run_all.sh --download        # --download only if compute nodes have no internet
bash geo/eval/slurm/run_all.sh --models 0,3,8    # a subset, or to resume unfinished models
bash geo/eval/slurm/run_all.sh --dry-run         # show the sbatch commands without submitting
MERGE_ARGS="--partition=cpu" bash geo/eval/slurm/run_all.sh   # if the default partition needs GPUs
```

**Manual submission**, if you prefer:

```bash
# models 0-7: each model is replicated on every GPU of one node, and each replica judges one shard
JOB=$(sbatch --parsable geo/eval/slurm/judge_reliability.sbatch)

# models 8 and 11 (72B, 70B) do not fit one GPU: one process per node, the model spread over its GPUs
sbatch --ntasks-per-node=1 --array=8,11 geo/eval/slurm/judge_reliability.sbatch

# only some models
sbatch --array=2,3 geo/eval/slurm/judge_reliability.sbatch

# more GPUs for one model: 2 nodes x 4 GPUs = 8 shards
sbatch --nodes=2 --array=4 geo/eval/slurm/judge_reliability.sbatch
```

Watch the jobs with `squeue -u $USER` and `tail -f logs/judge-rel-<jobid>_<index>.out`.

Jobs **resume**: if a job times out or is preempted, submit the same array index again. Each process appends
results to `geo/eval/reliability/<model>/part-*.jsonl`, and questions already judged are skipped, even when
the new job uses a different number of GPUs. Out-of-memory errors halve the batch size automatically.

Rough cost is about 38k questions per model. The local RTX 4060 managed about 7 questions/s with
Qwen2.5-7B in 4-bit. A single A100 or H100 in bf16 should be at least 10× faster, so the 7-14B models
take minutes on a 4-GPU node. The 72B spread over GPUs is the slowest, around an hour.

## 4. Merge and read the report

After the jobs finish (the merge runs on the CPU and takes seconds; the login node is fine):

```bash
python geo/eval/judge_reliability.py merge --out_dir geo/eval/reliability
# or run it automatically when the array finishes:
sbatch --dependency=afterany:$JOB --wrap "python geo/eval/judge_reliability.py merge --out_dir geo/eval/reliability"
```

The merge can also run on partial results. It warns about unfinished models and computes metrics on what
has been judged so far.

Outputs in `geo/eval/reliability/`:

| File | Content |
|---|---|
| `report.md` | tables: TPR, FPR (overall, absent, role swap), precision, balanced accuracy, AUC, best threshold, per category, agreement between judges |
| `summary.json` | the same numbers, machine-readable (`unique` = unweighted by duplicate count) |
| `hard_cases.jsonl` | questions most judges get wrong. These may be errors in the references or labels, so read them before blaming the judges |
| `questions.json` | the question set (fixed by `--seed`, `--neg_ratio`, `--ignore_role`) |

**How to read it:**
- **Choose the judge with the highest balanced accuracy and AUC.**
- **Look at FPR role swap:** a judge that accepts role swaps cannot score main vs secondary constituents.
  In that case, evaluate with `--ignore_role` in `eval_labels_llm.py`, or use a better judge.
- **If the best threshold is far from 0.5,** the judge is miscalibrated. Pass that threshold with
  `--threshold` to `eval_labels_llm.py`.

## Options

- `--neg_ratio 2`: twice as many `absent` negatives.
- `--seed N`: a different sample of negatives.
- `--ignore_role`: loose claims, where the label only has to be mentioned. Role swaps are then excluded.

Changing any of these requires a new `--out_dir` (set `OUT_DIR=...` when submitting:
`OUT_DIR=geo/eval/reliability_loose sbatch ...`, and add the flag to the `srun` line).

Without Slurm, a multi-GPU machine works through torchrun, which sets `RANK` and `WORLD_SIZE`:
`torchrun --nproc_per_node 4 geo/eval/judge_reliability.py run --model ...`
