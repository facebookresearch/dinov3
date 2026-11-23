# Metadata-Guided DINOv3

This package adds metadata-guided learning heads to DINOv3 SSL training. Heads
attach to the student's CLS pre-head embedding and learn to predict (or, with
GRL, be adversarially debiased from) per-sample metadata fields such as country,
sub-region, year, sensor angle, etc.

## Contents

- [Where everything lives](#where-everything-lives)
- [Dataset contract](#dataset-contract)
- [Enabling guides](#enabling-guides)
- [Gradient-norm normalization](#gradient-norm-normalization)
- [Lambda schedule](#lambda-schedule)
- [Optional: SIGReg regularization](#optional-sigreg-regularization)
- [Two-stage finetune (freeze + LR ramp)](#two-stage-finetune-freeze--lr-ramp)
- [Reference recipes](#reference-recipes)
- [Preparing data](#preparing-data)
- [Pretraining commands](#pretraining-commands)
- [Plugging in your own dataset](#plugging-in-your-own-dataset)
- [Adapting to a different channels combination](#adapting-to-a-different-channels-combination)
- [Config field reference](#config-field-reference)
- [Evaluating a trained checkpoint](#evaluating-a-trained-checkpoint)

Head types: **classification** (`Classifier`, MLP + CE/BCE), **regression**
(`Regressor`, MLP + MSE with optional bounded activation and target
normalization), and **prototypical contrastive** (`PrototypicalContrastiveHead`,
ProtoNCE against EMA centroids; buffers only, no learnable params). Every head
is wrapped in a `GradientScalingLayer` (GRL) so the same head serves as
auxiliary predictor (`lambda > 0`) or adversarial debiaser (sign-flipped).

## Where everything lives

| Component | Location |
|---|---|
| Meta-arch | `dinov3/train/guided_ssl_meta_arch.py` (`GuidedSSLMetaArch`) |
| Head modules + losses | `dinov3/train/metadata_utils/` (this package) |
| Collate (metadata path) | `dinov3/data/collate.py` (`_collate_metadata`, `collate_data_and_cast`) |
| Loader (target_transform) | `dinov3/data/loaders.py` and `dinov3/train/train.py` |
| FMoW reference dataset | `dinov3/data/datasets/fmow.py` (`FMoW`, `_Metadata`) |
| Config schema | `dinov3/configs/ssl_default_config.yaml` — `guide:` and `optim.grad_norm_normalization` |
| Two-stage finetune knobs | `dinov3/train/param_groups.py`, `dinov3/train/ssl_meta_arch.py` (`is_student_frozen`), `dinov3/train/train.py` (`apply_optim_scheduler`) |
| SIGReg loss (optional) | `dinov3/loss/sigreg_loss.py` (`DistributedSIGReg`) |
| Arch dispatch | `dinov3/train/train.py` — `MODEL.META_ARCHITECTURE: GuidedSSLMetaArch` |
| Reference config | `dinov3/configs/train/vitl16_fmow_guided.yaml` |

## Dataset contract

Datasets feeding `GuidedSSLMetaArch` **must** return samples shaped:

```python
(image, (label, metadata))
```

where `metadata` is a `@dataclass` whose field names match each guide's `name`.
`GuidedSSLMetaArch` looks up each guide's tensor via
`getattr(metadata, guide.name)`, so a typo or missing field is an error.

The collate fn (`dinov3/data/collate.py:_collate_metadata`) batches each field
into a tensor (numeric fields) or a Python list (string / mixed fields).

If `guide.enabled=True` but the batch has no `metadata` key,
`GuidedSSLMetaArch.forward_backward` raises a `RuntimeError`.

`dinov3/data/datasets/fmow.py` is the concrete example. `FMoW` emits a
`_Metadata` dataclass with fields like `country` (int), `sub_region` (int),
`year` (int), `coordinates` (tuple[float, float]), `gsd` (float), etc. Any
field can drive a guide head by naming the guide after it.

## Enabling guides

Set `MODEL.META_ARCHITECTURE: GuidedSSLMetaArch` and fill out `guide`:

```yaml
MODEL:
  META_ARCHITECTURE: GuidedSSLMetaArch

guide:
  enabled: true
  lambda_schedule:
    type: sigmoid          # sigmoid | linear | constant
    warmup_iterations: 1000
  guides:
    - name: sub_region                # must match a field on the metadata dataclass
      enabled: true
      method: classification          # classification | regression | prototypical
      hidden_dim: [512, 512, 256]
      dropout: 0.5
      loss_weight: 0.1
      n_outputs: 18
      grl: false
    - name: year
      enabled: true
      method: prototypical
      loss_weight: 0.1
      n_outputs: 16
      proto_temperature: 0.07
      proto_centroid_momentum: 0.999
    - name: gsd
      enabled: true
      method: regression
      hidden_dim: [256, 128]
      dropout: 0.1
      loss_weight: 0.05
      n_outputs: 1
      output_activation: sigmoid
      target_normalization:
        output_min: [0.2]
        output_max: [1.8]
```

Adversarial debiasing: set `grl: true`; lambda flips sign during backward and
`loss_weight` is divided by 10 to keep the adversarial signal from dominating
SSL.

## Gradient-norm normalization

```yaml
optim:
  grad_norm_normalization: true
```

When true, `GuidedSSLMetaArch` rescales the **guide losses against each other**
so that no single guide dominates the others — gradient magnitudes are
equalized between guides on the student's CLS pre-head. The SSL loss is not
touched by this mechanism. Computed via `torch.autograd.grad` on the
intermediate CLS activation (not on any sharded parameter), so it works under
FSDP without a second full backward pass. The per-guide `loss_weight` is
preserved as a multiplier on top.

## Lambda schedule

`compute_lambda` ramps the gradient scale from 0 to 1:

- `sigmoid` — DANN-style sigmoid (default).
- `linear` — straight ramp.
- `constant` — always 1.0.

`warmup_iterations` keeps lambda at 0 initially so the SSL signal stabilizes
before guide heads influence the backbone.

## Optional: SIGReg regularization

The reference recipe pairs metadata heads with SIGReg on the student DINO-head
bottleneck:

```yaml
sigreg:
  enabled: true
  mode: bottleneck
  loss_weight: 0.05
  koleo_too: false          # replaces KoLeo when false; runs alongside it when true
```

With `sigreg.enabled`, the base `SSLMetaArch` swaps KoLeo for
`DistributedSIGReg` (`dinov3/loss/sigreg_loss.py`) on the pre-normalization
bottleneck output of `student.dino_head`.

## Two-stage finetune (freeze + LR ramp)

The reference recipe warm-starts from a pretrained ViT-L teacher checkpoint:

- **Stage 1** (`0 .. freeze_student_iterations`): backbone frozen; SSL heads
  (`dino_head`, `ibot_head`, SIGReg bottleneck) and guide heads adapt.
  `patch_embed`, `cls_token`, `mask_token` are frozen by default — flip the
  matching `freeze_*` knob to keep any of them trainable.
- **Stage 2**: backbone unfreezes with a `backbone_warmup_after_freeze`-iter
  linear LR ramp.

Schedules v2 is required (the freeze mechanism is iteration-indexed).

> **Stage-1 length.** `10000` is a safe default but heads typically converge
> sooner; in compute-constrained runs drop it to ~1000.

```yaml
student:
  resume_from_teacher_chkpt: /path/to/pretrained_teacher.pth

schedules:
  lr:
    start: 0.0
    peak: 5.0e-05
    end: 5.0e-05
    warmup_iterations: 3000
    freeze_last_layer_iterations: 450     # DINO last-projection-layer freeze (independent of stage-1)
    cosine_iterations: 0
    freeze_student_iterations: 10000      # stage-1 length
    backbone_warmup_after_freeze: 1000    # stage-2 LR ramp (iterations)
    freeze_patch_embed: false
    freeze_cls_token: true
    freeze_mask_token: true
    unfreeze_last_n_blocks: 0             # >0: only last N blocks ever unfreeze
```

Each schedule field accepts an `_epochs` or `_iterations` form; when both are
present `_iterations` wins. Every freeze knob defaults to no-op when omitted,
so configs without `schedules.lr` behave as before.

## Reference recipes

Two runnable configs ship with the port:

- **FMoW** — `dinov3/configs/train/vitl16_fmow_guided.yaml`: 3-channel
  satellite imagery, two-guide recipe (sub_region auxiliary + year
  adversarial), SIGReg on the bottleneck, gradient-norm normalization,
  two-stage finetune from a pretrained ViT-L teacher.
- **HPA-WholeHR** — `dinov3/configs/train/vitl16_hpa_guided.yaml`:
  4-channel fluorescence imagery, single antibody (`guiding_labels`)
  prototypical guide, the same SIGReg + two-stage finetune machinery, and
  the HPA-specific bio augmentation pipeline (`train.cell_augmentation_type:
  hpa`, vertical flips on, gaussian blur off).

Adapt `output_dir`, `dataset_path`, and `student.resume_from_teacher_chkpt`
for your environment.

## Preparing data

### FMoW

`FMoW` reads from the WILDS FMoW bundle. Either install
[wilds](https://github.com/p-lambda/wilds) and let
`FMoWDataset(download=True)` fetch it, or pull the tarball directly from
CodaLab:

```bash
wget https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/ \
    -O fmow_v1.1.tar.gz
tar -xzf fmow_v1.1.tar.gz -C <root>/
```

Unpacking lands the three artifacts the dataset reads directly under `<root>`:

```
<root>/
  rgb_metadata_v2.csv
  country_code_mapping.csv
  images/rgb_img_<row_index>.png
```

Point the run at it with `FMoW:split=TRAIN:root=<root>` — no preprocessing
step.

### HPA-WholeHR

`HPAWholeHR` reads two CSVs from `root` (`hpa_whole_external_hr.csv`,
`hpa_whole_kaggle.csv`) and the matching JPGs from
`HPAexternal/jpg_{res}x{res}_4channels/` and
`HPAImageKaggle/jpg_{res}_{train,test}/`. The on-disk JPGs are 4 channels
packed Fortran-style along the image width; `PackedXChannelImageDecoder`
(in `decoders.py`) handles the reshape transparently. Target on-disk layout
under `<root>`:

```
<root>/
  hpa_whole_external_hr.csv
  hpa_whole_kaggle.csv
  HPAexternal/
    jpg_768x768_4channels/<id>.jpg     # packed 4-ch grayscale, shape (768, 768*4)
  HPAImageKaggle/
    jpg_768_train/<uuid>.jpg
    jpg_768_test/<uuid>.jpg
```

The four helpers shipped next to the dataset rebuild every file under
`<root>` from publicly-available sources.

**Step 0 — Build the two HPAWholeHR manifest CSVs from EBI biostudies
upstreams:**

```bash
wget https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/443/S-BIAD2443/Files/HPA/Master_fovHPA_512.csv
wget https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/443/S-BIAD2443/Files/HPA/whole_images_512_train.csv
python -m dinov3.data.datasets.hpa_whole_hr_build_csv \
    --master-csv <PATH/TO/Master_fovHPA_512.csv> \
    --train-csv  <PATH/TO/whole_images_512_train.csv> \
    --output-dir <root>/
```

**Step 1a — External: download per-channel JPGs from
images.proteinatlas.org:**

```bash
python -m dinov3.data.datasets.hpa_whole_hr_download \
    --csv <root>/hpa_whole_external_hr.csv \
    --output-dir <PATH/TO/external_per_channel_jpgs/> \
    --start 0 --stop 1000000
```

**Step 2a — External: pack the four per-channel JPGs into one
width-concatenated 4-channel JPG:**

```bash
python -m dinov3.data.datasets.hpa_whole_hr_pack_channels \
    --input-dir <PATH/TO/external_per_channel_jpgs/> \
    --output-dir <root>/HPAexternal/jpg_768x768_4channels/ \
    [--use-gpu] [--num-workers 16]
```

**Step 1b — Kaggle: download and extract the two full-size 7z bundles:**

```bash
wget https://storage.googleapis.com/kaggle-human-protein-atlas/train_full_size.7z
wget https://storage.googleapis.com/kaggle-human-protein-atlas/test_full_size.7z
7z x train_full_size.7z -o<PATH/TO/kaggle_train_tifs/>
7z x test_full_size.7z  -o<PATH/TO/kaggle_test_tifs/>
```

**Step 2b — Kaggle: pack the per-channel TIFs (same script, `--extension
tif`):**

```bash
python -m dinov3.data.datasets.hpa_whole_hr_pack_channels \
    --input-dir <PATH/TO/kaggle_train_tifs/> \
    --output-dir <root>/HPAImageKaggle/jpg_768_train/ \
    --extension tif [--use-gpu] [--num-workers 16]
python -m dinov3.data.datasets.hpa_whole_hr_pack_channels \
    --input-dir <PATH/TO/kaggle_test_tifs/> \
    --output-dir <root>/HPAImageKaggle/jpg_768_test/ \
    --extension tif [--use-gpu] [--num-workers 16]
```

Point the dataset at `<root>` with
`HPAWholeHR:split=TRAIN_SSL:root=<root>`. Repeat steps 2a/2b with a
different `--target-size` if you need a different on-disk resolution; the
dataset's `EXTERNAL_HR_IMAGE_RELDIR` / `KAGGLE_IMAGE_RELDIR` constants
encode 768.

Splits used by the HPA recipe:

- Pretrain: `HPAWholeHR:split=TRAIN_SSL:with_metadata=true` — iterates the
  on-disk JPGs and joins metadata by filename.
- Probe train / val / test:
  `HPAWholeHR:split={BAL_TRAIN,VAL,TEST}:with_metadata=false` — `BAL_TRAIN`
  is the class-balanced ProteinLocation split, `VAL` is the labeled
  validation split, and `TEST` is the Kaggle held-out split with no public
  labels (predictions are saved for external submission; the in-eval
  `MEAN_PER_CLASS_MULTILABEL_F1` on `TEST` is not directly meaningful).
  `with_metadata=false` is required at eval time because the
  attention-pool collate path is label-only.

## Pretraining commands

### FMoW

Single node, 8 GPUs:

```bash
PYTHONPATH=${PWD} torchrun --standalone --nproc_per_node=8 \
    dinov3/train/train.py \
    --config-file dinov3/configs/train/vitl16_fmow_guided.yaml \
    --output-dir /path/to/output \
    train.dataset_path=FMoW:split=TRAIN \
    student.resume_from_teacher_chkpt=/path/to/pretrained_teacher.pth
```

Multi-node SLURM via the bundled submitit wrapper (4 nodes × 8 GPUs = 32,
matching the source recipe):

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
    --nodes 4 --ngpus 8 \
    --config-file dinov3/configs/train/vitl16_fmow_guided.yaml \
    --output-dir /path/to/output \
    train.dataset_path=FMoW:split=TRAIN \
    student.resume_from_teacher_chkpt=/path/to/pretrained_teacher.pth
```

### HPA-WholeHR

The recipe starts from a 4-channel-expanded ViT-L teacher — see
[Adapting to a different channels combination](#adapting-to-a-different-channels-combination)
for the one-shot `expand_patch_embed` command.

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
    --nodes 4 --ngpus 8 \
    --config-file dinov3/configs/train/vitl16_hpa_guided.yaml \
    --output-dir /path/to/output \
    train.dataset_path=HPAWholeHR:split=TRAIN_SSL:root=<root>:with_metadata=true \
    student.resume_from_teacher_chkpt=/path/to/dinov3_vitl16_pretrain_lvd1689m_teacher_4ch.pth
```

## Plugging in your own dataset

`fmow.py` is the worked example — follow these steps:

1. **Subclass `ExtendedVisionDataset`** (`dinov3/data/datasets/extended.py`).
   Implement `get_image_data(index) -> bytes` and `__len__`.
2. **Declare a `_Metadata` dataclass** whose field names match every
   `guide.guides[*].name`. Numeric fields (int / float / tuple of floats) batch
   into tensors; string or mixed fields batch into Python lists.
3. **Implement `get_target(index) -> (label, _Metadata)`**. The loader's
   wrapper transform preserves the `(label, metadata)` tuple when
   `guide.enabled` is true.
4. **Accept `with_metadata: bool = True` in `__init__`** so the dataset can
   downgrade to `(image, label)` for baseline-SSL ablations. The
   `dataset_path` parser forwards `:key=value` kwargs to the constructor
   (e.g. `MyDataset:split=TRAIN:with_metadata=true`).
5. **Register** the dataset in `dinov3/data/loaders.py:_parse_dataset_str`.
   Add a `Split` enum if you want WILDS-style splits.
6. **Re-export** from `dinov3/data/datasets/__init__.py`.

Sanity check:

```python
ds = MyDataset(root=..., split=..., with_metadata=True)
image, (label, meta) = ds[0]
assert hasattr(meta, "sub_region") and hasattr(meta, "year")
```

If a guide is enabled but the named field is missing on the dataclass,
`GuidedSSLMetaArch.forward_backward` raises `AttributeError` on the first
iteration.

## Adapting to a different channels combination

To finetune on non-RGB inputs (e.g. 4-channel HPA stains) from a 3-channel
DINOv3 checkpoint, expand `patch_embed.proj.weight` first:

```bash
python -m dinov3.utils.expand_patch_embed \
    --input  <PATH/TO/dinov3_vitl16_pretrain_lvd1689m_teacher.pth> \
    --output <PATH/TO/dinov3_vitl16_pretrain_lvd1689m_teacher_4ch.pth> \
    --in-chans 4
```

The script fills new channels with the mean of the original 3 RGB filters and
multiplies the whole tensor by `3 / N` so per-patch activation magnitudes are
preserved at initialization. `--no-normalize` skips the rescale; pass
`--checkpoint-key ''` if the file is a raw state dict.

Then point the run at the expanded checkpoint and bump `in_chans`:

```yaml
student:
  in_chans: 4
  resume_from_teacher_chkpt: /path/to/dinov3_vitl16_pretrain_lvd1689m_teacher_4ch.pth
teacher:
  in_chans: 4
```

Student/teacher `in_chans` must match the expanded checkpoint and the channel
count emitted by your dataset.

## Config field reference

Runnable example:
[`dinov3/configs/train/vitl16_fmow_guided.yaml`](../../configs/train/vitl16_fmow_guided.yaml).

### `guide:`

| Field | Meaning |
|---|---|
| `enabled` | Master switch. When false, `GuidedSSLMetaArch` behaves as baseline `SSLMetaArch`. |
| `lambda_schedule.type` | `sigmoid` (DANN-style), `linear`, or `constant`. |
| `lambda_schedule.warmup_iterations` | Iterations at λ=0 before the ramp. |
| `guides[]` | List of guide entries (disabled entries are ignored). |

### `guide.guides[]` (each entry)

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | Must equal a field name on the dataset's metadata dataclass. |
| `enabled` | yes | Per-guide switch. |
| `method` | yes | `classification`, `regression`, or `prototypical`. |
| `n_outputs` | yes | Classes (classification / prototypical) or regression output dim. |
| `loss_weight` | yes | Multiplier on the guide loss; preserved by grad-norm normalization. |
| `grl` | no | If true, gradient is reversed → adversarial debiasing. |
| `grl_space` | no | `embedding` (default) or `prototype` — where the GRL hook attaches. |
| `hidden_dim` | classif / regr | List of MLP hidden widths. |
| `dropout` | classif / regr | Dropout p between MLP layers. |
| `use_bce` | classif | BCE-with-logits instead of CE (multi-label). |
| `output_activation`, `target_normalization` | regression | Bounded activation + target rescale. |
| `proto_temperature`, `proto_centroid_momentum`, `proto_phi_min` | prototypical | ProtoNCE temperature, EMA centroid momentum, minimum cluster mass. |

### `optim.grad_norm_normalization`

When true, equalize the **unweighted guide gradient magnitudes** on the student
CLS pre-head, then apply each guide's effective `loss_weight`. Guides with zero
weight do not affect the normalization target. The SSL loss is unaffected.
The norms are computed via `torch.autograd.grad` on an intermediate activation.

### `sigreg:` (optional)

| Field | Meaning |
|---|---|
| `enabled` | Swap KoLeo for `DistributedSIGReg` on the DINO-head bottleneck. |
| `mode` | `bottleneck`. |
| `loss_weight` | Scalar. |
| `koleo_too` | If true, keep KoLeo alongside SIGReg; if false (default), SIGReg replaces it. |

### `schedules.lr.*` (two-stage finetune)

See [Two-stage finetune](#two-stage-finetune-freeze--lr-ramp).
`_iterations` forms win over `_epochs` equivalents; freeze knobs
(`freeze_student_iterations`, `backbone_warmup_after_freeze`, `freeze_*`
gates, `unfreeze_last_n_blocks`) are no-ops when omitted.

## Evaluating a trained checkpoint

Training periodically dumps a teacher snapshot under
`<output_dir>/eval/<iter>/teacher_checkpoint.pth` (single file) or
`<output_dir>/eval/<iter>/sharded_teacher_checkpoint/` (DCP shards). The
`do_test` codepath ends there — eval orchestration is manual.

`model.pretrained_weights` accepts the single-file `teacher_checkpoint.pth`
or the sharded `sharded_teacher_checkpoint/` directory. For a quick
single-node run, swap `python -m dinov3.run.submit` for
`torchrun --standalone --nproc_per_node=8`.

### Attention pooling on FMoW

Two configs ship with the port —
`dinov3/eval/configs/benchmark-fmow-iid-apool.yaml` (in-distribution: VAL_ID /
TEST_ID, year < 2013) and `benchmark-fmow-ood-apool.yaml` (temporal OOD:
VAL_OOD / TEST_OOD). Both train a 4-block attention-pooling head at
crop 224 / resize 256, sweep 9 learning rates from `1e-4` to `1.0` over 5
epochs of 300 iters, and report test accuracy at the LR with the best
validation `WORST_GROUP_ACCURACY` — minimum per-region accuracy across the 6
FMoW continents (`Africa`, `Americas`, `Asia`, `Europe`, `Oceania`,
`Unknown`). The metric is wired off `_Metadata.region`; to reuse it elsewhere,
mirror the `_region_collate` + `FMOW_REGION_NAMES` branch in
`dinov3/eval/linear.py`.

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/attention_pooling.py \
    --config-file dinov3/eval/configs/benchmark-fmow-iid-apool.yaml \
    --output-dir <PATH/TO/EVAL/OUTPUT> \
    model.config_file=<TRAIN/OUTPUT>/config.yaml \
    model.pretrained_weights=<TRAIN/OUTPUT>/eval/<ITER>/teacher_checkpoint.pth
```

### Attention pooling on HPA-WholeHR

`dinov3/eval/configs/benchmark-hpawholehr-apool.yaml` trains a 4-block
attention-pooling probe at crop 512 on `BAL_TRAIN`, sweeps 6 learning rates
over 30 epochs, picks the best by validation `MEAN_PER_CLASS_MULTILABEL_F1`,
and runs the Kaggle `TEST` split. The HPA-specific knobs:

- `train.cell_augmentation_type: hpa` (in the pretraining config) — swaps
  the SSL color jitter for `RandomContrastProteinChannel`,
  `RandomRemoveChannelExceptProtein`, `RandomContrast`, `RandomBrightness`,
  and replaces the RGB normalize with per-image per-channel
  `SelfNormalizeNoDiv`. Default `null` leaves the ImageNet-style pipeline
  unchanged.
- `transform.use_bio_transform: true` (in the eval `TransformConfig`) —
  eval pipeline becomes `Div255 → optional Resize → CenterCrop →
  SelfNormalizeNoDiv`; the matching train pipeline (used when training a
  probe) is `Div255 → optional Resize → RandomCrop → flips →
  SelfNormalizeNoDiv`.

4-channel mechanics (`student.in_chans` / `teacher.in_chans: 4` + the
expanded checkpoint) are covered in
[Adapting to a different channels combination](#adapting-to-a-different-channels-combination).

```bash
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/attention_pooling.py \
    --config-file dinov3/eval/configs/benchmark-hpawholehr-apool.yaml \
    --output-dir <PATH/TO/EVAL/OUTPUT> \
    model.config_file=<TRAIN/OUTPUT>/config.yaml \
    model.pretrained_weights=<TRAIN/OUTPUT>/eval/<ITER>/teacher_checkpoint.pth
```

To turn the `TEST` predictions into a Kaggle submission CSV, use
`dinov3/eval/hpa_kaggle_submission.py`:

```bash
python -m dinov3.eval.hpa_kaggle_submission \
    --preds <EVAL/OUTPUT>/preds_HPAWholeHR_split=TEST_<...>_with_metadata=false.npy \
    --sample-submission <PATH/TO/kaggle_hpa_sample_submission.csv> \
    --output submission.csv
```

The script applies sigmoid + 0.3 threshold per class (falling back to top-1
when nothing exceeds the threshold) and writes Kaggle's space-separated
`Predicted` column. The 28 model logits are assumed to be aligned with the
Kaggle protein-location class order shipped in
`hpa_whole_hr.PROTEIN_LOCATION`. The sample submission CSV is the standard
Kaggle template (not shipped) and dictates the row order; the OSS `TEST`
split sorts kaggle JPGs by filename, so the template must match.

### Other evals

The same `model.config_file` / `model.pretrained_weights` invocation works for
`dinov3/eval/{knn,linear,log_regression}.py` and
`dinov3/eval/{segmentation,depth}/run.py` — see the top-level README for
ImageNet / ADE20K / NYU command lines.
