# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""SSLMetaArch with metadata-guided learning heads.

Adds classification, regression, and prototypical-contrastive heads on top of
the student's CLS pre-head embedding. Each head is independently configurable
via ``cfg.guide.guides`` and may use a Gradient Reversal Layer for adversarial
debiasing. Optionally normalizes the gradient magnitudes *between* the
individual guide losses on the backbone output so no single guide dominates
(``cfg.optim.grad_norm_normalization``); the SSL loss is untouched by this
mechanism.
"""

import logging

import torch
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

import dinov3.distributed as distributed
from dinov3.train.metadata_utils import (
    Classifier,
    PrototypicalContrastiveHead,
    Regressor,
    compute_classification_loss,
    compute_lambda,
    compute_prototypical_loss,
    compute_regression_loss,
)
from dinov3.train.param_groups import (
    fuse_params_groups,
    get_params_groups_with_decay_fsdp,
)
from dinov3.train.ssl_meta_arch import SSLMetaArch

logger = logging.getLogger("dinov3")

_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)


def _init_guide_weights(module: nn.Module) -> None:
    """Trunc-normal init for Linear; reset_parameters for known norm layers only.

    Restricted to norm layers (rather than any module exposing
    ``reset_parameters``) so guide-head Linears aren't re-initialized twice
    when ``apply`` walks the tree.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, _NORM_TYPES):
        module.reset_parameters()


class GuidedSSLMetaArch(SSLMetaArch):
    """SSLMetaArch extended with metadata-guided learning heads."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.guide_heads = nn.ModuleDict()
        self.guide_loss_fns: dict[str, nn.Module | None] = {}
        self.guide_configs: list = []

        guide_cfg_root = cfg.guide
        if not guide_cfg_root.enabled:
            logger.warning("GuidedSSLMetaArch instantiated but guide.enabled=False")
            self._lambda_schedule_type = "constant"
            self._lambda_warmup_steps = 0
            return

        for guide_cfg in guide_cfg_root.guides:
            if not guide_cfg.enabled:
                continue

            name = guide_cfg.name
            input_dim = (
                cfg.dino.head_n_prototypes
                if guide_cfg.grl and guide_cfg.grl_space == "prototype"
                else self.embed_dim
            )

            if guide_cfg.method == "regression":
                norm_cfg = guide_cfg.target_normalization
                output_min = tuple(norm_cfg.output_min) if norm_cfg.output_min else None
                output_max = tuple(norm_cfg.output_max) if norm_cfg.output_max else None
                self.guide_heads[name] = Regressor(
                    input_dim=input_dim,
                    hidden_dim=list(guide_cfg.hidden_dim),
                    n_outputs=guide_cfg.n_outputs,
                    dropout=guide_cfg.dropout,
                    output_activation=guide_cfg.output_activation,
                    output_min=output_min,
                    output_max=output_max,
                )
                self.guide_loss_fns[name] = nn.MSELoss()
                logger.info(
                    f"Guide '{name}': regression, n_outputs={guide_cfg.n_outputs}, "
                    f"activation={guide_cfg.output_activation}, GRL={guide_cfg.grl}"
                )
            elif guide_cfg.method == "prototypical":
                self.guide_heads[name] = PrototypicalContrastiveHead(
                    embed_dim=self.embed_dim,
                    n_classes=guide_cfg.n_outputs,
                    base_temperature=guide_cfg.proto_temperature,
                    centroid_momentum=guide_cfg.proto_centroid_momentum,
                    phi_min=guide_cfg.proto_phi_min,
                )
                self.guide_loss_fns[name] = None
                logger.info(
                    f"Guide '{name}': prototypical, n_classes={guide_cfg.n_outputs}, "
                    f"temperature={guide_cfg.proto_temperature}, GRL={guide_cfg.grl}"
                )
            elif guide_cfg.method == "classification":
                self.guide_heads[name] = Classifier(
                    input_dim=input_dim,
                    hidden_dim=list(guide_cfg.hidden_dim),
                    num_classes=guide_cfg.n_outputs,
                    dropout=guide_cfg.dropout,
                )
                self.guide_loss_fns[name] = (
                    nn.BCEWithLogitsLoss() if guide_cfg.use_bce else nn.CrossEntropyLoss()
                )
                logger.info(
                    f"Guide '{name}': classification, n_classes={guide_cfg.n_outputs}, "
                    f"GRL={guide_cfg.grl}, grl_space={guide_cfg.grl_space}, bce={guide_cfg.use_bce}"
                )
            else:
                raise ValueError(f"Unknown guide method: {guide_cfg.method}")

            self.guide_configs.append(guide_cfg)

        self._lambda_schedule_type = guide_cfg_root.lambda_schedule.type
        self._lambda_warmup_steps = guide_cfg_root.lambda_schedule.warmup_iterations
        self._total_iterations = cfg.train.OFFICIAL_EPOCH_LENGTH * cfg.optim.epochs

    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        del ignored_kwargs
        metrics_dict = {}

        n_global_crops = 2
        n_local_crops = self.n_local_crops
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["global_batch_size"]

        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        local_crops = data["collated_local_crops"].cuda(non_blocking=True)
        masks = data["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = data["mask_indices_list"].cuda(non_blocking=True)
        masks_weight = data["masks_weight"].cuda(non_blocking=True)
        n_masked_patches_tensor = data["n_masked_patches"].cuda(non_blocking=True)

        if self.has_gram_teacher:
            assert "collated_gram_teacher_crops" in data
            gram_teacher_crops = data["collated_gram_teacher_crops"].cuda(non_blocking=True)
        else:
            gram_teacher_crops = None

        teacher_global = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"],
        )

        student_global, student_local = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (n_local_crops, B)),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list,
        )

        if self.gram_use_loss:
            gram_global = self.get_gram_teacher_output(
                gram_teacher_crops.unflatten(0, (n_global_crops, B)) if gram_teacher_crops is not None else None,
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=global_crops.shape[-1],
            )
        else:
            gram_global = {}

        base_loss, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
        )

        guide_total: Tensor | float = 0.0
        per_guide_losses: dict[str, Tensor] = {}
        per_guide_weights: dict[str, float] = {}
        if self.cfg.guide.enabled and self.guide_heads:
            metadata = data.get("metadata")
            if metadata is None:
                raise RuntimeError(
                    "GuidedSSLMetaArch: guide.enabled=True but no 'metadata' key in batch. "
                    "Check that the dataset returns (image, (label, metadata_dataclass)) "
                    "and that the data loader's target_transform preserves the metadata "
                    "(see dinov3/train/train.py guide_enabled branch)."
                )
            guide_loss_dict = self._compute_guide_losses(
                student_global=student_global,
                teacher_global=teacher_global,
                metadata=metadata,
                iteration=iteration,
            )
            loss_dict.update(guide_loss_dict)
            for guide_cfg in self.guide_configs:
                name = guide_cfg.name
                guide_loss = guide_loss_dict.get(f"guide_{name}_loss")
                if not isinstance(guide_loss, Tensor):
                    continue
                weighted_value = guide_loss_dict[f"guide_{name}_weighted"]
                per_guide_losses[name] = guide_loss
                per_guide_weights[name] = (
                    guide_cfg.loss_weight / 10.0 if guide_cfg.grl else guide_cfg.loss_weight
                )
                guide_total = (
                    weighted_value if isinstance(guide_total, float) else guide_total + weighted_value
                )

        grad_norm_enabled = getattr(self.cfg.optim, "grad_norm_normalization", False)
        if (
            grad_norm_enabled
            and len(per_guide_losses) >= 2
            and isinstance(base_loss, Tensor)
        ):
            guide_contribution = self._apply_grad_norm_normalization(
                per_guide_losses=per_guide_losses,
                per_guide_weights=per_guide_weights,
                backbone_output=student_global["cls_pre_head"],
                loss_dict=loss_dict,
            )
            loss_accumulator = base_loss + guide_contribution
        else:
            loss_accumulator = base_loss + guide_total

        self.backprop_loss(loss_accumulator)

        return base_loss, metrics_dict | loss_dict

    @staticmethod
    def _apply_grad_norm_normalization(
        per_guide_losses: dict[str, Tensor],
        per_guide_weights: dict[str, float],
        backbone_output: Tensor,
        loss_dict: dict[str, Tensor | float],
    ) -> Tensor:
        """Equalize gradient magnitudes *between* the individual guide losses.

        The unweighted guide gradients w.r.t. the student CLS pre-head are
        rescaled to their geometric mean norm. The configured weights then
        control the relative contributions. The SSL loss is not part of this
        normalization.

        Uses ``torch.autograd.grad`` on an intermediate activation (not a
        sharded parameter) so it stays FSDP-safe.
        """
        weighted_losses = {name: loss * per_guide_weights[name] for name, loss in per_guide_losses.items()}
        device = backbone_output.device
        eps = 1e-6

        norms: dict[str, Tensor] = {}
        for name, loss in per_guide_losses.items():
            grad = None
            if per_guide_weights[name] != 0.0:
                grad = torch.autograd.grad(loss, backbone_output, retain_graph=True, allow_unused=True)[0]
            norm = (
                grad.float().norm()
                if grad is not None
                else torch.tensor(0.0, device=device)
            )
            norms[name] = norm
            loss_dict[f"grad_norm/{name}"] = norm

        # Geometric mean target across guides with non-vanishing gradients.
        valid_norms = [norm for norm in norms.values() if float(norm) > eps]
        if len(valid_norms) < 2:
            # Nothing meaningful to equalize against; fall back to plain sum.
            return sum(weighted_losses.values())

        log_sum = sum(torch.log(n + 1e-12) for n in valid_norms)
        target_norm = torch.exp(log_sum / len(valid_norms))
        loss_dict["grad_norm/target"] = target_norm

        contribution: Tensor | float = 0.0
        for name, weighted in weighted_losses.items():
            norm = norms[name]
            scale = target_norm / (norm + 1e-8) if float(norm) > eps else torch.tensor(1.0, device=device)
            loss_dict[f"grad_norm/{name}_scale"] = scale
            contribution = contribution + scale * weighted

        return contribution

    def _compute_guide_losses(
        self,
        *,
        student_global: dict[str, Tensor],
        teacher_global: dict[str, Tensor],
        metadata,
        iteration: int,
    ) -> dict[str, Tensor | float]:
        loss_dict: dict[str, Tensor | float] = {}
        lambda_value = compute_lambda(
            iteration,
            self._total_iterations,
            self._lambda_schedule_type,
            warmup_steps=self._lambda_warmup_steps,
        )
        loss_dict["guide_lambda"] = lambda_value

        cls_pre_head = student_global["cls_pre_head"]

        for guide_cfg in self.guide_configs:
            name = guide_cfg.name
            head = self.guide_heads[name]
            loss_fn = self.guide_loss_fns[name]

            labels = getattr(metadata, name).cuda(non_blocking=True)

            effective_lambda = -lambda_value if guide_cfg.grl else lambda_value
            loss_dict[f"guide_{name}_lambda"] = effective_lambda

            if guide_cfg.grl and guide_cfg.grl_space == "prototype":
                cls_input = student_global["cls_after_head"]
            else:
                cls_input = cls_pre_head

            if guide_cfg.method == "prototypical":
                teacher_cls = teacher_global["cls_pre_head"]
                guide_loss, accuracy = compute_prototypical_loss(
                    head, cls_input, labels, effective_lambda, teacher_cls_input=teacher_cls
                )
                loss_dict[f"guide_{name}_accuracy"] = accuracy
            elif guide_cfg.method == "regression":
                guide_loss, mse = compute_regression_loss(head, loss_fn, cls_input, labels, effective_lambda)
                loss_dict[f"guide_{name}_mse"] = mse
            else:
                guide_loss, accuracy = compute_classification_loss(
                    head, loss_fn, cls_input, labels, effective_lambda, use_bce=guide_cfg.use_bce
                )
                loss_dict[f"guide_{name}_accuracy"] = accuracy

            effective_weight = guide_cfg.loss_weight / 10.0 if guide_cfg.grl else guide_cfg.loss_weight
            loss_dict[f"guide_{name}_loss"] = guide_loss
            loss_dict[f"guide_{name}_weighted"] = effective_weight * guide_loss

        return loss_dict

    def init_weights(self) -> None:
        super().init_weights()
        for head in self.guide_heads.values():
            head.apply(_init_guide_weights)
            if isinstance(head, (PrototypicalContrastiveHead, Regressor)):
                head.reset_buffers()

    def prepare_for_distributed_training(self) -> None:
        super().prepare_for_distributed_training()

        if not self.guide_heads:
            return

        process_subgroup = distributed.get_process_subgroup()
        world_mesh = DeviceMesh.from_group(process_subgroup, "cuda")
        mp_policy = MixedPrecisionPolicy(
            param_dtype=_DTYPE_MAP[self.cfg.compute_precision.param_dtype],
            reduce_dtype=_DTYPE_MAP[self.cfg.compute_precision.reduce_dtype],
        )
        for name in list(self.guide_heads.keys()):
            logger.info(f"Wrapping guide head '{name}' with FSDP")
            self.guide_heads[name] = fully_shard(
                self.guide_heads[name],
                mesh=world_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
            )

    def get_params_groups(self):
        all_params_groups = super().get_params_groups()
        for name, head in self.guide_heads.items():
            logger.info(f"Getting parameter groups for guide head {name}")
            groups = get_params_groups_with_decay_fsdp(
                model=head,
                lr_decay_rate=self.cfg.optim.layerwise_decay,
                patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
                dino_head_wd_multiplier=self.cfg.optim.dino_head_wd_multiplier,
                **self._freeze_knobs(),
            )
            if self.cfg.optim.multi_tensor_optim:
                fused = fuse_params_groups(groups)
                for g in fused:
                    g["foreach"] = True
                    g["fused"] = True
                all_params_groups += list(fused)
            else:
                all_params_groups += groups
        return all_params_groups
