import torch
import sys
import os
from PIL import Image
sys.path.append(os.path.normpath(os.path.join(__file__, '../')))
from encoder import encoder_factory
from decoder import decoder_factory
from projector import ScaleEmbedding, multimodal_factory
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh

CHECKPOINT_FORMAT = 'trainable'
# modules trained from scratch: always in checkpoints, even when frozen (e.g. during RL)
TRAINED_MODULES = ('projection.', 'scale_embedding.')


def project_features(projection, scale_embedding, features, mm_per_pixel=None):
    """Visual tokens of encoder features. With a scale embedding (scale_token in the config), the embedding of
    log(mm_per_pixel), the physical size of one encoder input pixel of each image (tensor [b]), is the first visual
    token and also conditions the projection when it supports it (qformer)."""
    if scale_embedding is None:
        return projection(features)
    if mm_per_pixel is None:
        raise ValueError('scale_token is enabled but the images have no scale (set data.scale_field)')
    scale = scale_embedding(torch.log(torch.as_tensor(mm_per_pixel, dtype=torch.float32, device=features.device)))
    tokens = projection(features, condition=scale) if getattr(projection, 'conditioned', False) else projection(features)
    return torch.cat([scale[:, None].to(tokens.dtype), tokens], dim=1)


def checkpoint_state_dict(model, state_dict=None):
    """The part of the state dict a checkpoint needs: trainable parameters, buffers, LoRA adapters, the projection, the
    scale embedding and every weight the model was loaded with (load_checkpoint), e.g. the SFT projection when it is frozen during RL.
    Frozen pretrained encoder/decoder weights are left out: FossilVL(conf) builds them again from their sources."""
    state_dict = model.state_dict() if state_dict is None else state_dict
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    loaded = getattr(model, 'loaded_keys', set())
    return {k: v for k, v in state_dict.items()
            if k not in frozen or 'lora_' in k or k.startswith(TRAINED_MODULES) or k in loaded}


def save_checkpoint(model, path, state_dict=None):
    """Saves checkpoint_state_dict; state_dict is the full one when it was gathered elsewhere (e.g. from FSDP)."""
    tmp = path + '.tmp'
    torch.save({'format': CHECKPOINT_FORMAT, 'state_dict': checkpoint_state_dict(model, state_dict)}, tmp)
    os.replace(tmp, path)  # never leaves a half-written checkpoint behind


def load_checkpoint(model, path):
    """Loads a checkpoint of save_checkpoint, or a full state dict as written before it. LoRA must already be applied
    (decoder.apply_lora) when the checkpoint has adapters."""
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and ckpt.get('format') == CHECKPOINT_FORMAT:
        state_dict = ckpt['state_dict']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # the frozen pretrained weights are expected to be missing, the trained ones are not
        lost = [k for k in missing if 'lora_' in k or k.startswith(TRAINED_MODULES)]
        if unexpected or lost:
            raise RuntimeError(f'{path} does not match the model: unexpected keys {unexpected[:5]}, '
                               f'missing trained keys {lost[:5]} (is LoRA applied as in the config?)')
    else:
        state_dict = ckpt
        model.load_state_dict(state_dict)
    model.loaded_keys = set(state_dict)
    return model


class FossilVL(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.encoder = encoder_factory(conf)
        self.decoder = decoder_factory(conf)
        self.use_grid = False if 'mapper' in conf.multimodal else True
        # with use_grid the projection reads patch tokens, which can be narrower than the global embedding (DINOtxt)
        input_dim = getattr(self.encoder, 'grid_dim', self.encoder.dim) if self.use_grid else self.encoder.dim
        # scale_token: a learned token with the physical scale of each image (needs data.scale_field)
        self.scale_embedding = None
        if conf.get('scale_token', False):
            if not conf.data.get('scale_field'):
                raise ValueError('scale_token needs data.scale_field, the sample field with the image scale')
            self.scale_embedding = ScaleEmbedding(self.decoder.dim)
        self.projection = multimodal_factory(conf.multimodal, input_dim, self.decoder.dim,
                                             condition_dim=self.decoder.dim if self.scale_embedding is not None else None)
        self.log_first = True

    def encoder_sizes(self):
        # a list in the configs (encoder.size), a single int for CLIP
        im_size = self.encoder.im_size
        return [im_size] if isinstance(im_size, int) else list(im_size)

    def input_size(self, sizes):
        """Encoder input size for images of these sizes (the 'size' of the samples): with several encoder sizes, the
        largest one when an image is at least that large, else the smallest one; None with a single encoder size."""
        encoder_sizes = self.encoder_sizes()
        if len(encoder_sizes) == 1:
            return None
        return max(encoder_sizes) if max(sizes) >= max(encoder_sizes) else min(encoder_sizes)

    def mm_per_pixel(self, sizes, scales, input_size=None):
        """Physical size of one encoder input pixel per image: the center crop covers the shorter side (sizes, in
        pixels of the loaded image, scales in mm per pixel), resized to the encoder input size."""
        side = input_size or self.encoder_sizes()[0]
        return torch.as_tensor(sizes, dtype=torch.float32) * torch.as_tensor(scales, dtype=torch.float32) / side

    def project(self, features, sizes=None, scales=None, input_size=None):
        """Visual tokens of encoder features, with the scale token when scale_token is enabled."""
        mm = None
        if self.scale_embedding is not None:
            if sizes is None or scales is None:
                raise ValueError('scale_token is enabled: the size and scale of every image are needed')
            mm = self.mm_per_pixel(sizes, scales, input_size)
        return project_features(self.projection, self.scale_embedding, features, mm)

    def image_tensors(self, images, size):
        # CLIP's get_image_tensors has no size argument (single size only)
        if size is None:
            return self.encoder.get_image_tensors(images)
        return self.encoder.get_image_tensors(images, size=size)

    def forward(self, batch):
        # print('IM_SIZE', type(self.encoder.im_size))
        size = self.input_size(batch['size']) if len(self.encoder_sizes()) > 1 else None
        image_tensors = self.image_tensors(batch['image'], size)

        image_embeddings = self.encoder(image_tensors, return_grid=self.use_grid)
        image_embeddings = self.project(image_embeddings, batch.get('size'), batch.get('scale'), size)

        inputs = self.decoder.prepare_inputs(batch['conversation'])
        text_embeddings = self.decoder.get_input_embeds(inputs)
        model_inputs = self.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)

        if self.log_first:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0  # 0 without torchrun
            print('RANK:', rank, 'IMAGE SHAPE', image_tensors.shape)
            print('RANK:', rank, batch['image'][0])
            print('RANK', rank, 'TEXT SHAPE', inputs.shape)
            self.log_first = False
        
        return self.decoder(model_inputs)
    
    def generate(self, image, prompt, num_beams=10, do_sample=False, max_new_tokens=100, size=None, scale=None,
                 **kwargs):
        """size: the 'size' of the sample(s), as in training; without it, the shorter side of each image file.
        scale: mm per pixel of the image(s), the dataset's 'scale' (needed with scale_token)."""
        device = self.decoder.model.device
        sizes = None
        if len(self.encoder_sizes()) > 1 or self.scale_embedding is not None:
            if size is None:
                sizes = [min(Image.open(i).size) if isinstance(i, str) else min(i.size) for i in image]
            else:
                sizes = list(size) if isinstance(size, (list, tuple)) else [size]
        # single encoder size: the encoder's own
        input_size = self.input_size(sizes) if len(self.encoder_sizes()) > 1 else None
        scales = None if scale is None else (list(scale) if isinstance(scale, (list, tuple)) else [scale])
        image_tensors = self.image_tensors(image, input_size).to(device)
        image_embeddings = self.encoder(image_tensors, return_grid=self.use_grid)
        image_embeddings = self.project(image_embeddings, sizes, scales, input_size)

        messages = [
            {"role": "user", "content": prompt}
        ]
       
        inputs = self.decoder.prepare_inputs([messages], add_gen_prompt=True).to(device)
        text_embeddings = self.decoder.get_input_embeds(inputs)
        model_inputs = self.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
        return self.decoder.generate(model_inputs, num_beams=num_beams, do_sample=do_sample, **kwargs)
    
    def fsdp(self, fsdp_kwargs):
        self.decoder.fsdp(fsdp_kwargs)
        self.encoder.fsdp(fsdp_kwargs)
        fully_shard(self.projection, **fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)

    def trainable_params(self):
        trainable_params = 0
        total_params = 0 
        modules = [self.decoder, self.encoder, self.projection] + ([self.scale_embedding] if self.scale_embedding is not None else [])

        trainable_params = sum(
            p.numel()
            for module in modules
            for p in module.parameters()
            if p.requires_grad
        )

        total_params = sum(
            p.numel()
            for module in modules
            for p in module.parameters()
        )

        print(f"total params:     {total_params:,}")
        print(f"percentage:   {100 * trainable_params / total_params:.4f}%")


        trainable_params = sum(
                    p.numel()
                    for p in self.encoder.parameters()
                    if p.requires_grad
                )
        print(f"encoder trainable params: {trainable_params:,}")
                        
        trainable_params = sum(
                    p.numel()
                    for p in self.projection.parameters()
                    if p.requires_grad
                )
        print(f"multimodal trainable params: {trainable_params:,}")

        trainable_params = sum(
                    p.numel()
                    for p in self.decoder.parameters()
                    if p.requires_grad
                )
        print(f"decoder trainable params: {trainable_params:,}")
                