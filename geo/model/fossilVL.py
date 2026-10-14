import torch
import sys
import os
from PIL import Image
sys.path.append(os.path.normpath(os.path.join(__file__, '../')))
from encoder import encoder_factory
from decoder import decoder_factory
from projector import multimodal_factory
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh

CHECKPOINT_FORMAT = 'trainable'


def checkpoint_state_dict(model, state_dict=None):
    """The part of the state dict a checkpoint needs: trainable parameters, buffers, LoRA adapters, the projection and
    every weight the model was loaded with (load_checkpoint), e.g. the SFT projection when it is frozen during RL.
    Frozen pretrained encoder/decoder weights are left out: FossilVL(conf) builds them again from their sources."""
    state_dict = model.state_dict() if state_dict is None else state_dict
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    loaded = getattr(model, 'loaded_keys', set())
    return {k: v for k, v in state_dict.items()
            if k not in frozen or 'lora_' in k or k.startswith('projection.') or k in loaded}


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
        lost = [k for k in missing if 'lora_' in k or k.startswith('projection.')]
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
        self.projection = multimodal_factory(conf.multimodal, input_dim, self.decoder.dim)
        self.encoder
        self.log_first = True

    def input_size(self, sizes):
        """Encoder input size for images of these sizes (the 'size' of the samples): with several encoder sizes, the
        largest one when an image is at least that large, else the smallest one; None with a single encoder size."""
        if len(self.encoder.im_size) == 1:
            return None
        return max(self.encoder.im_size) if max(sizes) >= max(self.encoder.im_size) else min(self.encoder.im_size)

    def image_tensors(self, images, size):
        # CLIP's get_image_tensors has no size argument (single size only)
        if size is None:
            return self.encoder.get_image_tensors(images)
        return self.encoder.get_image_tensors(images, size=size)

    def forward(self, batch):
        # print('IM_SIZE', type(self.encoder.im_size))
        size = self.input_size(batch['size']) if len(self.encoder.im_size) > 1 else None
        image_tensors = self.image_tensors(batch['image'], size)

        image_embeddings = self.encoder(image_tensors, return_grid=self.use_grid)
        image_embeddings = self.projection(image_embeddings)

        inputs = self.decoder.prepare_inputs(batch['conversation'])
        text_embeddings = self.decoder.get_input_embeds(inputs)
        model_inputs = self.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)

        if self.log_first:
            print('RANK:', torch.distributed.get_rank(), 'IMAGE SHAPE', image_tensors.shape)
            print('RANK:', torch.distributed.get_rank(), batch['image'][0])
            print('RANK', torch.distributed.get_rank(), 'TEXT SHAPE', inputs.shape)                    
            self.log_first = False
        
        return self.decoder(model_inputs)
    
    def generate(self, image, prompt, num_beams=10, do_sample=False, max_new_tokens=100, size=None, **kwargs):
        """size: the 'size' of the sample(s), as in training; without it, the shorter side of each image file."""
        device = self.decoder.model.device
        if len(self.encoder.im_size) > 1:
            if size is None:
                size = [min(Image.open(i).size) if isinstance(i, str) else min(i.size) for i in image]
            size = self.input_size(size if isinstance(size, (list, tuple)) else [size])
        else:
            size = None  # single encoder size: the encoder's own
        image_tensors = self.image_tensors(image, size).to(device)
        image_embeddings = self.encoder(image_tensors, return_grid=self.use_grid)
        image_embeddings = self.projection(image_embeddings)

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
        modules = [self.decoder, self.encoder, self.projection]

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
                