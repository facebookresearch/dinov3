import torch
import sys
import os
sys.path.append(os.path.normpath(os.path.join(__file__, '../')))
from encoder import encoder_factory
from decoder import decoder_factory
from projector import multimodal_factory
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.distributed.tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh


class FossilVL(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.encoder = encoder_factory(conf)
        self.decoder = decoder_factory(conf)
        self.projection = multimodal_factory(conf.multimodal, self.encoder.dim, self.decoder.dim)
        self.use_grid = False if 'mapper' in conf.multimodal else True
        self.encoder
        self.log_first = True

    def forward(self, batch):
        # print('IM_SIZE', type(self.encoder.im_size))
        if len(self.encoder.im_size) > 1:
            if max(batch['size']) >= max(self.encoder.im_size):
                size = max(self.encoder.im_size)
            else:
                size = min(self.encoder.im_size)

            image_tensors = self.encoder.get_image_tensors(batch['image'], size=size)

        else:
            image_tensors = self.encoder.get_image_tensors(batch['image'], )
                
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
    
    def generate(self, image, prompt, num_beams=10, do_sample=False, max_new_tokens=100, **kwargs):
        device = self.decoder.model.device
        image_tensors = self.encoder.get_image_tensors(image).to(device)
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
                