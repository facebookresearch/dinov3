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

    def forward(self, batch):
        image_tensors = self.encoder.get_image_tensors(batch['image'], )
        image_embeddings = self.encoder(image_tensors, return_grid=self.use_grid)
        image_embeddings = self.projection(image_embeddings)

        inputs = self.decoder.prepare_inputs(batch['conversation'])

        text_embeddings = self.decoder.get_input_embeds(inputs)
        model_inputs = self.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)

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
        return self.decoder.generate(model_inputs, num_beams=num_beams, do_sample=do_sample, max_new_tokens=max_new_tokens, **kwargs)
    
    def fsdp(self, fsdp_kwargs):
        self.decoder.fsdp(fsdp_kwargs)
        self.encoder.fsdp(fsdp_kwargs)
        fully_shard(self.projection, **fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)

