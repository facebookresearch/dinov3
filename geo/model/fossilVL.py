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

    def forward(self, batch, device):
        image_tensors = self.encoder.get_image_tensors(batch['image'], ).to(device)
        image_embeddings = self.encoder(image_tensors, return_grid=self.use_grid)
        image_embeddings = self.projection(image_embeddings)
        
        inputs = self.decoder.prepare_inputs(batch['conversation']).to(device)
        
        text_embeddings = self.decoder.get_input_embeds(inputs)
        model_inputs = self.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
        
        return self.decoder(model_inputs)
    
    def generate(self, image, prompt):
        image_embeddings = self.encoder.get_image_features([image], self.use_grid)
        image_embeddings = self.projection(image_embeddings)
        return self.decoder.generate(image_embeddings, prompt)
    
    def fsdp(self, fsdp_kwargs):
        self.decoder.fsdp(fsdp_kwargs)
        self.encoder.fsdp(fsdp_kwargs)
        fully_shard(self.projection, **fsdp_kwargs)
        fully_shard(self, **fsdp_kwargs)

