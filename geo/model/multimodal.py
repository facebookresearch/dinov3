import torch
import sys
import os
sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
from encoder import encoder_factory
from decoder import decoder_factory
from projector import multimodal_factory
from PIL import Image

class FossilVL(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.encoder = encoder_factory(conf)
        self.decoder = decoder_factory(conf)
        self.projection = multimodal_factory(conf.multimodal, self.encoder.dim, self.decoder.dim)
        self.use_grid = False if 'mapper' in conf.multimodal else True
        self.encoder

    def forward(self, batch, encoder_training:bool=False):
        image_embeddings = self.encoder.get_image_features(
            batch['image'], self.use_grid, encoder_training
            )
        image_embeddings = self.projection(image_embeddings)
        inputs = self.decoder.prepare_inputs(batch['conversation'])
        text_embeddings = self.decoder.get_embedding_layer()(inputs)
        model_inputs = self.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
        return self.decoder(model_inputs)
    
    def generate(self, image, prompt):
        image_embeddings = self.encoder.get_image_features([image], self.use_grid, False)
        image_embeddings = self.projection(image_embeddings)
        return self.decoder.generate(image_embeddings, prompt)