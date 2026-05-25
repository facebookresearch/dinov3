import torch
from PIL import Image
import torchvision.transforms.functional as TF
from dino_utils import create_and_load_model, preprocess
from abc import ABC, abstractmethod
import sys
from transformers import AutoModel
import os
from omegaconf import OmegaConf
import clip
from torch.distributed.fsdp import fully_shard, FSDPModule
from geo.model.loratorch_utils import apply_lora_attn_mlp
from collections import OrderedDict


class Encoder(ABC):
    @abstractmethod
    def __init__(self, conf):
        pass

    @abstractmethod
    def get_image_tensors(self, images: list, return_grid: bool, ) -> torch.Tensor:
        '''
            preprocess and extract features of multiple images returning a list of tensors of 
            variables sizes

            images: list of paths or PIL Images
            return_grid: return patches embeddings if true, cls embedding otherwise
            returns: torch.Tensor ready to forward
        '''
        pass

    @abstractmethod
    def preprocess(self, image: Image, image_size: int, crop_center: bool) -> torch.Tensor:
        '''
            preprocess a single image

            image: PIL Image
            image_size: target image size
            crop_center: crop center to keep all image tensors the same size
            return: torch.Tensor with pixel values
        '''
        pass


def encoder_factory(conf):
    '''
        create the encoder accordingly to the conf file  

        conf: loaded configuration file with Omegaconf
    '''
    supported_models = {'Fossil': Fossil, 'DINOv3': Dinov3, 'CLIP': CLIP}
    if conf.encoder.name in supported_models.keys():
        return supported_models[conf.encoder.name](conf)
    


class Fossil(Encoder):
    def __init__(self, conf ):
        super().__init__()
        self.model = create_and_load_model(conf.encoder.config_path, conf.encoder.weight_path)
        self.dim = self.model.patch_embed.proj.weight.shape[0]
        self.im_size = conf.encoder.size
        

    def fsdp(self, fsdp_kwargs):
        for block in self.model.blocks:
            fully_shard(block, **fsdp_kwargs)

        fully_shard(self, **fsdp_kwargs)

    def preprocess(self, mask_image: Image, image_size: int, crop_center: bool = True) -> torch.Tensor:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        patch_size = 16

        w, h = mask_image.size
        # if crop center is true the smaller dimension will have the target size, 
        # if not the height dimension will have the target size as in the original dinov3 preprocess notebook
        if not crop_center or w > h:
            h_patches = int(image_size / patch_size)
            w_patches = int((w * image_size) / (h * patch_size))
        
        else:
            w_patches = int(image_size / patch_size)
            h_patches = int((h * image_size) / (w * patch_size))
            
        input = TF.to_tensor(TF.resize(mask_image, (w_patches * patch_size, h_patches * patch_size, )))
        input = TF.normalize(input, mean=mean, std=std)
        
        if crop_center:
            _, w, h = input.shape
            center = (int(w / 2), int(h/2))
            return input[:, 
                        center[0]-int(image_size/2):center[0]+int(image_size/2), 
                        center[1]-int(image_size/2):center[1]+int(image_size/2)]
    
        return input
    

    def get_image_tensors(self, images: list,):
        inputs = []
        for image in images:
            if type(image) == str:
                image = Image.open(image).convert('RGB')
            
            image = self.preprocess(image, self.im_size)
            inputs.append(image)

        if len(inputs) > 1:
            inputs = torch.stack(inputs)
        else:
            # batch size is 1
            inputs = inputs[0].unsqueeze(0)
        # list(patch_embed.parameters())[0]
        return inputs
    
    def forward(self, inputs, return_grid):
        device = self.model.embeddings.patch_embeddings.weight.device
        output = self.model.forward(inputs.to(device), return_grid=return_grid)
        return output

class Dinov3(torch.nn.Module, Fossil):
    def __init__(self, conf):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            conf.encoder.weight_path,
            device_map="cpu",
        )
        # print(self.model)
        self.dim = self.model.embeddings.patch_embeddings.weight.shape[0]
        self.im_size = conf.encoder.size
        self.num_registers = self.model.config.num_register_tokens
    
    def fsdp(self, fsdp_kwargs):
        for block in self.model.layer:
            fully_shard(block, **fsdp_kwargs)

        fully_shard(self, **fsdp_kwargs)

    def forward(self, inputs, return_grid):
        device = self.model.embeddings.patch_embeddings.weight.device
        output = self.model.forward(inputs.to(device))
        last_hidden = output.last_hidden_state

        if return_grid:
            return last_hidden[:, 1 + self.num_registers:, :] 
            
        else:
            return last_hidden[:, 0, :]

class CLIP(torch.nn.Module, Fossil):
    def __init__(self, conf):
        super().__init__()
        if os.path.exists(conf.encoder.config_path):
            # load finetuned model 
            enc_conf = OmegaConf.load(conf.encoder.config_path)        
            model, self._preprocess = clip.load(enc_conf.model.name.split(':')[-1], device='cpu', download_root=os.environ['HF_HOME'])
            if enc_conf.model.lora.apply:
                print('applying lora...')
                
                model = apply_lora_attn_mlp(model, enc_conf.model.lora)
                self.loratorch = True
                state_dict = torch.load(conf.encoder.config_path.replace('config.yaml', 'pytorch_model/pytorch_model.bin'))
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace('model.', '')] = v

                print('loading weights...')
                model.load_state_dict(new_state_dict, strict=False)
                
        else:
            model, self._preprocess = clip.load(conf.encoder.weight_path, device='cpu')
            
        self.model = model.visual
        self.dim = model.ln_final.weight.shape[0]
        # print(model.visual)
        self.model = model.visual
        self.im_size = 224

    def fsdp(self, fsdp_kwargs):
        for block in self.model.transformer.resblocks:
            fully_shard(block, **fsdp_kwargs)

        fully_shard(self, **fsdp_kwargs)

    def forward(self, inputs, return_grid):
        if return_grid:
            raise NotImplementedError('return grid not implemented for CLIP model')
        
        return self.model.forward(inputs.to(self.model.ln_post.weight.device))

    def preprocess(self, mask_image: Image, image_size: int, crop_center: bool = True) -> torch.Tensor:
        return self._preprocess(mask_image).squeeze(0)
    
    def get_image_tensors(self, images: list,):
        inputs = []
        for image in images:
            if type(image) == str:
                image = Image.open(image).convert('RGB')
            
            image = self.preprocess(image, self.im_size)
            inputs.append(image)

        if len(inputs) > 1:
            inputs = torch.stack(inputs)
        else:
            # batch size is 1
            inputs = inputs[0].unsqueeze(0)
        
        return inputs
        

if __name__ == '__main__':
    sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
    from dataset import ConversationDataset
    from fossilVL import FossilVL
    
    conf = OmegaConf.load('geo/config/baseNWPU.yaml')
    #conf.train.batch_size
    test_dataset = ConversationDataset(conf.data.root, conf.data.test)
    test_loader = test_dataset.get_loader(2, True)
    model = FossilVL(conf)
    
    for batch in test_loader:
        image_tensors = model.encoder.get_image_tensors(batch['image'], )
        # print(image_tensors.shape)
        image_embeddings = model.encoder(image_tensors, return_grid=False)
        print(image_embeddings.shape)
        image_embeddings = model.projection(image_embeddings)
        print(image_embeddings.shape)
        inputs = model.decoder.prepare_inputs(batch['conversation'])
        print(inputs.shape)
        # print(inputs)
        decoded = model.decoder.tokenizer.decode(inputs[0])
        print(decoded)
        text_embeddings = model.decoder.get_input_embeds(inputs)
        print(text_embeddings.shape)
        model_inputs = model.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
        print(model_inputs['input_embeddings'].shape)
        break
    
    
