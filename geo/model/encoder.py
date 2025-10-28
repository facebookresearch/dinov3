import torch
from PIL import Image
import torchvision.transforms.functional as TF
from dino_utils import create_and_load_model, preprocess
from abc import ABC, abstractmethod
import sys
import os
from omegaconf import OmegaConf


class Encoder(ABC):
    @abstractmethod
    def __init__(self, conf):
        pass

    @abstractmethod
    def get_image_features(self, images: list, return_grid: bool, is_training: bool) -> torch.Tensor:
        '''
            preprocess and extract features of multiple images returning a list of tensors of 
            variables sizes

            images: list of paths or PIL Images
            return_grid: return patches embeddings if true, cls embedding otherwise
            is_training: wheater or not to update gradients
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
    supported_models = {'Fossil': Fossil, }
    if conf.encoder.name in supported_models.keys():
        return supported_models[conf.encoder.name](conf)
        


class Fossil(torch.nn.Module, Encoder):
    def __init__(self, conf, ):
        super().__init__()
        self.model = create_and_load_model(conf.encoder.config_path, conf.encoder.weight_path)
        self.dim = self.model.patch_embed.proj.weight.shape[0]
        self.im_size = conf.encoder.size

    
    def preprocess(self, mask_image: Image, image_size: int, crop_center: bool = True) -> torch.Tensor:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        patch_size = 16

        w, h = mask_image.size
        # if crop center is true the smaller dimension will have the target size, 
        # if not the height dimension will have the target size as in the original dinov3 preprocess 
        if not crop_center or w > h:
            h_patches = int(image_size / patch_size)
            w_patches = int((w * image_size) / (h * patch_size))
        
        else:
            w_patches = int(image_size / patch_size)
            h_patches = int((h * image_size) / (w * patch_size))
            
        input = TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
        input = TF.normalize(input, mean=mean, std=std)
        
        if crop_center:
            _, w, h = input.shape
            center = (int(w / 2), int(h/2))
            return input[:, 
                        center[0]-int(image_size/2):center[0]+int(image_size/2), 
                        center[1]-int(image_size/2):center[1]+int(image_size/2)]
    
        return input
    

    def get_image_features(self, images: list, return_grid: bool, is_training:bool):
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

        output = self.model.forward(inputs, is_training=is_training, return_grid=return_grid)
        print(output.shape)
        return output


if __name__ == '__main__':
    sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
    from dataset import ConversationDataset
    
    conf = OmegaConf.load('geo/config/base.yaml')

    test_dataset = ConversationDataset(conf.data.root, conf.data.test)
    test_loader = test_dataset.get_loader(conf.train.batch_size, True)
    encoder = Fossil(conf)

    for batch in test_loader:
        out = encoder.get_image_features(batch['image'], False, False)
        break
    
