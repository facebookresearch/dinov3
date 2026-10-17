import torch
from PIL import Image
import torchvision.transforms.functional as TF
from dino_utils import create_and_load_model, load_backbone_weights, preprocess
from abc import ABC, abstractmethod
from collections import defaultdict
import sys
from transformers import AutoModel
import os
from omegaconf import OmegaConf
import clip
from PIL import Image, ImageFile
from torch.distributed.fsdp import fully_shard, FSDPModule
from geo.model.loratorch_utils import apply_lora_attn_mlp
from collections import OrderedDict
from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from glob import glob
import re
from projector import QFormer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_transform(image: Image, image_size: int = 224, patch_size: int = 16,) -> torch.Tensor:
    w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    image_resized = TF.to_tensor(TF.resize(image, (h_patches * patch_size, w_patches * patch_size)))
    return TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

def crop_transform(path: list, crop_size: int = 224, patch_size: int = 16) -> torch.Tensor:
    image = Image.open(path[0])
    center_x = image.width // 2
    center_y = image.height // 2
    left = center_x - crop_size // 2
    top = center_y - crop_size // 2
    right = center_x + crop_size // 2
    bottom = center_y + crop_size // 2
    cropped_image = image.crop((left, top, right, bottom))
    return resize_transform(cropped_image, image_size=crop_size, patch_size=patch_size)

# class Encoder(ABC):
#     @abstractmethod
#     def __init__(self, conf):
#         pass

#     @abstractmethod
#     def get_image_tensors(self, images: list, return_grid: bool, ) -> torch.Tensor:
#         '''
#             preprocess and extract features of multiple images returning a list of tensors of 
#             variables sizes

#             images: list of paths or PIL Images
#             return_grid: return patches embeddings if true, cls embedding otherwise
#             returns: torch.Tensor ready to forward
#         '''
#         pass

#     @abstractmethod
#     def preprocess(self, image: Image, image_size: int, crop_center: bool) -> torch.Tensor:
#         '''
#             preprocess a single image

#             image: PIL Image
#             image_size: target image size
#             crop_center: crop center to keep all image tensors the same size
#             return: torch.Tensor with pixel values
#         '''
#         pass


def encoder_factory(conf):
    '''
        create the encoder accordingly to the conf file  

        conf: loaded configuration file with Omegaconf
    '''
    supported_models = {'Fossil': Fossil, 'DINOv3': Dinov3, 'CLIP': CLIP, 'DINOtxt': Dinotxt, 'DualEncoder': DualEncoder}
    if conf.encoder.name in supported_models.keys():
        return supported_models[conf.encoder.name](conf)
    raise ValueError(f'unknown encoder {conf.encoder.name} ({", ".join(supported_models)})')


class Fossil(torch.nn.Module):
    def __init__(self, conf=None ):
        super().__init__()
        if conf is not None:
            self.model = create_and_load_model(conf.encoder.config_path, conf.encoder.weight_path)
            self.dim = self.model.patch_embed.proj.weight.shape[0]
            self.im_size = conf.encoder.size
        

    def fsdp(self, fsdp_kwargs):
        print(self.model)
        for block in self.model.blocks:
            fully_shard(block, **fsdp_kwargs)

        fully_shard(self, **fsdp_kwargs)

    def preprocess(self, mask_image: Image, image_size: int, crop_center: bool = True) -> torch.Tensor:
        """crop_center (default): the center image_size x image_size crop at the image's own resolution, as vlm-tuning's
        crop_transform, which the encoders were aligned on (crop first, no resize); an image whose shorter side is
        smaller than image_size is enlarged to it first. crop_center=False: the whole image resized to a height of
        image_size, as in the original dinov3 preprocess notebook."""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        patch_size = 16

        if crop_center:
            if min(mask_image.size) < image_size:
                # too small to crop: enlarged so that its shorter side is image_size
                mask_image = TF.resize(mask_image, image_size)
            w, h = mask_image.size
            left = w // 2 - image_size // 2
            top = h // 2 - image_size // 2
            input = TF.to_tensor(mask_image.crop((left, top, left + image_size, top + image_size)))

        else:
            w, h = mask_image.size
            h_patches = int(image_size / patch_size)
            w_patches = int((w * image_size) / (h * patch_size))
            # TF.resize takes (height, width)
            input = TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

        return TF.normalize(input, mean=mean, std=std)
    

    def get_image_tensors(self, images: list, size=None):
        inputs = []
        for image in images:
            if type(image) == str:
                image = Image.open(image).convert('RGB')
            
            image = self.preprocess(image, self.im_size[0] if size is None else size)
            inputs.append(image)

        if len(inputs) > 1:
            inputs = torch.stack(inputs)
        else:
            # batch size is 1
            inputs = inputs[0].unsqueeze(0)

        # list(patch_embed.parameters())[0]
        return inputs
    
    def forward(self, inputs, return_grid):
        device = self.model.patch_embed.proj.weight.device
        output = self.model.forward(inputs.to(device), return_grid=return_grid)
        return output

class Dinotxt(Fossil):
    def __init__(self, conf):
        super().__init__()
        model, _ = torch.hub.load(
            './', 
            'dinov3_vitl16_dinotxt_tet1280d20h24l', 
            source='local',
            weights='/nethome/recpinfo/users/fibz/cache/dinov3_vitl16_dinotxt.pth',
            pretrained=False
        )
        
        self.model = model
        self.dim = 2048 if conf.encoder.average_patch else 1024
        self.grid_dim = 1024  # width of each patch token (return_grid=True)
        self.im_size = conf.encoder.size
        self.encode_image = model.encode_image

        if os.path.exists(conf.encoder.config_path):
            enc_conf = OmegaConf.load(conf.encoder.config_path)
            # DINO-TXT aligned (in vlm-tuning) on top of a DINOv3 SSL backbone of this project instead of its own
            backbone_weights = enc_conf.model.get('backbone_weights')
            if backbone_weights:
                load_backbone_weights(model.visual_model.backbone, backbone_weights,
                                      enc_conf.model.get('backbone_branch', 'teacher'))

            # only the vision tower is used for captioning. With LoRA on the whole model (not vision_head_only) its
            # attention layers have adapters; with vision_head_only LoRA is on the text tower only, so none here
            visual = self.model.visual_model
            vision_head_only = enc_conf.model.get('vision_head_only', False)
            if enc_conf.model.lora.apply and not vision_head_only:
                config = LoraConfig(
                    r=enc_conf.model.lora.r,
                    lora_alpha=enc_conf.model.lora.alpha,
                    target_modules=["qkv"],
                    lora_dropout=enc_conf.model.lora.dropout_rate,
                    bias="none"
                )
                visual = get_peft_model(self.model, config).base_model.model.visual_model

            if enc_conf.model.load_weights:
                path = os.path.join(enc_conf.output_dir, 'pytorch_model', 'pytorch_model.bin') #'last.ckpt')
                print('LOADING MODEL AT', path)
                ckpt = torch.load(path, weights_only=False)
                # vlm-tuning names the vision tower <...>.visual.transformer.<visual_model key>, whatever the wrappers
                # (lightning module, peft, deepspeed) in front of it
                marker = 'visual.transformer.'
                visual_state = OrderedDict((k.split(marker, 1)[1], v) for k, v in ckpt.items() if marker in k)

                print('loading weights...')
                missing, unexpected = visual.load_state_dict(visual_state, strict=False)
                # frozen weights can be left out of the checkpoint (deepspeed exclude_frozen_parameters); the backbone
                # was loaded above when it is an SSL one, and comes from the checkpoint otherwise
                if backbone_weights:
                    missing = [k for k in missing if not k.startswith('backbone.')]
                if missing or unexpected or not visual_state:
                    raise RuntimeError(f'{path} does not match the DINO-TXT vision tower: {len(visual_state)} vision '
                                       f'weights, missing {missing[:5]}, unexpected {unexpected[:5]}')

            self.model = visual
                    
        print(self.model)

    def fsdp(self, fsdp_kwargs):
        for block in self.model.backbone.blocks:
            fully_shard(block, **fsdp_kwargs)

        fully_shard(self, **fsdp_kwargs)

    def forward(self, inputs, return_grid):
        device = self.model.backbone.patch_embed.proj.weight.device
        dtype = self.model.backbone.patch_embed.proj.weight.dtype
        if return_grid:
            # patch tokens after the vision head (b x patches x 1024), the ones encode_image averages into its second
            # half; for projections that read every patch (e.g. qformer)
            _, patch_tokens, _ = self.model(inputs.to(device, dtype=dtype))
            return patch_tokens

        x = self.encode_image(inputs.to(device, dtype=dtype))
        if self.dim == 1024: # only classification embedding
            x = x[:, :1024]

        return x

def vlm_tuning_checkpoint(enc_conf, weight_path=None):
    """Weights of a vlm-tuning run: weight_path if given, else from its output_dir the best checkpoint
    (checkpoint-<epoch>.ckpt, kept by its ModelCheckpoint), the consolidated deepspeed one or last.ckpt."""
    if weight_path:
        return weight_path
    out = enc_conf.output_dir
    best = sorted(glob(os.path.join(out, 'checkpoint-*.ckpt')), key=os.path.getmtime)
    candidates = best[-1:] + [os.path.join(out, 'pytorch_model', 'pytorch_model.bin'), os.path.join(out, 'last.ckpt')]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'no vlm-tuning checkpoint in {out} (checkpoint-*.ckpt, pytorch_model/pytorch_model.bin, '
                            'last.ckpt); set encoder.weight_path')


def head_module(kind, input_dim, output_dim):
    # vlm-tuning model/dual_encoder.build_head (same module names, so its weights load)
    if kind == 'linear':
        return torch.nn.Linear(input_dim, output_dim)
    if kind == 'mlp':
        hidden = max(input_dim, output_dim)
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, output_dim))
    raise ValueError(f'unknown aligner head {kind} (qformer, mlp, linear)')


class DualEncoder(Fossil):
    """Vision side of a vlm-tuning DualEncoder run (DINOv3 SSL backbone aligned with a text encoder): the SSL backbone,
    its LoRA adapters when the run trained them, and the trained vision head (Q-Former, MLP or linear).

    encoder.config_path: config.yaml of the vlm-tuning run; encoder.weight_path: its checkpoint (empty: found in the
    run's output_dir). Outputs, with a Q-Former head:
        return_grid=True   the query outputs, b x queries x embed_dim (the tokens compared with the text in
                           vlm-tuning), e.g. for multimodal 'projector' (one visual token per query, as in BLIP-2)
        return_grid=False  one vector per image, b x embed_dim, pooled as in vlm-tuning ('max' pooling: the mean of
                           the queries), e.g. for 'mapper'
    With an MLP/linear head: the backbone patch tokens (grid) or the head output (global).
    encoder.train_head: also train the vision head in SFT (the backbone follows train.stage2.frozen_encoder)."""

    def __init__(self, conf):
        super().__init__()
        enc_conf = OmegaConf.load(conf.encoder.config_path)
        if enc_conf.model.name.split(':')[0] != 'DualEncoder':
            raise ValueError(f'{conf.encoder.config_path} is not a vlm-tuning DualEncoder run ({enc_conf.model.name})')
        v, a, lora = enc_conf.model.vision, enc_conf.model.aligner, enc_conf.model.lora
        repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        self.backbone = torch.hub.load(repo, v.arch, source='local', pretrained=False)
        load_backbone_weights(self.backbone, v.weights, v.get('branch', 'teacher'))
        if lora.apply and lora.encoder in ('both', 'vision'):
            config = LoraConfig(r=lora.r, lora_alpha=lora.alpha, target_modules=list(lora.get('vision_targets', ['qkv'])),
                                lora_dropout=lora.dropout_rate, bias='none')
            inject_adapter_in_model(config, self.backbone)

        vision_dim = self.backbone.embed_dim
        self.kind = a.vision
        self.tokens = v.get('tokens', 'cls+patch')
        self.pooled = v.get('pooled', 'cls')
        self.pooling = a.get('pooling', 'mean')
        if self.kind == 'qformer':
            self.head = QFormer(vision_dim, a.embed_dim, a.queries, layers=a.layers, dim=a.width, heads=a.heads,
                                dropout=a.dropout)
            self.grid_dim = a.embed_dim
        else:
            self.head = head_module(self.kind, 2 * vision_dim if self.pooled == 'cls+mean' else vision_dim, a.embed_dim)
            self.grid_dim = vision_dim
        self.dim = a.embed_dim
        self.im_size = conf.encoder.size
        self.train_head = conf.encoder.get('train_head', False)

        path = vlm_tuning_checkpoint(enc_conf, conf.encoder.get('weight_path'))
        print('LOADING DUAL ENCODER WEIGHTS AT', path)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        ckpt = ckpt.get('state_dict', ckpt)
        # vlm-tuning keys: [wrappers.]model.vision_head.<head key>, [wrappers.]model.backbone.<backbone key>
        head = {m.group(1): w for k, w in ckpt.items() if (m := re.search(r'(?:^|\.)vision_head\.(.+)', k))}
        backbone = {m.group(1): w for k, w in ckpt.items() if (m := re.search(r'(?:^|\.)model\.backbone\.(.+)', k))}
        missing, unexpected = self.head.load_state_dict(head, strict=False)
        if missing or unexpected:
            raise RuntimeError(f'{path}: vision head does not match the config: missing {missing[:5]}, '
                               f'unexpected {unexpected[:5]}')
        # the backbone base weights are the SSL ones loaded above; the checkpoint adds the LoRA adapters, if any
        missing, unexpected = self.backbone.load_state_dict(backbone, strict=False)
        missing_lora = [k for k in missing if 'lora_' in k]
        if missing_lora or unexpected:
            raise RuntimeError(f'{path}: backbone weights do not match: missing adapters {missing_lora[:5]}, '
                               f'unexpected {unexpected[:5]}')
        print(f'loaded the {self.kind} head ({len(head)} tensors) and {len(backbone)} backbone tensors')

    def fsdp(self, fsdp_kwargs):
        for block in self.backbone.blocks:
            fully_shard(block, **fsdp_kwargs)

        fully_shard(self, **fsdp_kwargs)

    def forward(self, inputs, return_grid):
        weight = self.backbone.patch_embed.proj.weight
        out = self.backbone.forward_features(inputs.to(weight.device, dtype=weight.dtype))
        cls, patches = out['x_norm_clstoken'], out['x_norm_patchtokens']
        if self.kind == 'qformer':
            queries = self.head(torch.cat([cls[:, None], patches], 1) if self.tokens == 'cls+patch' else patches)
            if return_grid:
                return queries
            return queries[:, 0] if self.pooling == 'first' else queries.mean(1)
        if return_grid:
            return patches
        return self.head(torch.cat([cls, patches.mean(1)], -1) if self.pooled == 'cls+mean' else cls)


class Dinov3(Fossil):
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

class CLIP(Fossil):
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
    from dataset import ConversationDataset, conversation_collate
    from fossilVL import FossilVL
    
    conf = OmegaConf.load('geo/config/dinotxt.yaml')
    conf.encoder.average_patch = True
    #conf.train.batch_size

    test_dataset = ConversationDataset(conf.data.root, conf.data.test)
    test_loader = test_dataset.get_loader(conversation_collate, 4, True)
    model = FossilVL(conf)

    for name, param in model.decoder.named_parameters():
            param.requires_grad = 'lora' in name.lower()

    model.trainable_params()

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
    
    
