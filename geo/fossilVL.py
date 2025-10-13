import numpy as np
import torch
from transformers import AutoProcessor
import abc
import copy
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from encoder_utils import create_and_load_model, resize_transform
from projector import Qwen2_5_VLPatchMerger


class FossilVL(torch.nn.Module):
    def __init__(
            self, 
            decoder_name="Qwen/Qwen2.5-VL-3B-Instruct",
            weight_path = '/nethome/recpinfo/users/fibz/data/checkpoints/dinov3/ckpt/9999/consolidated_model/pytorch_model.bin',
            cfg_path = 'dinov3/configs/train/dinov3_vitl16_geo.yaml',
            pretrained_dino = None,

            ):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(decoder_name, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(decoder_name, patch_size=16, ) #temporal_patch_size=1, merge_size=1)
        self.patch_merger = Qwen2_5_VLPatchMerger(2048, 1024, 2)
        self.im_start = 151644
        
        # print(self.model)
        
        if pretrained_dino is None:
            self.model.model.visual = create_and_load_model(cfg_path, weight_path)
        else:
            # TODO add option to load original DINO from HF 
            raise(NotImplementedError, 'only local finetuned models are supported for now')

   
    def process_input(self, conversations: list, image_inputs: torch.Tensor):
        texts = []
        for conversation in conversations:
            texts.append(self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))
        print(texts[-1])
        return self.processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        
    
    def preprocess_images(self, images: list, image_size):
        inputs = []
  
        for image in images:
            if type(image) == str:
                image = Image.open(image).convert('RGB')

            tensor = resize_transform(image, image_size=image_size)
            inputs.append(tensor)
  
        inputs = torch.stack(inputs, dim=0)
        return inputs 
    

    def get_image_features(self, images: torch.Tensor, return_grid: bool, is_training:bool):
        output = self.model.model.visual.forward(images, is_training=is_training, return_grid=return_grid)

        if return_grid:
            return self.patch_merger(output)
        
        return output


    def get_input_embeddings(self):
        return self.model.get_input_embeddings()


    def merge_embeddings(self, vision_embeddings, text_embeddings, tokens):
        image_mask, _ = self.model.model.get_placeholder_mask(
            tokens,
            inputs_embeds=text_embeddings,
            image_features=vision_embeddings,
        )

        vision_embeddings = vision_embeddings.to(dtype=text_embeddings.dtype)
        return text_embeddings.masked_scatter(image_mask, vision_embeddings)


    def labels_from_input(self, inputs): 
        for input in inputs:
            for i in range(len(input) - 1, -1, -1):
                if input[i] == self.im_start:
                    input[:i] = -100
        return inputs
        

    def train_batch(self, batch,  image_size:int=512):
        image_inputs = self.preprocess_images(batch['image'], image_size)
        inputs = self.process_input(batch['conversation'], image_inputs)
    
        text_embeddings = self.get_input_embeddings()(inputs['input_ids'])
        vision_embeddings = self.get_image_features(image_inputs, True, False)
        input_embeds = self.merge_embeddings(vision_embeddings, text_embeddings, inputs['input_ids'])
        labels = self.labels_from_input(copy.deepcopy(inputs['input_ids']))
        
        return self.model.forward(inputs_embeds=input_embeds, attention_mask=inputs['attention_mask'], labels=labels)


if __name__ == '__main__': 
    # default: Load the model on the available device(s)
    from dataset import ConversationDataset
    dataset = ConversationDataset('/nethome/atena_projetos/fibz/images', '/nethome/atena_projetos/fibz/data/Dataset/simple_conversation/conv_test.json')
    loader = dataset.get_loader(2, False)

    model = FossilVL()
    
    model.train_epoch(loader, optim, 512)

    # print('input', input_embeds.shape)

    # position_ids, rope_deltas = model.model.get_rope_index(input_ids=inputs['input_ids'], image_grid_thw=inputs['image_grid_thw'], attention_mask=inputs['attention_mask'])

    # generated_ids = model.model.generate(
    #     input_ids=inputs['input_ids'],
    #     inputs_embeds=inputs_embeds,
    #     attention_mask=inputs['attention_mask'],
    #     # rope_deltas=rope_deltas,
    #     # position_ids=position_ids,
    #     max_new_tokens=128,
    #     # pixel_values=inputs['pixel_values'],
    #     # image_grid_thw=inputs['image_grid_thw']
    # )

    # generated_ids_trimmed = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # print(output_text)
