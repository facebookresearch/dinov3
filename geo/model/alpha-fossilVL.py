import torch
from transformers import AutoProcessor
import copy
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraModel, LoraConfig, get_peft_model
from geo.model.dino_utils import create_and_load_model
from geo.model.projector import Qwen2_5_VLPatchMerger


class FossilVLAlpha(torch.nn.Module):
    def __init__(self, conf, ):
        super().__init__()
        self.decoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(conf.decoder.name, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(conf.decoder.name, patch_size=16, ) #temporal_patch_size=1, merge_size=1)
        self.im_start = 151644
        
        self.encoder = create_and_load_model(conf.encoder.config_path, conf.encoder.weight_path)
        encoder_dim = self.encoder.patch_embed.proj.weight.shape[0]
        decoder_dim = self.decoder.language_model.embed_tokens.weight.shape[1]
        
        
        self.patch_merger = Qwen2_5_VLPatchMerger(decoder_dim, encoder_dim, 2)
        self.image_dim = conf.encoder.dim

        if conf.decoder.apply_lora:
            lora_config = LoraConfig(
                r=conf.decoder.lora_rank, 
                lora_alpha=conf.decoder.lora_alpha, 
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], 
                lora_dropout=0.1, 
                bias="none", 
                task_type="CAUSAL_LM"
            )
            self.model.language_model = get_peft_model(self.model.language_model, lora_config)

    def process_input(self, conversations: list, image_inputs: torch.Tensor, add_gen: bool):
        texts = []
        for conversation in conversations:
            texts.append(self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=add_gen))
        # print(texts[-1])
        return self.processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

    def resize_transform(self, mask_image: Image, image_size: int = 512, patch_size: int = 16,) -> torch.Tensor:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        w, h = mask_image.size
        if w < h:
            w_patches = int(image_size / patch_size)
            h_patches = int((h * image_size) / (w * patch_size))
            
        else:
            h_patches = int(image_size / patch_size)
            w_patches = int((w * image_size) / (h * patch_size))
        
        input = TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))
        input = TF.normalize(input, mean=mean, std=std)
        c, w, h = input.shape
        center = (int(w / 2), int(h/2))
        return input[:, 
                     center[0]-int(image_size/2):center[0]+int(image_size/2), 
                     center[1]-int(image_size/2):center[1]+int(image_size/2)]
    
    def preprocess_images(self, images: list, image_size):
        inputs = []
  
        for image in images:
            if type(image) == str:
                image = Image.open(image).convert('RGB')

            tensor = self.resize_transform(image, image_size=image_size)
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


    def merge_embeddings(self, vision_embeddings, text_embeddings, input_ids):
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=text_embeddings,
            image_features=vision_embeddings,
        )

        vision_embeddings = vision_embeddings.to(dtype=text_embeddings.dtype)
        return text_embeddings.masked_scatter(image_mask, vision_embeddings)


    def labels_from_input(self, inputs): 
        for input in inputs:
            for i in range(len(input) - 1, -1, -1):
                if input[i] == self.im_start:
                    input[:i+1] = -100
                    break
        return inputs
        

    def forward(self, image, conversation,  image_size:int=512):
        inputs = self.process_input(conversation, image, False)
        text_embeddings = self.get_input_embeddings()(inputs['input_ids'])
        vision_embeddings = self.get_image_features(image, True, False)
        input_embeds = self.merge_embeddings(vision_embeddings, text_embeddings, inputs['input_ids'])
        labels = self.labels_from_input(copy.deepcopy(inputs['input_ids']))
        
        print('text_embeddings', text_embeddings.shape)
        print('vision_embeddings', vision_embeddings.shape)
        print('input_embeddings', input_embeds.shape)

        return self.model.forward(inputs_embeds=input_embeds, attention_mask=inputs['attention_mask'], labels=labels)


    def generate(self, image, ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Descreva as características da rocha de acordo com a imagem microscópica."},
                ],
            }
        ]
        image_inputs = self.preprocess_images([image], self.image_dim)
        inputs = self.process_input([messages], image_inputs, True)
        
        text_embeddings = self.get_input_embeddings()(inputs['input_ids'])
        vision_embeddings = self.get_image_features(image_inputs, True, False)
        input_embeds = self.merge_embeddings(vision_embeddings, text_embeddings, inputs['input_ids'])
        
        generated_ids = self.model.generate(
            input_ids=inputs['input_ids'],
            inputs_embeds=input_embeds,
            attention_mask=inputs['attention_mask'],
            max_new_tokens=128,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


        