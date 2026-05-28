import torch
from PIL import Image
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import fully_shard
sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
from dataset import ConversationDataset
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard

def decoder_factory(conf):
    if 'qwen3' in conf.decoder.name.lower():
        return Qwen3(conf)
    
    elif 'llama-3.2' in conf.decoder.name.lower():
        return Llama3(conf)
    
    else:
        ValueError(f'{conf.decoder.name} is not supported')


class Qwen3(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(conf.decoder.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            conf.decoder.name,
            dtype="float",
            device_map="cpu"
        )
        self.dim = self.model.model.embed_tokens.weight.shape[1]
        self.peft = False
        self.tokenizer.pad_token_id = 151643
        self.think_end = 151668
        self.im_start = 151644
        
        
    def apply_lora(self, conf):
        if conf.decoder.apply_lora:
            lora_config = LoraConfig(
                r=conf.decoder.lora_rank,
                lora_alpha=conf.decoder.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.peft = True

    def fsdp(self, fsdp_kwargs):
        if not self.peft:
            for block in self.model.model.layers:
                fully_shard(block, **fsdp_kwargs)
        else:
            for block in self.model.base_model.model.model.layers:
                fully_shard(block, **fsdp_kwargs)
          

    def get_input_embeds(self, inputs):
        device = self.model.model.embed_tokens.weight.device
        if self.peft:
            return self.model.base_model.model.model.embed_tokens(inputs.to(device))
        
        return self.model.model.embed_tokens(inputs.to(device))

    def merge_inputs(self, vision_embeddings, text_embeddings, input_ids):
        # TODO: find dinamically where to split based on token ids and append image
        first_part = text_embeddings[:, :4, :]
        second_part = text_embeddings[:, 4:, :]

        embeddings = torch.concat((first_part, vision_embeddings , second_part), dim=1)
        print('input ids', input_ids)
        print('think end', self.think_end)
    
        # generation start, split is used to create labels
        split = (input_ids[0] == self.think_end).nonzero(as_tuple=True)[-1] + 2 
        print('SPLIT', split)
        # labels
        labels = torch.ones(embeddings.shape[:2]) * -100
        labels[:, split + vision_embeddings.shape[1]:] = input_ids[:, split:]
        labels[labels == self.tokenizer.pad_token] = -100
        labels = labels.to(dtype=torch.long)
        attention_mask = torch.ones_like(labels)
        
        return {
            'input_embeddings': embeddings,
            'attention_mask' : attention_mask,    
            'labels': labels,
            }


    def prepare_inputs(self, conversations, add_gen_prompt=False):
        texts = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
            enable_thinking=False, 
        )
        # print(texts)
        vl_text = []
        for text in texts:
            vl_text.append(text.replace('<|im_start|>user', 
                                        '<|im_start|>user\n<|vision_start|><|vision_end|>'))
        
        inputs = self.tokenizer(vl_text, return_tensors="pt", padding=True)
        # print(inputs)
        return inputs['input_ids']
        
    def forward(self, inputs ):
        device = self.model.model.embed_tokens.weight.device
    
        return self.model.forward(
            inputs_embeds=inputs['input_embeddings'].to(device), 
            labels=inputs['labels'].to(device), 
            attention_mask=inputs['attention_mask'].to(device),
            
            )

    def generate(self, model_inputs, num_beams, do_sample):
        generated_ids = self.model.generate(
            inputs_embeds=model_inputs['input_embeddings'].to(dtype=self.model.dtype, device=self.model.device),
            attention_mask=model_inputs['attention_mask'].to(dtype=self.model.dtype, device=self.model.device),
            max_new_tokens=100,
        )
        generated = self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True, 
            do_sample=do_sample, 
            num_beams=num_beams,
            
            ).strip("\n")
        return generated
        

class Llama3(Qwen3):
    def __init__(self, conf):
        super().__init__(conf)
        self.tokenizer.pad_token_id = 128004
        self.model.generation_config.pad_token_id = 128004 
        self.im_start = 128007 # not necessarily the im start (some models dont have it) used to define what to mask during loss computation
        self.image = 128256 # used to find where to put image tokens
        if len(self.tokenizer) <= 128256:
            special_tokens_dict = {'additional_special_tokens': ['<|image|>']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            


    def prepare_inputs(self, conversations, add_gen_prompt=False):
        vl_prompt = "<|image|><|begin_of_text|>"
        for conversation in conversations:
            for message in conversation:
                if message['role'] == 'user':
                    message['content'] = vl_prompt + message['content']
        
        # print(conversations)
        
        texts = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
            enable_thinking=False, 
        )
    
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        
        return inputs['input_ids']
    

    def merge_inputs(self, vision_embeddings, text_embeddings, input_ids):
        indices = (input_ids == self.image).nonzero() 
        index = indices[0][-1] + 1
        first_part = text_embeddings[:, :index, :]
        second_part = text_embeddings[:, index:, :]

        embeddings = torch.concat((first_part, vision_embeddings , second_part), dim=1)
        
        labels = torch.ones(input_ids.shape) * -100
        
        # where to mask during loss computation (-100) 
        split = (input_ids == self.im_start).nonzero(as_tuple=True)
        for i in range(input_ids.shape[0]):
            inner_indices = (split[0] == i).nonzero(as_tuple=True)
            cut = split[1][inner_indices[0].argmax()] + 1
            labels[i, cut:] = input_ids[i, cut:]
            
        labels[labels == self.tokenizer.pad_token] = -100
        
        # concat -100 for visual tokens
        vis_mask = torch.ones(vision_embeddings.shape[:2]) * -100
        labels = torch.concat([vis_mask, labels], dim=1)
       
        labels = labels.to(dtype=torch.long)
        attention_mask = torch.ones_like(labels)
       
        return {
            'input_embeddings': embeddings,
            'attention_mask' : attention_mask,    
            'labels': labels,

            }