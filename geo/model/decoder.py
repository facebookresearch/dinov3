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


class Decoder(ABC):
    @abstractmethod
    def __init__(self, conf):
        pass

    @abstractmethod
    def get_input_embeds(self, inputs):
        pass
    
    @abstractmethod
    def merge_inputs(self, vision_embeddings, text_embeddings, input_ids):
        pass

    @abstractmethod
    def prepare_inputs(self, conversation): 
        pass        

    @abstractmethod
    def forward(self, image, conversation,  image_size:int=512):
        pass

    @abstractmethod
    def generate(self, image, ):
        pass

    @abstractmethod
    def fsdp(self,):
        pass

def decoder_factory(conf):
    if 'Qwen3' in conf.decoder.name:
        return Qwen3(conf)
    else:
        ValueError(f'{conf.decoder.name} is not supported')

class Qwen3(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(conf.decoder.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            conf.decoder.name,
            dtype="auto",
            device_map="cpu"
        )
        self.dim = self.model.model.embed_tokens.weight.shape[1]
        self.peft = False

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
        if self.peft:
            return self.model.base_model.model.model.embed_tokens(inputs)
        
        return self.model.model.embed_tokens(inputs)

    def merge_inputs(self, vision_embeddings, text_embeddings, input_ids):
        padding = 151643
        
        first_part = text_embeddings[:, :4, :]
        second_part = text_embeddings[:, 4:, :]

        embeddings = torch.concat((first_part, vision_embeddings , second_part), dim=1)
        
        # generation start
        think_end = 151668
        im_start = 151644
        if think_end in input_ids[0]:
            split = (input_ids[0] == think_end).nonzero(as_tuple=True)[-1] + 2 
        else:
            split = (input_ids[0] == im_start).nonzero(as_tuple=True)[-1] + 2
        
        # labels
        labels = torch.ones(embeddings.shape[:2]) * -100
        labels[:, split + vision_embeddings.shape[1]:] = input_ids[:, split:]
        labels[labels == padding] = -100
        labels = labels.to(dtype=torch.long)
        attention_mask = torch.ones_like(labels)
        
        return {
            'input_embeddings': embeddings,
            'attention_mask' : attention_mask,    
            'labels': labels,

            }
        # return {
        #     'input_embeddings': embeddings.to(self.model.device, dtype=self.model.dtype),
        #     'attention_mask' : attention_mask.to(self.model.device),    
        #     'labels': labels.to(self.model.device),

        #     }



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
        # if "LOCAL_RANK" not in os.environ.keys():
        #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # else:
        #     index = torch.accelerator.current_device_index()
        #     device = f'cuda:{index}'
        # print(self.model.device, device, inputs['input_embeddings'].device)
        
        return self.model.forward(
            inputs_embeds=inputs['input_embeddings'], 
            labels=inputs['labels'], 
            attention_mask=inputs['attention_mask'],
            
            )

    def generate(self, image_embeddings, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        inputs = self.prepare_inputs([messages], add_gen_prompt=True)
        # print(inputs.shape)
        text_embeddings = self.get_input_embeds(inputs)
        model_inputs = self.merge_inputs(image_embeddings, text_embeddings, inputs)

        generated_ids = self.model.generate(
            inputs_embeds=model_inputs['input_embeddings'],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=32768,

        )
        output_ids = generated_ids[0][len(inputs[0]):].tolist() 
        # print(output_ids)
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return {'thinking': thinking_content, 'response': content}

        


if __name__ == '__main__':
    from torch.distributed.fsdp import MixedPrecisionPolicy
    sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
    from dataset import ConversationDataset
    
    conf = OmegaConf.load('geo/config/base.yaml')
    model = Qwen3(conf)
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }
    model.fsdp(fsdp_kwargs)
    print(model)
    # test_dataset = ConversationDataset(conf.data.root, conf.data.test)
    # test_loader = test_dataset.get_loader(conf.train.batch_size, True)
    
    # for batch in test_loader:
    #     input = model.prepare_inputs(batch['conversation'])
    #     text = model.get_embedding_layer()(input)
    #     vision = torch.rand((conf.train.batch_size, 64, model.dim))
    #     inputs = model.merge_inputs(vision, text, input)
    #     output = model(inputs)
    #     # print(output)
    #     break


    # output = model.generate(torch.rand((1, 64, model.dim)), 'descreva a imagem')
    # print(output['response'])