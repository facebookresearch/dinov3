import torch
from PIL import Image
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
from omegaconf import OmegaConf
sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
from dataset import ConversationDataset


class Decoder(ABC):
    @abstractmethod
    def __init__(self, conf):
        pass

    @abstractmethod
    def get_embedding_layer(self, ):
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

def decoder_factory(conf):
    supported_models = {'Qwen3': Qwen3}
    if conf.decoder.name in supported_models.keys():
        return supported_models[conf.decoder.name](conf) 
    

class Qwen3(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(conf.decoder.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            conf.decoder.name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.dim = self.model.model.embed_tokens.weight.shape[1]


    def get_embedding_layer(self, ):
        return self.model.get_input_embeddings()

    def merge_inputs(self, vision_embeddings, text_embeddings, input_ids):
        padding = 151643
        # print(vision_embeddings.shape)
        # print(text_embeddings.shape)
        # embeddings merge
        first_part = text_embeddings[:, :4, :]
        second_part = text_embeddings[:, 4:, :]
        embeddings = torch.concat((first_part, vision_embeddings, second_part), dim=1)
        
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
            'input_embeddings': embeddings.to(self.model.device, dtype=self.model.dtype),
            'attention_mask' : attention_mask.to(self.model.device),    
            'labels': labels.to(self.model.device),

            }

    def prepare_inputs(self, conversations, add_gen_prompt=False):
        texts = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False, 
        )
        vl_text = []
        for text in texts:
            vl_text.append(text.replace('<|im_start|>user', 
                                        '<|im_start|>user\n<|vision_start|><|vision_end|>'))
        inputs = self.tokenizer(vl_text, return_tensors="pt", padding=True).to(self.model.device)
        # print(inputs)
        return inputs['input_ids']
        
    def forward(self, inputs ):
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
        text_embeddings = self.get_embedding_layer()(inputs)
        model_inputs = self.merge_inputs(image_embeddings, text_embeddings, inputs)

        generated_ids = self.model.generate(
            inputs_embeds=model_inputs['input_embeddings'],
            attention_mask=model_inputs['attention_mask'],
            max_new_tokens=32768,

        )
        output_ids = generated_ids[0][len(inputs[0]):].tolist() 
        print(output_ids)
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
    sys.path.append(os.path.normpath(os.path.join(__file__, '../../')))
    from dataset import ConversationDataset
    
    conf = OmegaConf.load('geo/config/base.yaml')
    model = Qwen3(conf)

    test_dataset = ConversationDataset(conf.data.root, conf.data.test)
    test_loader = test_dataset.get_loader(conf.train.batch_size, True)
    
    # for batch in test_loader:
    #     input = model.prepare_inputs(batch['conversation'])
    #     text = model.get_embedding_layer()(input)
    #     vision = torch.rand((conf.train.batch_size, 64, model.dim))
    #     inputs = model.merge_inputs(vision, text, input)
    #     output = model(inputs)
    #     # print(output)
    #     break


    output = model.generate(torch.rand((1, 64, model.dim)), 'descreva a imagem')
    print(output['response'])