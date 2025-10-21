import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor
import abc
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image



if __name__ == '__main__':
 

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    # print(model)
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    im_name = "9A5EB9C53CF3588CE05351EB1D0A5F08.png"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"/nethome/atena_projetos/fibz/data/Dataset/images/{im_name}",
                },
                {"type": "text", "text": "Descreva a imagem em portugues."},
            ],
        },
    ]

    # # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",

    )
    
    # inputs = inputs.to("cuda")
    text_embeddings = model.get_input_embeddings()(inputs['input_ids'])
    vision_embeddings = model.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'])

    print(vision_embeddings[0].shape)
    print(inputs['image_grid_thw'])

    # merge text and image embeddings
    vision_embeddings = torch.cat(vision_embeddings, dim=0)
    model.model.visual = None
    image_mask, _ = model.model.get_placeholder_mask(
        inputs['input_ids'],
        inputs_embeds=text_embeddings,
        image_features=vision_embeddings,
    )
    inputs_embeds = text_embeddings.masked_scatter(image_mask, vision_embeddings)

    print('text_embeddings: ', text_embeddings.shape)
    print('vision_embeddings: ', vision_embeddings.shape)
    print('merge mask shape: ', image_mask.shape)
    print('input embeddings shape', inputs_embeds.shape)
    print('attention mask shape', inputs['attention_mask'].shape)
    # print('position id shape', position_ids.shape)

    generated_ids = model.generate(
        input_ids=inputs['input_ids'],
        inputs_embeds=inputs_embeds,
        attention_mask=inputs['attention_mask'],
        # rope_deltas=rope_deltas,
        # position_ids=position_ids,
        max_new_tokens=128,
        # pixel_values=inputs['pixel_values'],
        # image_grid_thw=inputs['image_grid_thw']
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
