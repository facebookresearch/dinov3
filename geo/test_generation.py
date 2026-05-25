from omegaconf import OmegaConf
from model.fossilVL import FossilVL
import torch
import os 
import json
from argparse import ArgumentParser
from dataset import ConversationDataset
from tqdm import tqdm

if __name__ == '__main__':
    model_path = '/nethome/recpinfo/users/fibz/data/checkpoints/fossil/nwpu/dinov3-qwen3-0.6B-longtrain'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(os.path.join(model_path, 'config.yaml'))
    model = FossilVL(conf)
    if conf.decoder.apply_lora:
        model.decoder.apply_lora(conf)

    ckpt = torch.load(os.path.join(model_path, 'checkpoint.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    model.to(device)

    model.eval()

    test_dataset = ConversationDataset(conf.data.root, conf.data.test)
    test_loader = test_dataset.get_loader(1, True)

    results = {'generated': [], }

    for batch in test_loader:
        with torch.no_grad():
            # FORWARD DO FOSSIL
            image_tensors = model.encoder.get_image_tensors(batch['image'], ).to(device)
            image_embeddings = model.encoder(image_tensors, return_grid=model.use_grid)
            image_embeddings = model.projection(image_embeddings)
            
            inputs = model.decoder.prepare_inputs(batch['conversation']).to(device)
            print(inputs)
            
            text_embeddings = model.decoder.get_input_embeds(inputs)
            model_inputs = model.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
            print(model_inputs['input_embeddings'].shape, model_inputs['attention_mask'].shape, model_inputs['labels'].shape, inputs.shape)

            print('\n\n\n\n')
            # GENERATE DO DECODER
            prompt = "Provide a concise caption of this satellite image"
            image_tensors = model.encoder.get_image_tensors(batch['image']).to(device)
            image_embeddings = model.encoder(image_tensors, return_grid=model.use_grid)
            image_embeddings = model.projection(image_embeddings)

            messages = [
                {"role": "user", "content": prompt}
            ]

            inputs = model.decoder.prepare_inputs([messages], add_gen_prompt=True).to(device)
            print(inputs)            
            text_embeddings = model.decoder.get_input_embeds(inputs)
            model_inputs = model.decoder.merge_inputs(image_embeddings, text_embeddings, inputs)
            print(model_inputs['input_embeddings'].shape, model_inputs['attention_mask'].shape, model_inputs['labels'].shape, inputs.shape)

        break

    # print(results)
    # json.dump(results, open(os.path.join(model_path, 'generated_captions.json'), 'w'), indent=2)
