from omegaconf import OmegaConf
from model.fossilVL import FossilVL
import torch
import os 
import json
from argparse import ArgumentParser
from dataset import ConversationDataset, conversation_collate
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', help='path to model output folder', required=True)
    parser.add_argument('--split',  help='split to load', required=True, choices=['test', 'train', 'val'])
    parser.add_argument('--max', default=None, help='number of captions to generate', type=int)
    parser.add_argument('--ckpt', choices=['best', 'last'], help='checkpoint to load', required=True)
    parser.add_argument('--campo', default=None, choices=['classification', 'texture', 'porosity', 'diagenesis', 'composition'])

    args = parser.parse_args()
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(os.path.join(args.model, 'config.yaml'))
    model = FossilVL(conf)
    model.to(torch.float32)

    if not hasattr(model.decoder.model, "peft_config") and conf.decoder.apply_lora:
        model.decoder.apply_lora(conf)
    
    print('loading model: {}'.format(os.path.join(args.model, f'{args.ckpt}_checkpoint.pt')))
    ckpt = torch.load(os.path.join(args.model, f'{args.ckpt}_checkpoint.pt'), map_location=torch.device('cpu'))

    model.load_state_dict(ckpt)
    model.to(device)

    model.eval()
    split = {'test': conf.data.test, 'train': conf.data.train, 'val': conf.data.val}[args.split]
    

    dataset = ConversationDataset(
        conf.data.root, 
        split,
        groups=conf.data.groups if hasattr(conf.data, 'groups') else None,
        min_size=max(conf.encoder.size),
        campo=args.campo,
        )
    
    loader = dataset.get_loader(conversation_collate, 1, False)

    if args.max == None:
        args.max = len(dataset)
        
    results = {'generated': [], }
    i = 0
    for batch in tqdm(loader):
        if args.max is not None and i >= args.max:
            break
        
        image = batch['image'][0]
        prompt = batch['conversation'][0][0]['content']
        target = batch['conversation'][0][1]['content']
        output = model.generate([image], prompt)
        results['generated'].append({'reference': target, 'prediction': output, 'prompt': prompt, 'image': image})

        i += 1

    print(results)
    campo = args.campo if args.campo is not None else "all"
    json.dump(results, open(os.path.join(args.model, f'generated_captions_{args.split}_{campo}_{args.ckpt}.json'), 'w'), indent=2)