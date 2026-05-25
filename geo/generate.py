from omegaconf import OmegaConf
from model.fossilVL import FossilVL
import torch
import os 
import json
from argparse import ArgumentParser
from dataset import ConversationDataset
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', help='path to model output folder', required=True)
    parser.add_argument('--split',  help='split to load', required=True, choices=['test', 'train'])
    parser.add_argument('--max', default=None, help='number of captions to generate', type=int)
    parser.add_argument('--ckpt', choices=['best', 'last'], help='checkpoint to load', required=True)

    args = parser.parse_args()
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(os.path.join(args.model, 'config.yaml'))
    model = FossilVL(conf)
    if conf.decoder.apply_lora:
        model.decoder.apply_lora(conf)

    print('loading model: {}'.format(os.path.join(args.model, f'{args.ckpt}_checkpoint.pt')))
    ckpt = torch.load(os.path.join(args.model, f'{args.ckpt}_checkpoint.pt'), map_location=torch.device('cpu'))
    print(ckpt.keys())
    model.load_state_dict(ckpt)
    model.to(device)

    model.eval()
    split =  conf.data.test if args.split == 'test' else conf.data.train
    dataset = ConversationDataset(conf.data.root, split.replace('_all', ''))
    loader = dataset.get_loader(1, False)

    target_dataset = json.load(open(split.replace('conversation_', '').replace('_all', ''), 'r'))
    # print(split.replace('conversation_', '').replace('_all', ''))
    
    if args.max == None:
        args.max = len(dataset)
        
    results = {'generated': [], }
    i = 0
    for batch in tqdm(loader):
        if args.max is not None and i >= args.max:
            break

        image = batch['image'][0]
        prompt = batch['conversation'][0][0]['content']
        target = target_dataset[i]["captions"]
        output = model.generate([image], prompt)
        results['generated'].append({'reference': target, 'prediction': output})
        i += 1

    print(results)
    json.dump(results, open(os.path.join(args.model, f'generated_captions_{args.split}_{args.max}_{args.ckpt}.json'), 'w'), indent=2)