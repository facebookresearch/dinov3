from torch.optim import AdamW
from dataset import ConversationDataset
from geo.model.qwenVL import FossilVL
from omegaconf import OmegaConf
import argparse
import torch
import os
import json
from tqdm import tqdm

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


def train_epoch(model, optim, loader):
    epoch_loss = []
    for batch in tqdm(loader, desc='training'):
        # print(batch.keys())
        optim.zero_grad()
        images = model.preprocess_images(batch['image'], model.image_dim).to(device)
        output = model(images, batch['conversation'])
        output.loss.backward()
        optim.step()
        epoch_loss.append(output.loss.detach().cpu().item())

    return sum(epoch_loss)/len(epoch_loss)

def val_epoch(model, loader):
    epoch_loss = []
    for batch in tqdm(loader, desc='validation'):
        with torch.no_grad():
            images = model.preprocess_images(batch['image'], model.image_dim).to(device)
            output = model(images, batch['conversation'])
            epoch_loss.append(output.loss.detach().cpu().item())
        

    return sum(epoch_loss)/len(epoch_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', '-c', type=str, default='geo/config/base.yaml')
    args = parser.parse_args()
    conf = OmegaConf.load(args.conf)
    # print(conf)

    train_dataset = ConversationDataset(conf.data.root, conf.data.train)
    test_dataset = ConversationDataset(conf.data.root, conf.data.test)
    train_loader = train_dataset.get_loader(conf.train.batch_size, True)
    test_loader = test_dataset.get_loader(conf.train.batch_size, True)

    model = FossilVL(conf)
    model.to(device)
    optim = AdamW(model.parameters(), lr=conf.train.stage1.learning_rate)    
    log = {'training loss': [], 'validation loss': []}
    
    # STAGE 1
    model.model.visual.requires_grad = False
    model.model.language_model.requires_grad = False
        
    for i in range(conf.train.stage1.epochs):
        train_loss = train_epoch(model, optim, train_loader)
        val_loss = val_epoch(model, test_loader)
        log['training loss'].append(train_loss)
        log['validation loss'].append(val_loss)

    # STAGE 2
    model.model.visual.requires_grad = True
    model.model.language_model.requires_grad = True
    for group in optim.param_groups:
        group['lr'] = conf.train.stage2.learning_rate
    
    for i in range(conf.train.stage2.epochs):
        train_loss = train_epoch(model, optim, train_loader)
        val_loss = val_epoch(model, test_loader)
        
        log['training loss'].append(train_loss)
        log['validation loss'].append(val_loss)

    os.makedirs(conf.save_path, exist_ok=True)
    with open(os.path.join(conf.save_path, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(conf.save_path, 'checkpoint.pt'))

    OmegaConf.save(config=conf, f=os.path.join(conf.save_path, 'config.yaml'))
      