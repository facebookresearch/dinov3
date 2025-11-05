from torch.optim import AdamW
from dataset import ConversationDataset
from model.fossilVL import FossilVL
from omegaconf import OmegaConf
import argparse
import torch
import os
import json
from tqdm import tqdm
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.state_dict import StateDictOptions


def train_epoch(model, optim, loader, device): 
    bar = not torch.accelerator.is_available() or torch.cuda.current_device() == 0
    if bar:
        pbar = tqdm(total=len(loader), desc=f"Rank {rank} Progress")

    epoch_loss = []
    for batch in loader:
        # print(batch.keys())
        output = model(batch, device)
        output.loss.backward()
        optim.step()
        optim.zero_grad()
            
        epoch_loss.append(output.loss.detach().cpu().item())
        print('loss', output.loss)
        if bar:
            pbar.update(1)
        
    return sum(epoch_loss)/len(epoch_loss)


def val_epoch(model, loader, device):
    epoch_loss = []
    bar = device == 'cpu' or device.split(':')[-1] == '0'
    if bar:
        pbar = tqdm(total=len(loader), desc=f"Rank {rank} Progress")
    
    for batch in loader:
        with torch.no_grad():
            output = model(batch, device)
            epoch_loss.append(output.loss.detach().cpu().item())
            print('loss', output.loss)
    
        if bar:
            pbar.update(1)
    
    return sum(epoch_loss)/len(epoch_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', '-c', type=str, default='geo/config/base.yaml')
    args = parser.parse_args()
    conf = OmegaConf.load(args.conf)
    # print(conf)
    if "LOCAL_RANK" in os.environ.keys():
        # using torchrun multigpu
        rank = int(os.environ["LOCAL_RANK"])
        if torch.accelerator.is_available():
            device_type = torch.accelerator.current_accelerator()
            device = torch.device(f"{device_type}:{rank}")
            torch.accelerator.set_device_index(rank)
            print(f"Running on rank {rank} on device {device}")
        else:
            device = torch.device("cpu")
            print(f"Running on device {device}")

        backend = torch.distributed.get_default_backend_for_device(device)
        torch.distributed.init_process_group(backend=backend, device_id=device)

    train_dataset = ConversationDataset(conf.data.root, conf.data.train)
    test_dataset = ConversationDataset(conf.data.root, conf.data.test)
    train_loader = train_dataset.get_loader(conf.train.batch_size, True)
    test_loader = test_dataset.get_loader(conf.train.batch_size, True)

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            # param_dtype=torch.float16,
            reduce_dtype=torch.float32,
        )
    }

    model = FossilVL(conf)
    if "LOCAL_RANK" in os.environ.keys():
        # using torchrun multi gpu
        model.fsdp(fsdp_kwargs)
    # else:
    #     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model.decoder = model.decoder.to(device)
    model = model.to(device)
    # print(model)

    optim = AdamW(model.parameters(), lr=conf.train.stage1.learning_rate)    
    log = {'training loss': [], 'validation loss': []}
    
    # STAGE 1
    model.encoder.requires_grad = False
    model.decoder.requires_grad = False


    for i in range(conf.train.stage1.epochs):
        train_loss = train_epoch(model, optim, train_loader, device)
        val_loss = val_epoch(model, test_loader, device)
        log['training loss'].append(train_loss)
        log['validation loss'].append(val_loss)

    # STAGE 2
    model.encoder.requires_grad = not conf.train.stage2.frozen_encoder
    model.decoder.requires_grad = True
    for group in optim.param_groups:
        group['lr'] = conf.train.stage2.learning_rate
    
    for i in range(conf.train.stage2.epochs):
        train_loss = train_epoch(model, optim, train_loader, device)
        val_loss = val_epoch(model, test_loader, device)
        
        log['training loss'].append(train_loss)
        log['validation loss'].append(val_loss)

    os.makedirs(conf.save_path, exist_ok=True)
    with open(os.path.join(conf.save_path, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2) 

    if "LOCAL_RANK" in os.environ.keys():
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
    
    else:
        model_state_dict = model.model.state_dict()
        
    torch.save(model_state_dict, os.path.join(conf.save_path, 'checkpoint.pt'))
    OmegaConf.save(config=conf, f=os.path.join(conf.save_path, 'config.yaml'))
      