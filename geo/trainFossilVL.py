from torch.optim import AdamW
from dataset import ConversationDataset, DatasetBatchSampler, conversation_collate, DistributedDatasetBatchSampler
from model.fossilVL import FossilVL
from omegaconf import OmegaConf
import argparse
import torch
import os
import json
from tqdm import tqdm
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.state_dict import StateDictOptions



def train_epoch(model, optim, loader, epoch): 
    bar = not torch.accelerator.is_available() or torch.cuda.current_device() == 0
    if bar:
        pbar = tqdm(total=len(loader), desc=f"training progress epoch: {epoch}")

    epoch_loss = []
    for batch in loader:
        # print(batch.keys())
        optim.zero_grad()
        output = model(batch)
        output.loss.backward()
        optim.step()
                    
        epoch_loss.append(output.loss.detach().cpu().item())
        # print('loss', output.loss)
        if bar:
            pbar.update(1)
        
    return sum(epoch_loss)/len(epoch_loss)


def val_epoch(model, loader):
    epoch_loss = []
    bar = not torch.accelerator.is_available() or torch.cuda.current_device() == 0
    if bar:
        pbar = tqdm(total=len(loader), desc=f"validation progress")
    
    for batch in loader:
        with torch.no_grad():
            output = model(batch)
            epoch_loss.append(output.loss.detach().cpu().item())
            # print('loss', output.loss)
    
        if bar:
            pbar.update(1)
    
    return sum(epoch_loss)/len(epoch_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', '-c', type=str, default='geo/config/base.yaml')
    parser.add_argument('--save_path', '-s', type=str, default=None)
    parser.add_argument('--config', '-w', type=str, default=None)
    parser.add_argument('--campo', '-f', type=str, default=None)
    parser.add_argument('--average_local', default=False, action='store_true', help='average local patch embeddings')

    args = parser.parse_args()
    conf = OmegaConf.load(args.conf)
    
    if args.save_path is not None:
        conf.save_path = args.save_path

    if args.config is not None:
        conf.encoder.config_path = args.config

    conf.encoder.average_patch = args.average_local

    os.makedirs(conf.save_path, exist_ok=True)
    print(conf.save_path)
    OmegaConf.save(config=conf, f=os.path.join(conf.save_path, 'config.yaml'))
        
    # print(conf)
    if "LOCAL_RANK" in os.environ.keys():
        # using torchrun multigpu
        rank = int(os.environ["LOCAL_RANK"])
        if torch.accelerator.is_available():
            device_type = str(torch.accelerator.current_accelerator())
            device = torch.device(f"{device_type}:{rank}")
            torch.accelerator.set_device_index(rank)
            print(f"Running on rank {rank} on device {device}")

        else:
            device = torch.device("cpu")
            device_type = str(torch.accelerator.current_accelerator())
            print(f"Running on device {device}")

        backend = torch.distributed.get_default_backend_for_device(device)
        torch.distributed.init_process_group(backend=backend, device_id=device)

    if len(conf.encoder.size) == 1:
        train_dataset = ConversationDataset(
            conf.data.root, 
            conf.data.train, 
            groups=conf.data.groups if hasattr(conf.data, 'groups') else None,
            campo=args.campo,
            )

        val_dataset = ConversationDataset(
            conf.data.root, 
            conf.data.val,
            groups=conf.data.groups if hasattr(conf.data, 'groups') else None,
            campo=args.campo,
            )

        train_loader = train_dataset.get_loader(conversation_collate, conf.train.batch_size, True)
        val_loader = val_dataset.get_loader(conversation_collate, conf.train.batch_size, True)

    elif len(conf.encoder.size) == 2:
        # Multi resolution training and validation
        train_dataset1 = ConversationDataset(
            conf.data.root, 
            conf.data.train,
            groups=conf.data.groups,
            campo=args.campo,
            min_size=max(conf.encoder.size),
            )

        train_dataset2 = ConversationDataset(
            conf.data.root, 
            conf.data.train, 
            groups=conf.data.groups,
            campo=args.campo,
            min_size=min(conf.encoder.size),
            max_size=max(conf.encoder.size),
            )
        
        train_dataset = ConcatDataset([train_dataset1, train_dataset2])

        print(f'Len datasets: {len(train_dataset1)}, {len(train_dataset2)}, {len(train_dataset)}')
        val_dataset1 = ConversationDataset(
            conf.data.root, 
            conf.data.val,
            groups=conf.data.groups,
            campo=args.campo,
            min_size=min(conf.encoder.size),
            max_size=max(conf.encoder.size),
            )
        
        val_dataset2 = ConversationDataset(
            conf.data.root, 
            conf.data.val,
            groups=conf.data.groups,
            min_size=max(conf.encoder.size),
            campo=args.campo,
            )
        
        val_dataset = ConcatDataset([val_dataset1, val_dataset2])

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=DistributedDatasetBatchSampler(
                len1=len(train_dataset1),
                len2=len(train_dataset2),
                batch_size=conf.train.batch_size,
            ),
            collate_fn=conversation_collate,
        )
    
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=DistributedDatasetBatchSampler(
                len1=len(val_dataset1),
                len2=len(val_dataset2),
                batch_size=conf.train.batch_size,
            ),
            collate_fn=conversation_collate,
        )
                
    model = FossilVL(conf)
    # model.to(torch.bfloat16)

    if "LOCAL_RANK" in os.environ.keys():
        # using torchrun multi gpu
        fsdp_kwargs = {
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
            }
        model.fsdp(fsdp_kwargs)
    
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device_type = str(torch.accelerator.current_accelerator())
        model.decoder = model.decoder.to(device)
        model.encoder = model.encoder.to(device)
        model = model.to(device)

    optim = AdamW(model.parameters(), lr=conf.train.stage1.learning_rate)    
    log = {'training loss': [], 'validation loss': []}

    # STAGE 1
    for name, param in model.decoder.named_parameters():
        param.requires_grad = False

    for name, param in model.encoder.named_parameters():
        param.requires_grad = False
    
    epoch = 0
    print('STAGE 1')
    model.trainable_params()

    for i in range(conf.train.stage1.epochs):
        if len(conf.encoder.size) > 1:
            train_loader.batch_sampler.set_epoch(epoch)
            val_loader.batch_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, optim, train_loader, epoch)
        val_loss = val_epoch(model, val_loader)
        log['training loss'].append(train_loss)
        log['validation loss'].append(val_loss)
        epoch += 1

    # STAGE 2 PREP
    # TODO: add lora option to encoder
    for name, param in model.decoder.named_parameters():
        param.requires_grad = not conf.decoder.apply_lora

    for name, param in model.encoder.named_parameters():
        param.requires_grad = not conf.train.stage2.frozen_encoder


    # adding lora to a fsdp model will end in error if base model dtype if different from fsdp dtype, lora will use the same dtype as base model 
    if hasattr(model.decoder.model, "peft_config"):
        for name, param in model.decoder.named_parameters():
            param.requires_grad = False

        model.decoder.model.set_adapter("default") 
        model.decoder.model.enable_adapters()

    elif conf.decoder.apply_lora:
        model.decoder.apply_lora(conf)

    for group in optim.param_groups:
        group['lr'] = conf.train.stage2.learning_rate

    print('STAGE 2')
    model.trainable_params()
    
    # STAGE 2
    for i in range(conf.train.stage2.epochs):
        if len(conf.encoder.size) > 1:
            train_loader.batch_sampler.set_epoch(epoch)
            val_loader.batch_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, optim, train_loader, epoch)
        val_loss = val_epoch(model, val_loader)
        epoch += 1
        
        log['training loss'].append(train_loss)
        log['validation loss'].append(val_loss)

        # Log and chekpointing
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
        
        if val_loss <= min(log['validation loss']):
            torch.save(model_state_dict, os.path.join(conf.save_path, f'best_checkpoint.pt'))
            
        torch.save(model_state_dict, os.path.join(conf.save_path, f'last_checkpoint.pt'))
        