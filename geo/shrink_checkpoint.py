"""
Shrinks full FossilVL checkpoints (every weight, as trainFossilVL.py and train_fossilvl_grpo.py wrote them before) to
the format of model/fossilVL.save_checkpoint, which load_checkpoint reads.

A weight is dropped only when it is identical to the one FossilVL(conf) builds from the pretrained encoder/decoder (so
loading the small checkpoint on top of a fresh model gives back exactly the full one; a randomly initialised weight
never matches and is kept). The projection, scale embedding and LoRA adapters are always kept. The small checkpoints need the same
pretrained weights (paths in config.yaml) to load. Needs memory for about two copies of the model.

Without --in_place the small checkpoint is written next to the full one as <name>.small.pt.

Usage:
    python geo/shrink_checkpoint.py --model <output folder> --ckpt best last
    python geo/shrink_checkpoint.py --model <output folder> --ckpt best last --in_place
"""
import os
import sys
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.fossilVL import CHECKPOINT_FORMAT, TRAINED_MODULES, FossilVL  # noqa: E402


def size_mb(state_dict):
    return sum(v.numel() * v.element_size() for v in state_dict.values()) / 2 ** 20


def shrink(base, path, in_place):
    full = torch.load(path, map_location='cpu')
    if isinstance(full, dict) and full.get('format') == CHECKPOINT_FORMAT:
        print(f'{path}: already small, skipped')
        return
    if set(full) != set(base):
        raise RuntimeError(f'{path} does not match the model built from config.yaml: '
                           f'{len(set(full) - set(base))} unexpected and {len(set(base) - set(full))} missing keys')

    keep = {k: v for k, v in full.items()
            if 'lora_' in k or k.startswith(TRAINED_MODULES) or v.dtype != base[k].dtype or not torch.equal(v, base[k])}

    out = path if in_place else path[:-len('.pt')] + '.small.pt'
    tmp = out + '.tmp'
    torch.save({'format': CHECKPOINT_FORMAT, 'state_dict': keep}, tmp)
    os.replace(tmp, out)
    print(f'{path}: kept {len(keep)} of {len(full)} tensors, {size_mb(full):,.0f} MB -> {size_mb(keep):,.0f} MB, '
          f'saved to {out}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', required=True, help='output folder with config.yaml and <ckpt>_checkpoint.pt')
    parser.add_argument('--ckpt', nargs='+', default=['best', 'last'])
    parser.add_argument('--in_place', action='store_true', help='replace the full checkpoints')
    args = parser.parse_args()

    conf = OmegaConf.load(os.path.join(args.model, 'config.yaml'))
    model = FossilVL(conf)  # as load_checkpoint's callers build it
    if conf.decoder.apply_lora and not hasattr(model.decoder.model, 'peft_config'):
        model.decoder.apply_lora(conf)
    base = model.state_dict()

    for name in args.ckpt:
        path = os.path.join(args.model, f'{name}_checkpoint.pt')
        if not os.path.exists(path):
            print(f'{path}: not found, skipped')
            continue
        shrink(base, path, args.in_place)
