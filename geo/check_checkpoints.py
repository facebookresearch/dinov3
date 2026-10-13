"""
Checks whether the parts of FossilVL checkpoints were actually trained.

From the checkpoint alone:
    LoRA adapters  lora_B starts at zero, so an all-zero lora_B was never trained. For trained adapters the update
                   scale * B @ A is reported, relative to the frozen layer weight when the checkpoint has it (full
                   checkpoints), with scale = lora_alpha / lora_rank from config.yaml.
    sizes          tensors and MB per part (encoder, decoder base, decoder LoRA, projection), so full and small
                   checkpoints (model/fossilVL.save_checkpoint) can be told apart.
With --compare_base the model is built from config.yaml as the loaders do (pretrained encoder/decoder, LoRA applied)
and every encoder/decoder weight is compared with the pretrained one; this needs memory for about two models.
The projection starts from a random init, so there is nothing to compare it with.

Usage:
    python geo/check_checkpoints.py <output folder> [<output folder> ...] [--ckpt best last] [--compare_base]
    python geo/check_checkpoints.py /path/to/checkpoints/*/ --ckpt best
"""
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.fossilVL import CHECKPOINT_FORMAT  # noqa: E402

PARTS = ('encoder', 'decoder (base)', 'decoder (LoRA)', 'projection', 'other')


def part_of(key):
    if 'lora_' in key:
        return 'decoder (LoRA)' if key.startswith('decoder.') else 'encoder'
    for prefix in ('encoder', 'decoder', 'projection'):
        if key.startswith(prefix + '.'):
            return 'decoder (base)' if prefix == 'decoder' else prefix
    return 'other'


def load_state_dict(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and ckpt.get('format') == CHECKPOINT_FORMAT:
        return ckpt['state_dict'], 'small'
    return ckpt, 'full'


def lora_report(state_dict, conf):
    """One entry per adapted layer: is lora_B zero, and the size of the update scale * B @ A."""
    rank = conf.decoder.get('lora_rank') if 'decoder' in conf else None
    alpha = conf.decoder.get('lora_alpha') if 'decoder' in conf else None
    scale = alpha / rank if rank and alpha else 1.0
    layers = []
    for key_b, b in state_dict.items():
        if '.lora_B.' not in key_b or not key_b.startswith('decoder.'):
            continue
        a = state_dict.get(key_b.replace('.lora_B.', '.lora_A.'))
        layer = key_b.split('.lora_B.')[0]
        base = state_dict.get(layer + '.base_layer.weight')
        entry = {'layer': layer, 'zero': not bool(b.abs().max()), 'max_abs_B': float(b.abs().max())}
        if a is not None:
            delta = scale * b.float() @ a.float()
            entry['update_norm'] = float(delta.norm())
            if base is not None:
                entry['relative_update'] = float(delta.norm() / base.float().norm())
        layers.append(entry)
    return layers


def base_report(state_dict, base):
    """Per part: tensors compared, tensors that differ from the pretrained ones and the largest relative change."""
    out = defaultdict(lambda: {'compared': 0, 'changed': 0, 'max_relative_change': 0.0})
    for key, value in state_dict.items():
        part = part_of(key)
        if key not in base or 'lora_' in key or part in ('projection', 'other'):
            continue
        ref = base[key]
        stats = out[part]
        stats['compared'] += 1
        if value.shape != ref.shape or not torch.equal(value.to(ref.dtype), ref):
            stats['changed'] += 1
            if value.shape == ref.shape and value.is_floating_point():
                change = float((value.float() - ref.float()).norm() / (ref.float().norm() + 1e-12))
                stats['max_relative_change'] = max(stats['max_relative_change'], change)
    return dict(out)


def build_base(conf):
    from model.fossilVL import FossilVL
    model = FossilVL(conf)
    if conf.decoder.apply_lora:
        model.decoder.apply_lora(conf)
    return model.state_dict()


def check(path, conf, base=None):
    state_dict, kind = load_state_dict(path)
    print(f'\n=== {path} ({kind} checkpoint)')
    sizes = defaultdict(lambda: [0, 0.0])
    for key, value in state_dict.items():
        sizes[part_of(key)][0] += 1
        sizes[part_of(key)][1] += value.numel() * value.element_size() / 2 ** 20
    for part in PARTS:
        if part in sizes:
            print(f'  {part:15s} {sizes[part][0]:6d} tensors {sizes[part][1]:10,.1f} MB')

    projection = [v.float() for k, v in state_dict.items() if k.startswith('projection.') and v.is_floating_point()]
    if projection:
        print(f'  projection norm: {torch.cat([p.flatten() for p in projection]).norm():.4f} '
              '(random init, so it cannot be compared with a reference)')

    layers = lora_report(state_dict, conf)
    if layers:
        zero = sum(entry['zero'] for entry in layers)
        verdict = 'NOT TRAINED (every lora_B is zero)' if zero == len(layers) else \
            'trained' if zero == 0 else f'partly trained ({zero} adapters still zero)'
        print(f'  LoRA: {len(layers)} adapted layers, {zero} with lora_B all zero -> {verdict}')
        trained = [entry for entry in layers if not entry['zero']]
        if trained:
            norms = [entry['update_norm'] for entry in trained if 'update_norm' in entry]
            relative = [entry['relative_update'] for entry in trained if 'relative_update' in entry]
            if norms:
                print(f'        update |scale*B@A|: mean {sum(norms) / len(norms):.4g}, max {max(norms):.4g}')
            if relative:
                print(f'        relative to the frozen weight: mean {sum(relative) / len(relative):.4g}, '
                      f'max {max(relative):.4g}')
    elif conf.decoder.get('apply_lora'):
        print('  LoRA: apply_lora is set in config.yaml but the checkpoint has no adapters')

    if base is not None:
        for part, stats in base_report(state_dict, base).items():
            verdict = 'unchanged (frozen)' if stats['changed'] == 0 else 'trained'
            print(f"  {part:15s} vs pretrained: {stats['changed']} of {stats['compared']} tensors differ, "
                  f"max relative change {stats['max_relative_change']:.4g} -> {verdict}")
        if kind == 'small':
            print('  (small checkpoint: weights left out are the pretrained ones by construction)')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('models', nargs='+', help='output folders with config.yaml and <ckpt>_checkpoint.pt')
    parser.add_argument('--ckpt', nargs='+', default=['best', 'last'])
    parser.add_argument('--compare_base', action='store_true',
                        help='also compare encoder/decoder weights with the pretrained ones (builds the model)')
    args = parser.parse_args()

    for folder in args.models:
        conf_path = os.path.join(folder, 'config.yaml')
        if not os.path.exists(conf_path):
            print(f'\n{folder}: no config.yaml, skipped')
            continue
        conf = OmegaConf.load(conf_path)
        stage2 = conf.get('train', {}).get('stage2', {})
        print(f"\n##### {folder}\n  config: apply_lora={conf.decoder.get('apply_lora')}, "
              f"lora_rank={conf.decoder.get('lora_rank')}, lora_alpha={conf.decoder.get('lora_alpha')}, "
              f"stage2.frozen_encoder={stage2.get('frozen_encoder')}, stage2.epochs={stage2.get('epochs')}")
        paths = [os.path.join(folder, f'{name}_checkpoint.pt') for name in args.ckpt]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            print('  no checkpoints found')
            continue
        base = build_base(conf) if args.compare_base else None
        for path in paths:
            check(path, conf, base)
        del base
