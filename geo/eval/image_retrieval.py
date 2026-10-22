"""
Image-to-image retrieval with vision encoders, with the metrics of text_retrieval.py, to compare the encoders on the
thin-section images.

Every distinct combination of labels of a description type (campo) is a class. Each image is a query against every
other image of the same campo:
    exact     the positives are the images with exactly the same labels; R@K is the share of queries with a positive
              in the top K, mAP the mean average precision. Only queries with a positive count.
    graded    nDCG@10 with relevance = Jaccard overlap of the two label sets (as (category, value) pairs), so images
              sharing most labels count too; every image is a query.
Each (image, labels) pair is used once per campo; the same image under other labels (another annotation) is left out
of its rankings, since its embedding is identical. --exclude_same_slide also leaves out the images of the query's slide
(neighbouring crops of a slide look alike and mostly share the labels).

Encoders (--configs): geo config files (only their encoder section is read, see model/encoder.encoder_factory), so any
encoder FossilVL can use: DINOv3 (HuggingFace), Fossil (DINOv3 SSL checkpoint of this project), DINOtxt (off the shelf,
or a vlm-tuning run with encoder.config_path), DualEncoder (vlm-tuning run) and CLIP. Embeddings:
    --pooling global       the encoder's own image vector (return_grid=False): CLS for DINOv3/Fossil, CLS or CLS + mean
                           patch for DINOtxt (encoder.average_patch), the aligned head output for DualEncoder
    --pooling mean         mean of the grid tokens (return_grid=True: patch tokens, or the Q-Former query outputs)
    --pooling global+mean  both, concatenated (each normalised first)

Images are center crops of --size pixels at their native resolution (the standardized px per mm is kept; as
Fossil.preprocess, the order used in training), the same crop for every encoder; images whose shorter side is smaller
are skipped. CLIP uses its own preprocessing (224 px).

Usage:
    python geo/eval/image_retrieval.py --input geo/eval/conv_test.json --root <images> \
        --configs geo/eval/encoders/dinov3.yaml geo/eval/encoders/dinotxt.yaml geo/eval/encoders/fossil.yaml
    python geo/eval/image_retrieval.py --configs geo/config/dual.yaml --set encoder.weight_path=<run>/checkpoint-03.ckpt
"""
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageFile

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.dirname(HERE)
ROOT = os.path.dirname(GEO)
sys.path[:0] = [HERE, os.path.join(GEO, 'model'), GEO, ROOT]
from judge_reliability import normalize_labels  # noqa: E402
from text_retrieval import jaccard_matrix, retrieval_metrics  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
METRICS = ('R@1', 'R@5', 'R@10', 'mAP', 'nDCG@10')


# ------------------------------------------------------------------------------------------------------------ data

def image_path(root, sample):
    # as dataset.ConversationDataset
    if 'cd_guid' in sample:
        return os.path.join(root, '{}.png'.format(sample['cd_guid']))
    if 'image_name' in sample:
        return os.path.join(root, sample['image_name'].replace('\\', '/'))
    if 'image_id' in sample:
        return os.path.join(root, '{}.png'.format(sample['image_id']))
    raise ValueError('sample without image: {}'.format(sorted(sample)))


def load_items(path, root, campos=None, groups=None, min_size=None):
    """Distinct (image, labels) pairs per campo: {campo: [dict(image, slide, key, labels)]}, and the number of images
    skipped for being smaller than min_size."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seen, items, small = set(), defaultdict(list), set()
    for sample in data:
        campo = sample.get('campo')
        if campos and campo not in campos:
            continue
        if groups and sample.get('user_group') not in groups:
            continue
        if not sample.get('labels'):
            continue
        labels = normalize_labels(sample['labels'])
        if not labels:
            continue
        image = image_path(root, sample)
        if min_size and sample.get('size') is not None and sample['size'] < min_size:
            small.add(image)
            continue
        key = json.dumps(labels, ensure_ascii=False, sort_keys=True)
        if (campo, image, key) in seen:
            continue
        seen.add((campo, image, key))
        items[campo].append({'image': image, 'slide': sample.get('slide_id'), 'key': key,
                             'labels': {(c, v) for c, vs in labels.items() for v in vs}})
    return dict(items), len(small)


def exclusion_mask(items, same_slide):
    """[n, n] pairs left out of the rankings: the same image (other labels), and with same_slide the same slide."""
    images = np.array([it['image'] for it in items])
    mask = images[:, None] == images[None, :]
    if same_slide:
        slides = np.array([str(it['slide']) for it in items])
        known = np.array([it['slide'] is not None for it in items])
        mask |= (slides[:, None] == slides[None, :]) & known[:, None] & known[None, :]
    return mask


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert('RGB'))


class Crop:
    """Center crop of size pixels at native resolution, normalised (Fossil.preprocess, crop_center=True)."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        from encoder import Fossil
        return Fossil.preprocess(None, image, self.size)


# -------------------------------------------------------------------------------------------------------- encoders

def build_encoder(path, overrides, device):
    from encoder import encoder_factory
    conf = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides))
    encoder = encoder_factory(conf).to(device).eval()
    return encoder, conf


def model_dtype(encoder):
    return next(p for p in encoder.parameters() if p.is_floating_point()).dtype


@torch.no_grad()
def encode(encoder, paths, size, pooling, device, batch_size, workers, bf16):
    is_clip = type(encoder).__name__ == 'CLIP'
    if is_clip and pooling != 'global':
        raise ValueError('CLIP has no grid output, use --pooling global')
    transform = encoder._preprocess if is_clip else Crop(size)
    loader = torch.utils.data.DataLoader(Images(paths, transform), batch_size=batch_size, num_workers=workers)
    dtype = model_dtype(encoder)
    out = []
    for i, x in enumerate(loader):
        x = x.to(device, dtype=dtype)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16, enabled=bf16):
            parts = []
            if pooling in ('global', 'global+mean'):
                parts.append(encoder(x, return_grid=False))
            if pooling in ('mean', 'global+mean'):
                parts.append(encoder(x, return_grid=True).mean(1))
        parts = [torch.nn.functional.normalize(p.float(), dim=-1) for p in parts]
        out.append(torch.cat(parts, -1).cpu())
        if i % 20 == 0:
            print(f'  {min((i + 1) * batch_size, len(paths))}/{len(paths)} images')
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------------------------------------- report

def report(results, campos):
    lines = ['| encoder | campo | R@1 | R@5 | R@10 | mAP | nDCG@10 (Jaccard) | queries / images |', '|---' * 8 + '|']
    for name, per_campo in results.items():
        for campo in campos + ['mean']:
            m = per_campo.get(campo)
            if m:
                count = f"{m['queries']} / {m['items']}" if 'queries' in m else ''
                lines.append(f"| {name} | {campo} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['mAP']:.3f} "
                             f"| {m['nDCG@10']:.3f} | {count} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--input', default='geo/eval/conv_test.json', help='split json with the samples and labels')
    parser.add_argument('--root', default=None, help='image folder (default: data.root of the first config)')
    parser.add_argument('--configs', nargs='+', required=True, help='geo config files, one per encoder')
    parser.add_argument('--names', nargs='+', default=None, help='encoder names in the results (default: config names)')
    parser.add_argument('--set', nargs='*', default=[], help='config overrides applied to every encoder, KEY=VALUE')
    parser.add_argument('--campos', nargs='+', default=['composition', 'texture', 'porosity', 'diagenesis'])
    parser.add_argument('--groups', nargs='+', type=int, default=None, help='user groups kept (default: all)')
    parser.add_argument('--size', type=int, default=256, help='center crop side in pixels (smaller images skipped)')
    parser.add_argument('--pooling', choices=['global', 'mean', 'global+mean'], default='global')
    parser.add_argument('--exclude_same_slide', action='store_true', help='leave out images of the query slide')
    parser.add_argument('--random', action='store_true', help='add a random-embedding baseline')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--bf16', action='store_true', help='autocast the encoders to bfloat16')
    parser.add_argument('--out', default=None, help='json with the results (default: <input>_image_retrieval.json)')
    args = parser.parse_args()

    names = args.names or [os.path.splitext(os.path.basename(c))[0] for c in args.configs]
    if len(names) != len(args.configs) or len(set(names)) != len(names):
        raise ValueError('--names: one distinct name per config')
    broken = [s for s in args.set if '=' not in s]
    if broken:
        raise ValueError(f'--set entries must be KEY=VALUE, got {broken}')
    root = args.root or OmegaConf.load(args.configs[0]).get('data', {}).get('root')
    if not root:
        raise ValueError('--root: image folder (the first config has no data.root)')

    items, small = load_items(args.input, root, args.campos, args.groups, args.size)
    campos = [c for c in args.campos if c in items]
    print(f'images from {root}; {small} images smaller than {args.size} px skipped')
    jaccard, exclude = {}, {}
    for campo in campos:
        keys = [it['key'] for it in items[campo]]
        jaccard[campo] = jaccard_matrix([it['labels'] for it in items[campo]])
        exclude[campo] = exclusion_mask(items[campo], args.exclude_same_slide)
        print(f'{campo}: {len(keys)} images, {len(set(keys))} label combinations')
    paths = sorted({it['image'] for campo in campos for it in items[campo]})
    index = {p: i for i, p in enumerate(paths)}
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images not found, e.g. {missing[:3]}')

    def evaluate(embeddings):
        out = {}
        for campo in campos:
            rows = [index[it['image']] for it in items[campo]]
            m = retrieval_metrics(embeddings[rows], [it['key'] for it in items[campo]], jaccard[campo], exclude[campo])
            out[campo] = m
            print(f"  {campo:12s} R@1 {m['R@1']:.3f} R@5 {m['R@5']:.3f} R@10 {m['R@10']:.3f} mAP {m['mAP']:.3f} "
                  f"nDCG@10 {m['nDCG@10']:.3f} ({m['queries']} queries / {m['items']} images)")
        out['mean'] = {k: float(np.mean([out[c][k] for c in campos])) for k in METRICS}
        return out

    results = {}
    if args.random:
        print('\n=== random')
        results['random'] = evaluate(np.random.default_rng(0).standard_normal((len(paths), 256)).astype(np.float32))

    for name, config in zip(names, args.configs):
        print(f'\n=== {name} ({config})')
        encoder, _ = build_encoder(config, args.set, args.device)
        embeddings = encode(encoder, paths, args.size, args.pooling, args.device, args.batch_size, args.workers,
                            args.bf16)
        print(f'  {embeddings.shape[1]}-d embeddings')
        results[name] = evaluate(embeddings)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\n' + report(results, campos))
    out = args.out or os.path.splitext(args.input)[0] + '_image_retrieval.json'
    settings = {k: getattr(args, k) for k in ('input', 'root', 'configs', 'set', 'groups', 'size', 'pooling',
                                              'exclude_same_slide')}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'settings': settings, 'results': results}, f, indent=2)
    print(f'\nsaved to {out}')
