"""
Runs eval_labels_llm.py with several judge models and stores the results for comparison.

Besides judging preds and refs (positives), each judge also scores a negative control: every reference
paired with labels taken from other samples that are not annotated for it. Recall on those is the false
positive rate, which tells apart a judge that reads the text from one that always answers Yes.

Usage:
    python geo/eval/compare_judges.py --out_dir geo/eval/judge_comparison
"""
import json
import os
import random
import re
import time
from argparse import ArgumentParser

from eval_labels_llm import HFJudge, run

# (model, load_in_4bit, batch_size); all fit an 8 GB GPU
MODELS = [
    ('Qwen/Qwen2.5-0.5B-Instruct', False, 16),
    ('Qwen/Qwen2.5-1.5B-Instruct', False, 16),
    ('Qwen/Qwen2.5-3B-Instruct', False, 8),
    ('microsoft/Phi-3.5-mini-instruct', True, 8),
    ('Qwen/Qwen2.5-7B-Instruct', True, 4),
    ('Qwen/Qwen3-8B', True, 4),
]


def build_negatives(data, seed=0):
    """For each sample, labels of the same category borrowed from other samples and absent from its own."""
    rng = random.Random(seed)
    labels = data['labels']
    pools = {}
    for sample_labels in labels:
        for category, values in sample_labels.items():
            pools.setdefault(category, set()).update(v.strip() for v in values)
    negatives = []
    for sample_labels in labels:
        neg = {}
        for category, values in sample_labels.items():
            own = {v.strip().lower() for v in values}
            candidates = sorted(v for v in pools[category] if v.lower() not in own)
            k = min(len(values), len(candidates))
            if k:
                neg[category] = rng.sample(candidates, k)
        negatives.append(neg)
    return {'refs': data['refs'], 'labels': negatives}


class LexicalJudge:
    """Baseline: yes when the label (text before ':') appears verbatim in the description."""

    def score(self, batch_messages):
        probs = []
        for messages in batch_messages:
            user = messages[-1]['content']
            text = user.split('Description:\n')[1].split('\n\nClaim:')[0].lower()
            value = re.search(r'"(.+?)"', user.split('Claim:')[1]).group(1).lower().split(':')[0]
            probs.append(1.0 if value in text else 0.0)
        return probs


def evaluate(judge, data, negatives, batch_size):
    results = run(data, judge, texts=('preds', 'refs'), batch_size=batch_size)
    results['refs_negative'] = run(negatives, judge, texts=('refs',), batch_size=batch_size)['refs']
    return results


def slim(results):
    """Keeps summaries and per-question probabilities, drops the repeated description text."""
    out = {}
    for key, r in results.items():
        out[key] = {'summary': r['summary'],
                    'questions': [{k: v for k, v in q.items() if k != 'text'} for q in r['questions']]}
    return out


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', default='geo/eval/selected.json')
    parser.add_argument('--out_dir', default='geo/eval/judge_comparison')
    parser.add_argument('--models', nargs='*', default=None, help='subset of MODELS names to run')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    negatives = build_negatives(data)
    os.makedirs(args.out_dir, exist_ok=True)

    path = os.path.join(args.out_dir, 'lexical.json')
    if not os.path.exists(path):
        start = time.time()
        results = evaluate(LexicalJudge(), data, negatives, 256)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'model': 'lexical', 'seconds': time.time() - start, **slim(results)}, f, ensure_ascii=False)

    import torch

    for name, four_bit, batch_size in MODELS:
        if args.models and name not in args.models:
            continue
        path = os.path.join(args.out_dir, name.replace('/', '__') + '.json')
        if os.path.exists(path):
            print(f'skipping {name}, {path} exists')
            continue
        print(f'\n##### {name} (4bit={four_bit}, batch={batch_size})')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        judge = HFJudge(name, load_in_4bit=four_bit)
        tok = judge.tokenizer
        yes = [tok.decode([i]) for i in judge.yes_ids]
        no = [tok.decode([i]) for i in judge.no_ids]
        print('yes tokens', yes, 'no tokens', no)
        assert not set(judge.yes_ids) & set(judge.no_ids), 'yes/no token sets overlap'

        start = time.time()
        results = evaluate(judge, data, negatives, batch_size)
        record = {'model': name, 'load_in_4bit': four_bit, 'batch_size': batch_size,
                  'seconds': time.time() - start, 'peak_vram_gb': torch.cuda.max_memory_allocated() / 2**30,
                  'yes_tokens': yes, 'no_tokens': no, **slim(results)}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
        print(f"done in {record['seconds']:.0f}s, peak {record['peak_vram_gb']:.2f} GB, "
              f"preds recall {results['preds']['summary']['label_recall']:.4f}, "
              f"refs recall {results['refs']['summary']['label_recall']:.4f}, "
              f"refs FPR {results['refs_negative']['summary']['label_recall']:.4f}")
        del judge
