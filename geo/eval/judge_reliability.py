"""
Reliability of the LLM judges of eval_labels_llm.py, measured on ground-truth texts only.

The references were written from the labels, so a reliable judge must say Yes when a reference is paired
with its own labels (positives) and No when it is paired with labels it does not have (negatives).
Negatives are of two kinds:
    absent     a label of the same category borrowed from another sample (--neg_ratio per positive), sharing no
               '/'-separated part with any of the sample's labels (so Arbustos/Dígitos is never a negative
               for a sample annotated Arbustos)
    role_swap  every main constituent of the sample claimed as secondary, and vice versa (not with --ignore_role)

Identical (reference, labels) pairs are judged once and weighted by how often they occur.

Subcommands:
    download  pre-fetches models into the HF cache (run on a node with internet)
    run       judges every question with one model; shards over GPUs/processes and resumes after restarts
    merge     joins the shards of every model in --out_dir and writes report.md, summary.json, hard_cases.jsonl

Sharding: every process judges the questions with uid % world_size == rank. rank/world_size come from
--shard/--num_shards, else SLURM_PROCID/SLURM_NTASKS (srun), else RANK/WORLD_SIZE (torchrun), else 0/1.
See geo/eval/slurm/README.md.

    python geo/eval/judge_reliability.py run --model Qwen/Qwen2.5-7B-Instruct --out_dir geo/eval/reliability
    python geo/eval/judge_reliability.py merge --out_dir geo/eval/reliability
"""
import glob
import hashlib
import json
import os
import random
import sys
import time
from argparse import ArgumentParser
from collections import Counter, defaultdict

from eval_labels_llm import build_claim, build_messages

QUESTIONS_FILE = 'questions.json'
ROLE_SWAP = {'Constituintes Principais': 'Constituintes Secundários',
             'Constituintes Secundários': 'Constituintes Principais'}


# ----------------------------------------------------------------------------------------------- questions

def _parts(value):
    return {p.strip() for p in value.strip().lower().split('/') if p.strip()}


def build_question_set(data, seed=0, neg_ratio=1.0, ignore_role=False):
    """Unique (ref, labels) samples with their occurrence counts, and one question per positive/negative claim."""
    counts = Counter()
    first = {}
    for ref, labels in zip(data['refs'], data['labels']):
        labels = {c: sorted({v.strip() for v in vs if v.strip()}) for c, vs in labels.items()}
        labels = {c: vs for c, vs in labels.items() if vs}
        key = json.dumps([ref, labels], ensure_ascii=False, sort_keys=True)
        counts[key] += 1
        first.setdefault(key, (ref, labels))
    samples = [{'text': first[k][0], 'labels': first[k][1], 'count': counts[k]} for k in first]

    pools = defaultdict(set)
    for s in samples:
        for c, vs in s['labels'].items():
            pools[c].update(vs)
    pools = {c: sorted(vs) for c, vs in pools.items()}

    rng = random.Random(seed)
    questions = []
    for idx, s in enumerate(samples):
        sample_parts = set().union(*[_parts(v) for vs in s['labels'].values() for v in vs])
        for category, values in s['labels'].items():
            for value in values:
                questions.append({'sample': idx, 'category': category, 'label': value, 'gold': 1, 'kind': 'positive'})
            candidates = [v for v in pools[category] if not _parts(v) & sample_parts]
            k = min(round(len(values) * neg_ratio), len(candidates))
            for value in rng.sample(candidates, k):
                questions.append({'sample': idx, 'category': category, 'label': value, 'gold': 0, 'kind': 'absent'})
            # every constituent of the other role, claimed in this role (with loose claims it would be true)
            other = ROLE_SWAP.get(category)
            if other and not ignore_role:
                own = set().union(*[_parts(v) for v in values])
                for value in s['labels'].get(other, []):
                    if not _parts(value) & own:
                        questions.append({'sample': idx, 'category': category, 'label': value, 'gold': 0,
                                          'kind': 'role_swap'})
    for uid, q in enumerate(questions):
        q['uid'] = uid
        q['claim'] = build_claim(q['category'], q['label'], ignore_role)
    return samples, questions


def load_or_create_questions(args):
    """Every process builds the same set; the first one to finish saves it, the others check it matches."""
    with open(args.input, 'rb') as f:
        input_sha1 = hashlib.sha1(f.read()).hexdigest()
    params = {'input': os.path.abspath(args.input), 'input_sha1': input_sha1, 'seed': args.seed,
              'neg_ratio': args.neg_ratio, 'ignore_role': args.ignore_role}
    path = os.path.join(args.out_dir, QUESTIONS_FILE)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        mismatch = {k: (saved['params'].get(k), v) for k, v in params.items()
                    if k != 'input' and saved['params'].get(k) != v}
        if mismatch:
            sys.exit(f'{path} was built with different settings {mismatch}; use another --out_dir')
        return saved

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples, questions = build_question_set(data, args.seed, args.neg_ratio, args.ignore_role)
    saved = {'params': params, 'n_input_samples': len(data['refs']), 'samples': samples, 'questions': questions}
    os.makedirs(args.out_dir, exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic: concurrent writers produce identical content
    return saved


# --------------------------------------------------------------------------------------------------- run

def dist_env(args):
    if args.num_shards is not None:
        return args.shard, args.num_shards, args.shard
    for rank, world, local in (('SLURM_PROCID', 'SLURM_NTASKS', 'SLURM_LOCALID'), ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')):
        if rank in os.environ and world in os.environ:
            return int(os.environ[rank]), int(os.environ[world]), int(os.environ.get(local, 0))
    return 0, 1, 0


def run_tag(args):
    tag = args.model.replace('/', '__')
    return tag + ('-4bit' if args.load_in_4bit else '')


def read_done(model_dir):
    done = {}
    for path in glob.glob(os.path.join(model_dir, 'part-*.jsonl')):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # line cut by a killed job
                    continue
                done[r['uid']] = r['p_yes']
    return done


def cmd_run(args):
    rank, world, local_rank = dist_env(args)
    qset = load_or_create_questions(args)
    samples, questions = qset['samples'], qset['questions']

    model_dir = os.path.join(args.out_dir, run_tag(args))
    os.makedirs(model_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(model_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump({'model': args.model, 'load_in_4bit': args.load_in_4bit, 'device_map': args.device_map,
                       'batch_size': args.batch_size, 'world_size': world}, f, indent=2)

    done = read_done(model_dir)
    todo = [q for q in questions if q['uid'] % world == rank and q['uid'] not in done]
    print(f'[rank {rank}/{world}] {args.model}: {len(todo)} questions to judge '
          f'({len(questions)} in total, {len(done)} already done)', flush=True)
    if not todo:
        return

    import torch
    from tqdm import tqdm
    from eval_labels_llm import HFJudge

    if args.device_map:
        device = None
    elif torch.cuda.is_available():
        device = f'cuda:{local_rank % torch.cuda.device_count()}'
    else:
        device = 'cpu'
    judge = HFJudge(args.model, load_in_4bit=args.load_in_4bit, device=device, device_map=args.device_map)
    if set(judge.yes_ids) & set(judge.no_ids):
        sys.exit(f'yes/no token ids overlap for {args.model}: {judge.yes_ids} {judge.no_ids}')

    # longest first: similar lengths share batches (less padding) and an OOM shows up at the start
    todo.sort(key=lambda q: len(samples[q['sample']]['text']) + len(q['claim']), reverse=True)
    batch_size, start, t0 = args.batch_size, 0, time.time()
    out_path = os.path.join(model_dir, f'part-{rank:03d}-of-{world:03d}.jsonl')
    with open(out_path, 'a', encoding='utf-8') as f, tqdm(total=len(todo), mininterval=30, file=sys.stdout) as bar:
        while start < len(todo):
            batch = todo[start:start + batch_size]
            messages = [build_messages({'text': samples[q['sample']]['text'], 'claim': q['claim']}) for q in batch]
            try:
                probs = judge.score(messages)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size //= 2
                torch.cuda.empty_cache()
                print(f'[rank {rank}] OOM, batch size reduced to {batch_size}', flush=True)
                continue
            for q, p in zip(batch, probs):
                f.write(json.dumps({'uid': q['uid'], 'p_yes': p}) + '\n')
            f.flush()
            start += len(batch)
            bar.update(len(batch))
    print(f'[rank {rank}] done in {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------- merge

def _rates(gold, pred, w):
    """Weighted TPR, FPR, precision and balanced accuracy of binary decisions."""
    pos, neg = gold == 1, gold == 0
    tpr = float((pred[pos] * w[pos]).sum() / w[pos].sum()) if pos.any() else float('nan')
    fpr = float((pred[neg] * w[neg]).sum() / w[neg].sum()) if neg.any() else float('nan')
    tp, fp = (pred * pos * w).sum(), (pred * neg * w).sum()
    precision = float(tp / (tp + fp)) if tp + fp else float('nan')
    return tpr, fpr, precision, (tpr + 1 - fpr) / 2 if pos.any() and neg.any() else float('nan')


def evaluate_model(qs, samples, p, threshold):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    gold = np.array([q['gold'] for q in qs])
    w = np.array([samples[q['sample']]['count'] for q in qs], dtype=float)
    pred = (p >= threshold).astype(float)
    tpr, fpr, precision, bacc = _rates(gold, pred, w)
    res = {'tpr': tpr, 'fpr': fpr, 'precision': precision, 'balanced_accuracy': bacc,
           'auc': float(roc_auc_score(gold, p, sample_weight=w))}

    u = np.ones_like(w)
    res['unique'] = dict(zip(('tpr', 'fpr', 'precision', 'balanced_accuracy'), _rates(gold, pred, u)))
    res['unique']['auc'] = float(roc_auc_score(gold, p))

    kinds = np.array([q['kind'] for q in qs])
    res['fpr_by_kind'] = {k: float((pred[kinds == k] * w[kinds == k]).sum() / w[kinds == k].sum())
                          for k in ('absent', 'role_swap') if (kinds == k).any()}
    cats = np.array([q['category'] for q in qs])
    res['per_category'] = {}
    for c in sorted(set(cats)):
        m = cats == c
        c_tpr, c_fpr, _, c_bacc = _rates(gold[m], pred[m], w[m])
        res['per_category'][c] = {'tpr': c_tpr, 'fpr': c_fpr, 'balanced_accuracy': c_bacc,
                                  'n_pos': int((gold[m] == 1).sum()), 'n_neg': int((gold[m] == 0).sum())}

    # best threshold for balanced accuracy (tuned on this same data, so optimistic)
    grid = np.linspace(0.01, 0.99, 99)
    baccs = [_rates(gold, (p >= t).astype(float), w)[3] for t in grid]
    res['best_threshold'] = float(grid[int(np.argmax(baccs))])
    res['best_balanced_accuracy'] = float(max(baccs))

    # per sample: all its positives accepted and none of its negatives
    ok = defaultdict(lambda: True)
    for q, d in zip(qs, pred):
        ok[q['sample']] &= bool(d == q['gold'])
    total = sum(samples[s]['count'] for s in ok)
    res['samples_fully_correct'] = sum(samples[s]['count'] for s, v in ok.items() if v) / total
    return res


def cmd_merge(args):
    import numpy as np
    from sklearn.metrics import cohen_kappa_score

    with open(os.path.join(args.out_dir, QUESTIONS_FILE), 'r', encoding='utf-8') as f:
        qset = json.load(f)
    samples, questions = qset['samples'], qset['questions']

    models, probs = {}, {}
    for meta_path in sorted(glob.glob(os.path.join(args.out_dir, '*', 'meta.json'))):
        model_dir = os.path.dirname(meta_path)
        tag = os.path.basename(model_dir)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        done = read_done(model_dir)
        missing = len(questions) - len(done)
        if missing:
            print(f'WARNING {tag}: {missing}/{len(questions)} questions not judged yet, metrics use the rest')
        if not done:
            continue
        qs = [q for q in questions if q['uid'] in done]
        p = np.array([done[q['uid']] for q in qs])
        res = evaluate_model(qs, samples, p, args.threshold)
        res.update(meta=meta, judged=len(done), missing=missing)
        models[tag] = res
        probs[tag] = done

    if not models:
        sys.exit(f'no model results found in {args.out_dir}')

    # agreement between judges on the questions judged by both
    tags = list(models)
    agreement = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            common = sorted(set(probs[a]) & set(probs[b]))
            da = np.array([probs[a][u] for u in common]) >= args.threshold
            db = np.array([probs[b][u] for u in common]) >= args.threshold
            agreement[f'{a} | {b}'] = {'agreement': float((da == db).mean()), 'kappa': float(cohen_kappa_score(da, db))}

    # questions most judges get wrong: label errors in the references, or claims that are hard for every judge
    hard = []
    for q in questions:
        votes = {t: probs[t][q['uid']] for t in tags if q['uid'] in probs[t]}
        wrong = sum((p >= args.threshold) != q['gold'] for p in votes.values())
        if votes and wrong / len(votes) > 0.5:
            s = samples[q['sample']]
            hard.append({'gold': q['gold'], 'kind': q['kind'], 'category': q['category'], 'label': q['label'],
                         'claim': q['claim'], 'text': s['text'], 'count': s['count'], 'wrong_judges': wrong,
                         'p_yes': votes})
    hard.sort(key=lambda h: (-h['wrong_judges'], -h['count']))

    n_q = Counter(q['kind'] for q in questions)
    summary = {'params': qset['params'], 'threshold': args.threshold, 'n_input_samples': qset['n_input_samples'],
               'n_unique_samples': len(samples), 'n_questions': dict(n_q), 'models': models, 'agreement': agreement}
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'hard_cases.jsonl'), 'w', encoding='utf-8') as f:
        for h in hard:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    report = render_report(summary, len(hard))
    with open(os.path.join(args.out_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)


def render_report(s, n_hard):
    order = sorted(s['models'], key=lambda t: -s['models'][t]['balanced_accuracy'])
    lines = [
        '# Judge reliability on ground-truth texts', '',
        f"Input: `{s['params']['input']}`: {s['n_input_samples']} samples, {s['n_unique_samples']} unique "
        f"(reference, labels) pairs. Questions: {s['n_questions'].get('positive', 0)} positive, "
        f"{s['n_questions'].get('absent', 0)} absent negatives, {s['n_questions'].get('role_swap', 0)} role-swap "
        f"negatives. Threshold {s['threshold']}. Rates are weighted by how often each pair occurs in the input.", '',
        '| Judge | TPR ↑ | FPR ↓ | FPR absent | FPR role swap | Precision | Bal. acc. ↑ | AUC ↑ | Best thr. (bal. acc.) '
        '| Samples fully correct | Missing |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for t in order:
        m = s['models'][t]
        k = m['fpr_by_kind']
        lines.append(f"| {t} | {m['tpr']:.3f} | {m['fpr']:.3f} | {k.get('absent', float('nan')):.3f} | "
                     f"{k.get('role_swap', float('nan')):.3f} | {m['precision']:.3f} | {m['balanced_accuracy']:.3f} | "
                     f"{m['auc']:.3f} | {m['best_threshold']:.2f} ({m['best_balanced_accuracy']:.3f}) | "
                     f"{m['samples_fully_correct']:.3f} | {m['missing']} |")
    cats = sorted({c for m in s['models'].values() for c in m['per_category']})
    lines += ['', '## Per category (TPR / FPR)', '', '| Judge | ' + ' | '.join(cats) + ' |',
              '|---|' + '---|' * len(cats)]
    for t in order:
        pc = s['models'][t]['per_category']
        cells = [f"{pc[c]['tpr']:.3f} / {pc[c]['fpr']:.3f}" if c in pc else '-' for c in cats]
        lines.append(f'| {t} | ' + ' | '.join(cells) + ' |')
    if s['agreement']:
        lines += ['', '## Agreement between judges', '', '| Pair | Agreement | Cohen κ |', '|---|---|---|']
        for pair, a in sorted(s['agreement'].items(), key=lambda x: -x[1]['kappa']):
            lines.append(f"| {pair} | {a['agreement']:.3f} | {a['kappa']:.3f} |")
    lines += ['', f'{n_hard} questions are judged wrongly by most judges; see `hard_cases.jsonl` '
                  '(possible errors in the references or labels).', '']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------- download

def cmd_download(args):
    from huggingface_hub import snapshot_download
    for model in args.models:
        path = snapshot_download(model, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.py',
                                                        '*.tiktoken', 'tokenizer*'])
        print(f'{model} -> {path}')


# -------------------------------------------------------------------------------------------------- main

def main():
    parser = ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('run', help='judge all questions with one model (sharded)')
    p.add_argument('--model', required=True)
    p.add_argument('--input', default='geo/eval/todos.json', help='json with refs and labels')
    p.add_argument('--out_dir', default='geo/eval/reliability')
    p.add_argument('--load_in_4bit', action='store_true')
    p.add_argument('--device_map', default=None,
                   help="'auto' spreads one model over all GPUs visible to the process (models too big for one GPU)")
    p.add_argument('--batch_size', default=32, type=int, help='starting batch size, halved on out-of-memory')
    p.add_argument('--seed', default=0, type=int, help='seed for sampling negatives')
    p.add_argument('--neg_ratio', default=1.0, type=float, help='negatives per positive, per sample and category')
    p.add_argument('--ignore_role', action='store_true', help='loose claims (label mentioned, role ignored)')
    p.add_argument('--shard', default=0, type=int)
    p.add_argument('--num_shards', default=None, type=int, help='overrides SLURM/torchrun rank detection')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('merge', help='compute metrics for every model in out_dir')
    p.add_argument('--out_dir', default='geo/eval/reliability')
    p.add_argument('--threshold', default=0.5, type=float)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser('download', help='pre-fetch models into the HF cache')
    p.add_argument('models', nargs='+')
    p.set_defaults(func=cmd_download)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
