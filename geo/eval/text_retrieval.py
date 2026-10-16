"""
Text-to-text retrieval with pretrained text encoders, to compare them on the geology descriptions.

Every distinct combination of labels of a description type (campo) is a class. Each description is a query against
every other description of the same campo (descriptions of different campos are trivially separable):
    exact     the positives are the descriptions with exactly the same labels; R@K is the share of queries with a
              positive in the top K, mAP the mean average precision. Only queries with a positive count.
    graded    nDCG@10 with relevance = Jaccard overlap of the two label sets (as (category, value) pairs), so
              descriptions sharing most labels count too; every description is a query.
Each (description, labels) pair is used once: the same description repeated for many images would otherwise find
itself. The share of descriptions longer than the encoder's context (truncated) is reported too.

Encoders (--models), as <kind>:<name>:
    hf:<model or preset>   HuggingFace encoder; presets set the pooling and prefix each model expects (see PRESETS),
                           any other model id uses mean pooling
    clip:<name>            OpenAI CLIP text tower (clip package), 77 tokens
    dinotxt:<weights.pth>  DINO-TXT text tower (torch hub of this repository; --bpe for the tokenizer vocabulary)
    tfidf, random          lexical and chance baselines

Usage:
    python geo/eval/text_retrieval.py --input geo/eval/conv_test.json --models tfidf random hf:bertimbau-base hf:e5-base
    python geo/eval/text_retrieval.py --input geo/eval/conv_val.json --campos composition texture --models hf:bge-m3
"""
import json
import math
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_reliability import normalize_labels  # noqa: E402

# name -> (model id, pooling, prefix); pooling: mean, cls or last (last non-padding token, decoder embedders)
PRESETS = {
    'bertimbau-base': ('neuralmind/bert-base-portuguese-cased', 'mean', ''),
    'bertimbau-large': ('neuralmind/bert-large-portuguese-cased', 'mean', ''),
    'mpnet-multilingual': ('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'mean', ''),
    'minilm-multilingual': ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'mean', ''),
    'labse': ('sentence-transformers/LaBSE', 'cls', ''),
    'e5-base': ('intfloat/multilingual-e5-base', 'mean', 'query: '),
    'e5-large': ('intfloat/multilingual-e5-large', 'mean', 'query: '),
    'bge-m3': ('BAAI/bge-m3', 'cls', ''),
    'gte-multilingual': ('Alibaba-NLP/gte-multilingual-base', 'cls', ''),
    'qwen3-embedding-0.6b': ('Qwen/Qwen3-Embedding-0.6B', 'last', ''),
    'xlm-roberta-base': ('FacebookAI/xlm-roberta-base', 'mean', ''),
}
KS = (1, 5, 10)


# ------------------------------------------------------------------------------------------------------------ data

def load_items(path, campos=None):
    """Distinct (description, labels) pairs per campo: {campo: [(text, label key, label set)]}."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seen, items = set(), defaultdict(list)
    for sample in data:
        campo = sample.get('campo')
        if campos and campo not in campos:
            continue
        text = sample['conversation'][1]['content']
        if not isinstance(text, str) or not text.strip() or not sample.get('labels'):
            continue
        labels = normalize_labels(sample['labels'])
        if not labels:
            continue
        key = json.dumps(labels, ensure_ascii=False, sort_keys=True)
        if (campo, text.strip(), key) in seen:
            continue
        seen.add((campo, text.strip(), key))
        items[campo].append((text.strip(), key, {(c, v) for c, vs in labels.items() for v in vs}))
    return dict(items)


# -------------------------------------------------------------------------------------------------------- encoders

class HFEncoder:
    def __init__(self, name, device, batch_size, max_length=None):
        from transformers import AutoModel, AutoTokenizer
        model_id, self.pooling, self.prefix = PRESETS.get(name, (name, 'mean', ''))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.pooling == 'last':
            self.tokenizer.padding_side = 'left'  # the last position is then the last token of every text
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        self.device, self.batch_size = device, batch_size
        limit = self.tokenizer.model_max_length
        self.max_length = max_length or (limit if limit and limit < 100_000 else 512)

    def truncated(self, texts):
        lengths = [len(ids) for ids in self.tokenizer([self.prefix + t for t in texts])['input_ids']]
        return float(np.mean([n > self.max_length for n in lengths]))

    @torch.no_grad()
    def encode(self, texts):
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer([self.prefix + t for t in texts[i:i + self.batch_size]], padding=True,
                                   truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            if self.pooling == 'cls':
                pooled = hidden[:, 0]
            elif self.pooling == 'last':
                pooled = hidden[:, -1]
            else:
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu())
        return torch.cat(out).numpy()


class CLIPEncoder:
    def __init__(self, name, device, batch_size):
        import clip
        self.clip = clip
        self.model, _ = clip.load(name, device=device)
        self.device, self.batch_size, self.max_length = device, batch_size, 77

    def truncated(self, texts):
        tokenizer = self.clip.simple_tokenizer.SimpleTokenizer()
        return float(np.mean([len(tokenizer.encode(t)) + 2 > self.max_length for t in texts]))

    @torch.no_grad()
    def encode(self, texts):
        return torch.cat([self.model.encode_text(self.clip.tokenize(texts[i:i + self.batch_size], truncate=True)
                                                 .to(self.device)).float().cpu()
                          for i in range(0, len(texts), self.batch_size)]).numpy()


class DINOTxtEncoder:
    def __init__(self, weights, device, batch_size, bpe=None):
        repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        # pretrained=False: the vision backbone is not needed here, only the text tower is loaded from `weights`
        kwargs = {'pretrained': False}
        if bpe:
            kwargs['bpe_path_or_url'] = bpe
        model, self.tokenizer = torch.hub.load(repo, 'dinov3_vitl16_dinotxt_tet1280d20h24l', source='local', **kwargs)
        state = torch.load(weights, map_location='cpu')
        state = state.get('model', state.get('state_dict', state))
        text_keys = [k for k in model.state_dict() if k.startswith('text_model.')]
        loaded = set(model.load_state_dict(state, strict=False).missing_keys)
        missing = [k for k in text_keys if k in loaded]
        if missing:
            raise ValueError(f'{weights}: {len(missing)}/{len(text_keys)} text tower weights missing, e.g. {missing[:3]}')
        self.model = model.to(device).eval()
        self.device, self.batch_size, self.max_length = device, batch_size, 77

    def truncated(self, texts):
        return float(np.mean([len(self.tokenizer.encode(t)) + 2 > self.max_length for t in texts]))

    @torch.no_grad()
    def encode(self, texts):
        return torch.cat([self.model.encode_text(self.tokenizer.tokenize(texts[i:i + self.batch_size]).to(self.device))
                          .float().cpu() for i in range(0, len(texts), self.batch_size)]).numpy()


class TfidfEncoder:
    """Lexical baseline: word 1-2 grams, fitted on the descriptions being ranked."""
    max_length = None

    def truncated(self, texts):
        return 0.0

    def encode(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts).toarray()


class RandomEncoder:
    max_length = None

    def truncated(self, texts):
        return 0.0

    def encode(self, texts):
        return np.random.default_rng(0).standard_normal((len(texts), 256))


def build_encoder(spec, args):
    kind, _, name = spec.partition(':')
    if kind == 'hf':
        return HFEncoder(name, args.device, args.batch_size, args.max_length)
    if kind == 'clip':
        return CLIPEncoder(name, args.device, args.batch_size)
    if kind == 'dinotxt':
        return DINOTxtEncoder(name, args.device, args.batch_size, args.bpe)
    if kind == 'tfidf':
        return TfidfEncoder()
    if kind == 'random':
        return RandomEncoder()
    raise ValueError(f'unknown encoder {spec} (hf:, clip:, dinotxt:, tfidf, random)')


# --------------------------------------------------------------------------------------------------------- metrics

def jaccard_matrix(label_sets):
    """Pairwise Jaccard overlap of the label sets, [n, n]."""
    vocab = {pair: i for i, pair in enumerate(sorted(set().union(*label_sets)))}
    onehot = np.zeros((len(label_sets), len(vocab)), dtype=np.float32)
    for row, labels in enumerate(label_sets):
        onehot[row, [vocab[p] for p in labels]] = 1
    inter = onehot @ onehot.T
    sizes = onehot.sum(1)
    return inter / (sizes[:, None] + sizes[None, :] - inter)


def retrieval_metrics(embeddings, keys, jaccard):
    """Exact-class R@K and mAP, and Jaccard-graded nDCG@10, every item against all the others."""
    x = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)  # a query never retrieves itself
    order = np.argsort(-sim, axis=1)[:, :-1]  # self (-inf) is last
    keys = np.array(keys)

    hits = {k: [] for k in KS}
    aps, ndcgs = [], []
    for q in range(len(keys)):
        ranked = order[q]
        relevant = keys[ranked] == keys[q]
        if relevant.any():
            for k in KS:
                hits[k].append(relevant[:k].any())
            precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
            aps.append(float((precision * relevant).sum() / relevant.sum()))
        gains = jaccard[q, ranked]
        discounts = 1 / np.log2(np.arange(2, 12))
        ideal = (np.sort(gains)[::-1][:10] * discounts).sum()
        ndcgs.append(float((gains[:10] * discounts).sum() / ideal) if ideal > 0 else 0.0)
    out = {f'R@{k}': float(np.mean(v)) if v else float('nan') for k, v in hits.items()}
    out.update({'mAP': float(np.mean(aps)) if aps else float('nan'), 'nDCG@10': float(np.mean(ndcgs)),
                'queries': len(aps), 'items': len(keys)})
    return out


def report(results, campos):
    lines = ['| encoder | campo | R@1 | R@5 | R@10 | mAP | nDCG@10 (Jaccard) | truncated |', '|---' * 8 + '|']
    for name, per_campo in results.items():
        for campo in campos + ['mean']:
            m = per_campo.get(campo)
            if m:
                lines.append(f"| {name} | {campo} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['mAP']:.3f} "
                             f"| {m['nDCG@10']:.3f} | {m['truncated']:.0%} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', default='geo/eval/conv_test.json', help='split json with conversations and labels')
    parser.add_argument('--campos', nargs='+', default=['composition', 'texture', 'porosity', 'diagenesis'],
                        help='description types (classification has only 8 distinct descriptions)')
    parser.add_argument('--models', nargs='+', default=['random', 'tfidf', 'hf:bertimbau-base', 'hf:mpnet-multilingual',
                                                        'hf:e5-base', 'hf:bge-m3'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=None, type=int, help='token limit of hf encoders (default: the model\'s)')
    parser.add_argument('--bpe', default=None, help='BPE vocabulary for dinotxt (default: downloaded)')
    parser.add_argument('--out', default=None, help='json with the results (default: <input>_text_retrieval.json)')
    args = parser.parse_args()

    items = load_items(args.input, args.campos)
    campos = [c for c in args.campos if c in items]
    jaccard = {}
    for campo in campos:
        keys = [k for _, k, _ in items[campo]]
        jaccard[campo] = jaccard_matrix([s for _, _, s in items[campo]])
        print(f'{campo}: {len(keys)} descriptions, {len(set(keys))} label combinations')

    results = {}
    for spec in args.models:
        print(f'\n=== {spec}')
        encoder = build_encoder(spec, args)
        results[spec] = {}
        for campo in campos:
            texts = [t for t, _, _ in items[campo]]
            m = retrieval_metrics(encoder.encode(texts), [k for _, k, _ in items[campo]], jaccard[campo])
            m['truncated'] = encoder.truncated(texts)
            results[spec][campo] = m
            print(f"  {campo:12s} R@1 {m['R@1']:.3f} R@5 {m['R@5']:.3f} R@10 {m['R@10']:.3f} mAP {m['mAP']:.3f} "
                  f"nDCG@10 {m['nDCG@10']:.3f} ({m['queries']} queries) truncated {m['truncated']:.0%}")
        results[spec]['mean'] = {k: float(np.mean([results[spec][c][k] for c in campos]))
                                 for k in ('R@1', 'R@5', 'R@10', 'mAP', 'nDCG@10', 'truncated')}
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\n' + report(results, campos))
    out = args.out or os.path.splitext(args.input)[0] + '_text_retrieval.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'input': args.input, 'results': results}, f, indent=2)
    print(f'\nsaved to {out}')
