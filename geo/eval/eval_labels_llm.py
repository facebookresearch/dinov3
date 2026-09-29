"""
LLM-as-judge evaluation: checks whether each human-annotated label is expressed in the generated text.

For every (sample, label) pair a yes/no question is asked to a local HuggingFace instruct model.
Instead of free generation, the answer is read from the next-token logits of "Yes" vs "No",
so a single forward pass per question is needed and the output can never be malformed.

The reference texts can also be judged (--texts preds refs): since the references were written from
the labels, recall on refs is a sanity check of how reliable the judge is.

Usage:
    python geo/eval/eval_labels_llm.py --input geo/eval/selected.json --texts preds refs
    python geo/eval/eval_labels_llm.py --input geo/eval/selected.json --model Qwen/Qwen2.5-7B-Instruct --load_in_4bit
"""
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are an expert petrographer evaluating automatically generated descriptions of thin sections "
    "of carbonate rocks, written in Portuguese. Given a description and a claim, decide whether the "
    "description states the claim. Synonyms, singular/plural and different wording count as a match, "
    "but the information must be explicitly present in the description. Answer only Yes or No."
)

# how each label category is phrased as a claim; {value} is the annotated label
CLAIMS = {
    'Constituintes Principais': '"{value}" is listed as a main constituent (constituinte principal) of the sample.',
    'Constituintes Secundários': '"{value}" is listed as a secondary constituent (constituinte secundário) of the sample.',
    'Tamanho do Elemento': 'The size of the elements is described as "{value}".',
    'Comp. Atual do Elemento': 'The elements are described as currently composed of "{value}".',
}
LOOSE_CLAIM = '"{value}" ({category}) is mentioned in the description.'


def build_claim(category, value, ignore_role=False):
    if ignore_role or category not in CLAIMS:
        return LOOSE_CLAIM.format(value=value, category=category)
    return CLAIMS[category].format(value=value)


def build_questions(texts, labels, ignore_role=False):
    """Flattens the dataset into one question per (sample, label)."""
    questions = []
    for idx, (text, sample_labels) in enumerate(zip(texts, labels)):
        for category, values in sample_labels.items():
            for value in values:
                value = value.strip()
                questions.append({
                    'sample': idx,
                    'category': category,
                    'label': value,
                    'claim': build_claim(category, value, ignore_role),
                    'text': text,
                })
    return questions


def build_messages(question):
    user = (
        f"Description:\n{question['text']}\n\n"
        f"Claim: {question['claim']}\n\n"
        "Does the description state the claim? Answer Yes or No."
    )
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]


class HFJudge:
    """Scores yes/no questions with a HuggingFace causal LM. Returns P(yes) per question."""

    def __init__(self, model_name, load_in_4bit=False, device=None, device_map=None):
        """device: e.g. 'cuda:1'. device_map='auto' splits a model too large for one GPU over all visible GPUs."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {'torch_dtype': torch.bfloat16 if str(self.device).startswith('cuda') else torch.float32}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4')
            kwargs['device_map'] = device_map or self.device
        elif device_map:
            kwargs['device_map'] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if 'device_map' in kwargs:
            self.device = self.model.device  # inputs go to the device holding the embeddings
        else:
            self.model.to(self.device)
        self.model.eval()

        self.yes_ids = self._token_ids(['Yes', 'yes', ' Yes', ' yes'])
        self.no_ids = self._token_ids(['No', 'no', ' No', ' no'])

    def _token_ids(self, words):
        # first non-blank token of each variant, deduplicated; sentencepiece tokenizers (llama, phi)
        # encode ' Yes' as ['▁', '▁Yes'] and the bare '▁' would otherwise be shared by yes and no
        ids = set()
        for w in words:
            tokens = self.tokenizer.encode(w, add_special_tokens=False)
            ids.add(next(t for t in tokens if self.tokenizer.decode([t]).strip()))
        return sorted(ids)

    def _prompt(self, messages):
        # enable_thinking is used by Qwen3 templates and ignored by the others
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def score(self, batch_messages):
        torch = self.torch
        prompts = [self._prompt(m) for m in batch_messages]
        enc = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            # only the last position is needed; full logits would cost batch x seq x vocab memory
            logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        yes = torch.logsumexp(logits[:, self.yes_ids], dim=-1)
        no = torch.logsumexp(logits[:, self.no_ids], dim=-1)
        return torch.sigmoid(yes - no).tolist()


def judge_questions(judge, questions, batch_size):
    for start in tqdm(range(0, len(questions), batch_size)):
        batch = questions[start:start + batch_size]
        probs = judge.score([build_messages(q) for q in batch])
        for q, p in zip(batch, probs):
            q['p_yes'] = p
    return questions


def summarize(questions, n_samples, threshold=0.5):
    per_category = defaultdict(lambda: [0, 0])
    per_sample = defaultdict(lambda: [0, 0])
    for q in questions:
        found = q['p_yes'] >= threshold
        q['found'] = found
        per_category[q['category']][0] += found
        per_category[q['category']][1] += 1
        per_sample[q['sample']][0] += found
        per_sample[q['sample']][1] += 1

    total_found = sum(v[0] for v in per_category.values())
    total = sum(v[1] for v in per_category.values())
    sample_recalls = [f / t for f, t in per_sample.values()]
    return {
        'label_recall': total_found / total if total else 0.0,
        'mean_sample_recall': sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0,
        'samples_all_labels_found': sum(f == t for f, t in per_sample.values()) / n_samples if n_samples else 0.0,
        'n_samples': n_samples,
        'n_labels': total,
        'per_category': {k: {'recall': f / t, 'found': f, 'total': t} for k, (f, t) in per_category.items()},
    }


def print_summary(name, summary):
    print(f'\n=== {name} ===')
    print(f"label recall:             {summary['label_recall']:.4f} ({summary['n_labels']} labels)")
    print(f"mean per-sample recall:   {summary['mean_sample_recall']:.4f}")
    print(f"samples w/ all labels:    {summary['samples_all_labels_found']:.4f} ({summary['n_samples']} samples)")
    for category, stats in summary['per_category'].items():
        print(f"  {category:28s} {stats['recall']:.4f} ({stats['found']}/{stats['total']})")


def run(data, judge, texts=('preds',), batch_size=8, threshold=0.5, ignore_role=False, max_samples=None):
    labels = data['labels'][:max_samples]
    results = {}
    for key in texts:
        questions = build_questions(data[key][:max_samples], labels, ignore_role)
        judge_questions(judge, questions, batch_size)
        summary = summarize(questions, len(labels), threshold)
        results[key] = {'summary': summary, 'questions': questions}
    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', default='geo/eval/selected.json', help='json with refs, preds and labels')
    parser.add_argument('--output', default=None, help='output json, defaults to <input>_llm_eval.json')
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct', help='HuggingFace instruct model')
    parser.add_argument('--load_in_4bit', action='store_true', help='4-bit quantization (needs bitsandbytes)')
    parser.add_argument('--texts', nargs='+', default=['preds'], choices=['preds', 'refs'],
                        help='which texts to judge; judging refs measures the reliability of the judge')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--threshold', default=0.5, type=float, help='P(yes) threshold to count a label as found')
    parser.add_argument('--ignore_role', action='store_true',
                        help='only check that the label is mentioned, not whether it is main/secondary')
    parser.add_argument('--max', default=None, type=int, help='evaluate only the first N samples')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    judge = HFJudge(args.model, load_in_4bit=args.load_in_4bit)
    results = run(data, judge, args.texts, args.batch_size, args.threshold, args.ignore_role, args.max)

    for key, result in results.items():
        print_summary(key, result['summary'])

    output = args.output or os.path.splitext(args.input)[0] + '_llm_eval.json'
    results = {'model': args.model, 'threshold': args.threshold, 'ignore_role': args.ignore_role, **results}
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nsaved to {output}')
