
import nltk
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
import os
from glob import glob

# Download necessary NLTK data (only first time)
nltk.download('wordnet')
nltk.download('omw-1.4')


def evaluate(predictions, references):
    """
    predictions: List[str]
    references: List[List[str]] (multiple references per prediction)
    """

    assert len(predictions) == len(references), "Mismatched lengths"

    # ---------------- Tokenization ----------------
    tokenized_preds = [pred.split() for pred in predictions]
    tokenized_refs = [[ref.split() for ref in refs] for refs in references]

    smoothie = SmoothingFunction().method4

    # ---------------- BLEU (1 to 4) ----------------
    bleu1 = corpus_bleu(tokenized_refs, tokenized_preds,
                        weights=(1, 0, 0, 0),
                        smoothing_function=smoothie)

    bleu2 = corpus_bleu(tokenized_refs, tokenized_preds,
                        weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smoothie)

    bleu3 = corpus_bleu(tokenized_refs, tokenized_preds,
                        weights=(1/3, 1/3, 1/3, 0),
                        smoothing_function=smoothie)

    bleu4 = corpus_bleu(tokenized_refs, tokenized_preds,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothie)

    # ---------------- METEOR ----------------
    meteor_scores = []
    for pred, refs in zip(predictions, references):
        tokenized_pred = pred.split()
        tokenized_refs = [r.split() for r in refs]

        meteor_scores.append(
            meteor_score(tokenized_refs, tokenized_pred)
        )

    meteor_avg = np.mean(meteor_scores)

    # ---------------- ROUGE ----------------
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1, rouge2, rougeL = [], [], []

    for pred, refs in zip(predictions, references):
        scores = [scorer.score(ref, pred) for ref in refs]

        rouge1.append(max(s['rouge1'].fmeasure for s in scores))
        rouge2.append(max(s['rouge2'].fmeasure for s in scores))
        rougeL.append(max(s['rougeL'].fmeasure for s in scores))

    rouge1_avg = np.mean(rouge1)
    rouge2_avg = np.mean(rouge2)
    rougeL_avg = np.mean(rougeL)

    # ---------------- Results ----------------
    results = {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "METEOR": meteor_avg,
        "ROUGE-1": rouge1_avg,
        "ROUGE-2": rouge2_avg,
        "ROUGE-L": rougeL_avg
    }

    return results

if __name__ == "__main__":
    # arg = '/nethome/recpinfo/users/fibz/data/checkpoints/capincho/nwpu/b512-t5-llama-3.2-1B-large/generated_captions.json'
    arg = '/nethome/recpinfo/users/fibz/data/checkpoints/fossil/nwpu/dinosat-llama3.2_1B-mapper/*3150_last.json'
    paths = glob(arg)
    
    for path in paths:
        print(f'computing metrics for {path}')
        data = json.load(open(path, 'r'))
        ref = []
        pred = []
        for i, e in enumerate(data['generated']):
            pred.append(e['prediction'])
            ref.append(e['reference'])
        
        
        results = evaluate(pred, ref)

        print(results)

        with open(os.path.join(os.path.dirname(path), 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
