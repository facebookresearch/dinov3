# LLM judge comparison for `eval_labels_llm.py`

*2026-09-29 · data: `geo/eval/selected.json` (670 samples, 2,652 annotated labels) · GPU: RTX 4060 Laptop, 8 GB*

## Summary

- **Recommended judge: `Qwen/Qwen2.5-7B-Instruct` with `--load_in_4bit`.** It separates true from false claims best (AUC 0.989, balanced accuracy 0.964 at the default 0.5 threshold) and uses 5.8 GB of VRAM.
- **`Qwen/Qwen3-8B` in 4-bit is just as good** (balanced accuracy 0.967). It agrees with Qwen2.5-7B on 95.7% of prediction labels (Cohen's κ = 0.91).
- **According to the two strong judges, the captioning model's label recall is 0.42–0.44.** Only about 9% of samples have every annotated label stated in the generated text.
- **The current default judge, `Qwen2.5-3B-Instruct`, is too strict at a threshold of 0.5.** It rejects 19% of the labels in the reference texts, so it reports a preds recall of 0.37 instead of about 0.43.
- **Small judges overstate the recall.** Qwen2.5-0.5B accepts 74% of false claims, and it accepts every false constituent claim. Its preds recall of 0.83 has no meaning.

## Method

Each judge answered three sets of yes/no questions. Every question pairs one text with one label. The answer is P(Yes) from the next-token logits, and a label counts as found when P(Yes) ≥ 0.5.

| Set | Text | Labels | What it measures |
|---|---|---|---|
| `preds` | generated description | the sample's own labels (2,652) | **what we want to know:** label recall of the captioning model |
| `refs` | reference description | the sample's own labels (2,652) | judge sensitivity (TPR). The references were written from the labels, so they should almost always be judged "Yes" |
| `refs_negative` | reference description | labels of the same category taken from other samples and not annotated for this one (2,548) | judge false-positive rate (FPR) |

Without the negative set, a judge that always answers "Yes" would look perfect on `refs`. The negative set is somewhat noisy: a few borrowed labels are near-synonyms of the true ones, such as *Arbustos/Dígitos* for a sample annotated *Arbustos*, or *Esférulas* for *Esferulitos*. The real FPRs are therefore a little lower than those reported. The **lexical** baseline counts a label as found when it appears verbatim in the text.

## Judge reliability

| Judge | Quant. | Refs TPR ↑ | Neg. FPR ↓ | Bal. acc. @0.5 ↑ | AUC ↑ | Peak VRAM | Time (7,852 q) |
|---|---|---|---|---|---|---|---|
| lexical match | – | 0.893 | 0.033 | 0.930 | – | – | <1 s |
| Qwen2.5-0.5B-Instruct | bf16 | 0.994 | **0.736** | 0.629 | 0.836 | 2.1 GB | 1.5 min |
| Qwen2.5-1.5B-Instruct | bf16 | 0.953 | 0.161 | 0.896 | 0.953 | 4.2 GB | 4.2 min |
| Qwen2.5-3B-Instruct *(current default)* | bf16 | **0.806** | 0.051 | 0.877 | 0.968 | 6.5 GB | 8.1 min |
| Phi-3.5-mini-instruct | 4-bit | 0.960 | 0.082 | 0.939 | 0.973 | 3.3 GB | 10.2 min |
| **Qwen2.5-7B-Instruct** | 4-bit | 0.967 | **0.038** | 0.964 | **0.989** | 5.8 GB | 18.3 min |
| **Qwen3-8B** | 4-bit | **0.977** | 0.043 | **0.967** | 0.986 | 6.4 GB | 20.4 min |

- **AUC does not depend on the threshold.** Qwen2.5-3B has a good AUC (0.968), but its P(Yes) is shifted low. With a well-chosen threshold it reaches a balanced accuracy of 0.935, but 0.5 is the wrong threshold for it.
- **Lexical matching is precise but misses paraphrases.** In the references it misses 11% of labels because of wording changes such as "bioclastos de artrópodes ostracodes" for *Bioclastos Artrópodes Ostracode*. In generated text, which is worded more loosely, it misses more.

False-positive rate by category on the negative set:

| Judge | Const. principais | Const. secundários | Tamanho | Comp. atual |
|---|---|---|---|---|
| Qwen2.5-0.5B | 1.000 | 1.000 | 0.034 | 0.454 |
| Qwen2.5-1.5B | 0.245 | 0.215 | 0.000 | 0.042 |
| Qwen2.5-3B | 0.046 | 0.121 | 0.000 | 0.000 |
| Phi-3.5-mini | 0.077 | 0.151 | 0.000 | 0.081 |
| Qwen2.5-7B | 0.065 | 0.047 | 0.000 | 0.000 |
| Qwen3-8B | 0.049 | 0.060 | 0.000 | 0.054 |

Most errors on constituents come from the main/secondary role: when a constituent is mentioned in the text, weaker judges accept it in either role.

## Captioning model results (`preds`)

| Judge | Label recall | Samples with all labels | Principais | Secundários | Tamanho | Comp. atual |
|---|---|---|---|---|---|---|
| lexical match | 0.457 | 0.107 | 0.460 | 0.284 | 0.501 | 0.701 |
| Qwen2.5-0.5B | *0.827* | *0.621* | *0.991* | *0.909* | 0.496 | 0.735 |
| Qwen2.5-1.5B | *0.556* | 0.224 | 0.699 | 0.382 | 0.442 | 0.665 |
| Qwen2.5-3B | *0.373* | 0.081 | 0.480 | 0.275 | 0.212 | 0.509 |
| Phi-3.5-mini | 0.454 | 0.106 | 0.505 | 0.291 | 0.455 | 0.613 |
| **Qwen2.5-7B** | **0.423** | **0.084** | **0.453** | **0.208** | **0.446** | **0.701** |
| **Qwen3-8B** | **0.438** | **0.093** | **0.486** | **0.211** | **0.455** | **0.704** |

*Italics: judges that are unreliable for that category (FPR above 15%, or TPR below 0.85).*

- **Composition** is the category the model states best (about 0.70). **Secondary constituents** are the weakest (about 0.21): the model rarely names them, or names them in the wrong role.
- **Element size** comes out at about 0.45 with every reliable judge.

## Recommendations

1. Make `Qwen/Qwen2.5-7B-Instruct --load_in_4bit` the default judge in `eval_labels_llm.py`. On `preds` alone (2,652 questions) it runs in about 6 minutes. Use Qwen3-8B as a second judge when a result needs confirming.
2. If VRAM or time is limited, use `Phi-3.5-mini-instruct --load_in_4bit` (3.3 GB). It is slightly lenient: expect preds recall about 0.03 higher than with the 7B.
3. Do not use judges smaller than 3B. If you keep Qwen2.5-3B, recalibrate the threshold on the refs and negative sets first.
4. Report the judge's refs TPR and negative-set FPR alongside every preds number. `compare_judges.py` produces both.

## Notes

- **Bug fixed during this run.** `HFJudge._token_ids` took the first token of `" Yes"` and `" No"`. SentencePiece tokenizers (Phi, Llama, Mistral) encode these as `['▁', '▁Yes']`, so the bare `'▁'` token went into both the Yes and No sets and pulled every P(Yes) toward 0.5. It now takes the first non-blank token. Qwen tokenizers were not affected, so earlier Qwen results remain valid.
- **Environment.** The runs used the base conda env with `KMP_DUPLICATE_LIB_OK=TRUE`, which works around the conflicting OpenMP runtimes of torch and MKL.
- **Reproduce.** `python geo/eval/compare_judges.py` writes one JSON per judge, with summaries and P(Yes) for every question, to `geo/eval/judge_comparison/`. Runs that already exist are skipped.
