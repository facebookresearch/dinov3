"""
Tests for eval_labels_llm.py.

The pipeline tests use a stub judge and run without torch. The last test loads a real small model
(Qwen2.5-0.5B-Instruct by default, override with EVAL_LLM_TEST_MODEL) and is skipped when torch is unavailable.

    pytest geo/eval/test_eval_labels_llm.py -v
"""
import json
import os
import re

import pytest

from eval_labels_llm import build_claim, build_messages, build_questions, run, summarize

DATA_PATH = os.path.join(os.path.dirname(__file__), 'selected.json')


class StubJudge:
    """Says yes when the label value (lowercase, alphanumeric part only) appears in the description."""

    def score(self, batch_messages):
        probs = []
        for messages in batch_messages:
            user = messages[-1]['content']
            text = user.split('Description:\n')[1].split('\n\nClaim:')[0].lower()
            value = re.search(r'"(.+?)"', user.split('Claim:')[1]).group(1).lower()
            value = value.split(':')[0]  # 'muito pequeno: < 0,2 cm' -> 'muito pequeno'
            probs.append(1.0 if value in text else 0.0)
        return probs


@pytest.fixture
def toy_data():
    return {
        'refs': ['A lâmina apresenta arbustos como constituintes principais. Os elementos são compostos por calcita.',
                 'A lâmina apresenta pelóides como constituintes principais.'],
        'preds': ['A lâmina apresenta arbustos como constituintes principais. Os elementos são compostos por dolomita.',
                  'A lâmina apresenta esferulitos como constituintes principais.'],
        'labels': [{'Constituintes Principais': [' Arbustos'], 'Comp. Atual do Elemento': [' Calcita']},
                   {'Constituintes Principais': [' Pelóides']}],
    }


def test_build_questions_flattens_labels(toy_data):
    questions = build_questions(toy_data['preds'], toy_data['labels'])
    assert len(questions) == 3
    assert [q['sample'] for q in questions] == [0, 0, 1]
    assert questions[0]['label'] == 'Arbustos'  # leading space stripped
    assert 'main constituent' in questions[0]['claim']
    assert 'composed of "Calcita"' in questions[1]['claim']


def test_claims():
    assert 'secondary constituent' in build_claim('Constituintes Secundários', 'Esferulitos')
    assert 'mentioned' in build_claim('Constituintes Secundários', 'Esferulitos', ignore_role=True)
    assert 'mentioned' in build_claim('Categoria Desconhecida', 'X')


def test_messages_contain_text_and_claim(toy_data):
    q = build_questions(toy_data['preds'], toy_data['labels'])[0]
    messages = build_messages(q)
    assert messages[0]['role'] == 'system'
    assert q['text'] in messages[1]['content'] and q['claim'] in messages[1]['content']


def test_run_with_stub_judge(toy_data):
    results = run(toy_data, StubJudge(), texts=('preds', 'refs'), batch_size=2)
    preds, refs = results['preds']['summary'], results['refs']['summary']
    assert preds['label_recall'] == pytest.approx(1 / 3)
    assert preds['samples_all_labels_found'] == 0.0
    assert preds['per_category']['Constituintes Principais']['found'] == 1
    assert refs['label_recall'] == 1.0
    assert refs['samples_all_labels_found'] == 1.0


def test_summarize_threshold():
    questions = [{'sample': 0, 'category': 'A', 'p_yes': 0.6}, {'sample': 1, 'category': 'A', 'p_yes': 0.4}]
    assert summarize(questions, 2, threshold=0.5)['label_recall'] == 0.5
    assert summarize(questions, 2, threshold=0.7)['label_recall'] == 0.0


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason='selected.json not found')
def test_run_on_selected_json_with_stub():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert len(data['refs']) == len(data['preds']) == len(data['labels'])
    results = run(data, StubJudge(), texts=('preds', 'refs'), batch_size=64)
    # references were written from the labels, so they must score higher than predictions
    assert results['refs']['summary']['label_recall'] > results['preds']['summary']['label_recall']
    assert results['refs']['summary']['n_labels'] == sum(len(v) for l in data['labels'] for v in l.values())


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # ImportError or DLL load errors
        return False


@pytest.mark.skipif(not _torch_available(), reason='torch not available')
def test_real_model_small():
    from eval_labels_llm import HFJudge

    judge = HFJudge(os.environ.get('EVAL_LLM_TEST_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct'))
    text = 'A lâmina apresenta arbustos como constituintes principais. Os elementos são compostos por calcita.'
    positive = build_messages({'text': text, 'claim': build_claim('Constituintes Principais', 'Arbustos')})
    negative = build_messages({'text': text, 'claim': build_claim('Constituintes Principais', 'Bioclastos Artrópodes Ostracode')})
    p_pos, p_neg = judge.score([positive, negative])  # batched, exercises left padding
    assert 0.0 <= p_neg <= 1.0 and 0.0 <= p_pos <= 1.0
    assert p_pos > p_neg
