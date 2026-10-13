"""
Scores of the human qualitative evaluation (resultados/primeiro/*.docx): mean, std, min and max per model.

Every document has one section per image:
    <ID>.png
    [image]
    original: <reference text>
    Labels: {...}                  (only in some documents)
    <model name>: <generated text>  (one line per model)
    table: Descrição | Nota (1-5) | Observações, with rows "modelo 1", "modelo 2", ...
"modelo N" is the N-th model line of the section. Scores use a decimal comma; empty scores are skipped.

Images are checked for duplicates both by ID and by the bytes of the embedded picture, within and across documents.
By default only the first occurrence of a duplicated image is used in the metrics (--keep_duplicates uses all).

Usage:
    python geo/eval/human_eval_scores.py --input resultados/primeiro
    python geo/eval/human_eval_scores.py --input resultados/primeiro --csv resultados/primeiro/scores.csv
"""
import csv
import glob
import hashlib
import os
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile
from argparse import ArgumentParser
from collections import defaultdict

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
REL = '{http://schemas.openxmlformats.org/package/2006/relationships}'
NOT_MODELS = {'original', 'labels'}


def text_of(element):
    return ''.join(t.text or '' for t in element.iter(W + 't')).strip()


def parse_score(value):
    value = value.replace(' ', '').replace(',', '.')
    return float(value) if value else None


def image_hashes(archive):
    """relationship id -> sha1 of the embedded image."""
    rels = ET.fromstring(archive.read('word/_rels/document.xml.rels'))
    hashes = {}
    for rel in rels.iter(REL + 'Relationship'):
        if rel.get('Type', '').endswith('/image'):
            hashes[rel.get('Id')] = hashlib.sha1(archive.read('word/' + rel.get('Target').lstrip('/'))).hexdigest()
    return hashes


def parse_document(path):
    """One dict per image section: id, image hash, model names and scores (None when empty)."""
    with zipfile.ZipFile(path) as archive:
        hashes = image_hashes(archive)
        body = ET.fromstring(archive.read('word/document.xml')).find(W + 'body')

    sections = []
    for element in body:
        if element.tag == W + 'p':
            text = text_of(element)
            blips = [b.get(R + 'embed') for b in element.iter(A + 'blip')]
            if text.lower().endswith('.png'):
                sections.append({'file': os.path.basename(path), 'id': text[:-4], 'image': None, 'models': [],
                                 'scores': {}})
            elif not sections:
                continue
            elif blips:
                sections[-1]['image'] = hashes.get(blips[0])
            elif ':' in text:
                name = text.split(':', 1)[0].strip()
                if len(name) <= 40 and name.lower() not in NOT_MODELS:
                    sections[-1]['models'].append(name)
        elif element.tag == W + 'tbl' and sections:
            section = sections[-1]
            for row in element.iter(W + 'tr'):
                cells = [text_of(c) for c in row.findall(W + 'tc')]
                match = re.fullmatch(r'modelo\s*(\d+)', cells[0].lower()) if cells else None
                if not match:
                    continue
                n = int(match.group(1))
                if n > len(section['models']):
                    raise ValueError(f"{section['file']} {section['id']}: table row 'modelo {n}' but only "
                                     f"{len(section['models'])} model descriptions")
                section['scores'][section['models'][n - 1]] = parse_score(cells[1]) if len(cells) > 1 else None
    return sections


def find_duplicates(sections):
    """Groups of sections sharing an ID or an image, as lists of indices into sections."""
    groups = []
    for key in ('id', 'image'):
        by_key = defaultdict(list)
        for i, s in enumerate(sections):
            if s[key]:
                by_key[s[key]].append(i)
        groups += [(key, idx) for idx in by_key.values() if len(idx) > 1]
    return groups


def describe(values):
    if not values:
        return {'n': 0, 'mean': float('nan'), 'std': float('nan'), 'min': float('nan'), 'max': float('nan')}
    return {'n': len(values), 'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0, 'min': min(values), 'max': max(values)}


def print_table(title, rows):
    print(f'\n=== {title} ===')
    print(f"{'model':28s} {'n':>4s} {'mean':>7s} {'std':>7s} {'min':>6s} {'max':>6s}")
    for name, d in rows.items():
        print(f"{name:28s} {d['n']:4d} {d['mean']:7.3f} {d['std']:7.3f} {d['min']:6.2f} {d['max']:6.2f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', default='resultados/primeiro', help='folder with the evaluation .docx files')
    parser.add_argument('--keep_duplicates', action='store_true', help='use every occurrence of duplicated images')
    parser.add_argument('--csv', default=None, help='also save the extracted scores to this csv')
    args = parser.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.input, '*.docx')) if not os.path.basename(f).startswith('~$'))
    sections = [s for f in files for s in parse_document(f)]
    models = list(dict.fromkeys(m for s in sections for m in s['models']))
    print(f'{len(sections)} images in {len(files)} files: ' +
          ', '.join(f"{os.path.basename(f)} ({sum(s['file'] == os.path.basename(f) for s in sections)})" for f in files))

    # ---- duplicates
    duplicated = set()
    groups = find_duplicates(sections)
    print(f'\n=== duplicates ({len(groups)}) ===' if groups else '\nno duplicated images')
    for key, idx in groups:
        print(f"same {key}: " + ' | '.join(f"{sections[i]['file']} {sections[i]['id']}" for i in idx))
        scores = [tuple(sections[i]['scores'].get(m) for m in models) for i in idx]
        if len(set(scores)) > 1:
            print('    scores differ: ' + ' | '.join(str(s) for s in scores))
        duplicated.update(idx[1:])

    # ---- metrics
    used = [s for i, s in enumerate(sections) if args.keep_duplicates or i not in duplicated]
    missing = {m: sum(s['scores'].get(m) is None for s in used) for m in models}
    print(f"\nusing {len(used)} images ({'duplicates kept' if args.keep_duplicates else 'duplicates removed'}); "
          'empty scores skipped: ' + ', '.join(f'{m}: {n}' for m, n in missing.items()))

    print_table('all files', {m: describe([s['scores'][m] for s in used if s['scores'].get(m) is not None])
                              for m in models})
    for f in files:
        name = os.path.basename(f)
        print_table(name, {m: describe([s['scores'][m] for s in used
                                        if s['file'] == name and s['scores'].get(m) is not None]) for m in models})

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            writer.writerow(['file', 'id', 'image_sha1', 'duplicate'] + models)
            for i, s in enumerate(sections):
                writer.writerow([s['file'], s['id'], s['image'], i in duplicated] +
                                ['' if s['scores'].get(m) is None else s['scores'][m] for m in models])
        print(f'\nsaved to {args.csv}')
