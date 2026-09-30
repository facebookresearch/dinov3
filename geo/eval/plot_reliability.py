"""
Plots the judge comparison in the summary.json written by `judge_reliability.py merge`.

    python geo/eval/plot_reliability.py --summary geo/eval/reliability/summary.json

Writes PNG and PDF figures to <summary dir>/plots/:
    error_rates       missed labels, accepted absent labels and accepted role swaps per judge (log scale)
    quality_vs_size   balanced accuracy and AUC against model size
    category_errors   misses and false positives per judge and label category
    agreement         Cohen's kappa between judges
and a markdown table with the plotted numbers (table.md).
"""
import json
import os
import re
from argparse import ArgumentParser

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# reference palette (dataviz skill): first three categorical slots, validated all-pairs in light mode
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
BLUE_RAMP = ['#f3f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
ORANGE_RAMP = ['#fef5f0', '#fbd9c8', '#f6b394', '#f08c61', '#eb6834', '#c9501f', '#9c3c15', '#6e2a0e']
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'

# parameters in billions, for the size axis
SIZES = {'Phi-3.5-mini-instruct': 3.8, 'Qwen2.5-0.5B-Instruct': 0.5, 'Qwen2.5-1.5B-Instruct': 1.5,
         'Qwen2.5-3B-Instruct': 3.1, 'Qwen2.5-7B-Instruct': 7.6, 'Qwen2.5-14B-Instruct': 14.7,
         'Qwen2.5-32B-Instruct': 32.5, 'Qwen2.5-72B-Instruct': 72.7, 'Qwen3-8B': 8.2, 'Qwen3-14B': 14.8,
         'Qwen3-32B': 32.8, 'Llama-3.2-3B-Instruct': 3.2, 'Llama-3.1-8B-Instruct': 8.0,
         'Llama-3.3-70B-Instruct': 70.6}
# colour follows the model family (fixed slot order, never by rank); a 4th family would need facets instead
FAMILIES = [('Qwen', SERIES[0]), ('Llama', SERIES[1]), ('Other', SERIES[2])]


def family(tag):
    name = tag.split('__', 1)[-1]
    return next((f for f, _ in FAMILIES[:-1] if name.startswith(f)), 'Other')


def short_name(tag):
    name = tag.split('__', 1)[-1]
    four_bit = name.endswith('-4bit')
    name = name.removesuffix('-4bit').removesuffix('-Instruct').removesuffix('-instruct')
    return name + (' (4-bit)' if four_bit else '')


def model_size(tag):
    name = tag.split('__', 1)[-1].removesuffix('-4bit')
    if name in SIZES:
        return SIZES[name]
    m = re.search(r'(\d+(?:\.\d+)?)B', name)
    return float(m.group(1)) if m else None


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'], 'font.size': 10,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 12,
        'axes.titleweight': 'semibold', 'axes.titlelocation': 'left', 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.8, 'grid.linestyle': '-', 'axes.axisbelow': True, 'xtick.color': MUTED,
        'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'legend.frameon': False,
        'legend.labelcolor': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    })


def save(fig, out_dir, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=9, va='bottom')


def plot_error_rates(models, order, out_dir, floor=1e-4):
    """Dot plot, one row per judge: the three ways a judge can be wrong, on a log scale."""
    rows = [
        ('Missed a true label (1 − TPR)', lambda m: 1 - m['tpr']),
        ('Accepted an absent label', lambda m: m['fpr_by_kind'].get('absent', np.nan)),
        ('Accepted a role swap', lambda m: m['fpr_by_kind'].get('role_swap', np.nan)),
    ]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(order) + 1.8))
    y = np.arange(len(order))
    offsets = (-0.2, 0, 0.2)
    for (label, fn), color, dy in zip(rows, SERIES, offsets):
        vals = np.array([max(fn(models[t]), floor) for t in order])
        ax.scatter(vals, y + dy, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
    ax.set_xscale('log')
    ax.set_xlim(floor * 0.8, 1)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(['≤0.01%', '0.1%', '1%', '10%', '100%'])
    ax.set_yticks(y, [short_name(t) for t in order])
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Error rate (log scale; lower is better)')
    ax.set_title('How each judge goes wrong', pad=28)
    subtitle(ax, 'False positives (orange, aqua) are what an RL policy can exploit; misses (blue) only add noise')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12 - 0.3 / len(order)), ncol=3, fontsize=9,
              handletextpad=0.2, columnspacing=1.2)
    save(fig, out_dir, 'error_rates')


def plot_quality_vs_size(models, order, out_dir):
    """Balanced accuracy and AUC against size: two small multiples sharing the x axis, one y scale each."""
    metrics = [('balanced_accuracy', 'Balanced accuracy at threshold 0.5'), ('auc', 'AUC (threshold-free)')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)
    sized = [t for t in order if model_size(t)]
    colors = dict(FAMILIES)
    labels = []
    for ax, (key, title) in zip(axes, metrics):
        for t in sized:
            x, y = model_size(t), models[t][key]
            ax.scatter(x, y, s=64, color=colors[family(t)], marker='s' if t.endswith('-4bit') else 'o',
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            labels.append((ax, key, t, ax.annotate(short_name(t).replace(' (4-bit)', ''), (x, y), xytext=(6, 0),
                                                   textcoords='offset points', va='center', fontsize=8,
                                                   color=INK2)))
        sizes = [model_size(t) for t in sized]
        ticks = [s for s in (2, 4, 8, 16, 32, 64, 128) if min(sizes) / 1.5 <= s <= max(sizes) * 1.5]
        ax.set_xscale('log')
        ax.set_xticks(ticks, [f'{s}B' for s in ticks])
        ax.minorticks_off()
        ax.set_xlim(min(sizes) / 1.3, max(sizes) * 2.2)  # room for the labels right of the points
        ax.set_xlabel('Parameters (log scale)')
        ax.set_title(title, fontsize=11)
    present = [f for f, _ in FAMILIES if any(family(t) == f for t in sized)]
    others = sorted({t.split('__', 1)[-1].split('-')[0] for t in sized if family(t) == 'Other'})
    names = {'Other': ' / '.join(others) if 0 < len(others) <= 2 else 'Other'}
    handles = [plt.Line2D([], [], ls='', marker='o', ms=8, color=colors[f], label=names.get(f, f)) for f in present]
    handles += [plt.Line2D([], [], ls='', marker=m, ms=8, color=MUTED, label=q) for m, q in (('o', 'bf16'), ('s', '4-bit'))
                if any(t.endswith('-4bit') == (q == '4-bit') for t in sized)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Judge quality against model size', x=0.01, ha='left', fontsize=12, fontweight='semibold',
                 color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    place_labels(fig, labels, sized, models)
    save(fig, out_dir, 'quality_vs_size')


def place_labels(fig, labels, sized, models):
    """Puts each point label right, above, below or left of its point: the first spot, measured on the final
    layout, that covers no other point, no label placed before it and stays inside the axes."""
    renderer = fig.canvas.get_renderer()
    spots = [((6, 0), 'left', 'center'), ((0, 7), 'center', 'bottom'), ((0, -7), 'center', 'top'),
             ((-6, 0), 'right', 'center')]
    placed = []
    for ax, key, t, text in labels:
        points = [ax.transData.transform((model_size(o), models[o][key])) for o in sized if o != t]
        frame = ax.get_window_extent(renderer)
        for offset, ha, va in spots:
            text.set_position(offset)
            text.set_ha(ha)
            text.set_va(va)
            box = text.get_window_extent(renderer)
            clear = (not any(box.x0 - 5 <= px <= box.x1 + 5 and box.y0 - 5 <= py <= box.y1 + 5 for px, py in points)
                     and not any(box.overlaps(b) for b in placed)
                     and frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1)
            if clear:
                break
        else:
            text.set_position(spots[0][0])
            text.set_ha(spots[0][1])
            text.set_va(spots[0][2])
        placed.append(text.get_window_extent(renderer))


def heatmap(ax, matrix, row_labels, col_labels, vmin, vmax, fmt, ramp=BLUE_RAMP):
    cmap = LinearSegmentedColormap.from_list('ramp', ramp)
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)  # surface gap between cells
    ax.tick_params(which='minor', length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            dark = (v - vmin) / (vmax - vmin) > 0.55
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8.5, color='white' if dark else INK)


def plot_category_errors(models, order, out_dir, min_labels=100):
    """Misses and false positives per label category, side by side, each on its own scale."""
    ref = models[order[0]]['per_category']
    cats = sorted((c for c in ref if ref[c]['n_pos'] >= min_labels), key=lambda c: -ref[c]['n_pos'])
    short = {'Constituintes Principais': 'Main constituents', 'Constituintes Secundários': 'Secondary constituents',
             'Tamanho do Elemento': 'Element size', 'Comp. Atual do Elemento': 'Current composition'}
    col_labels = [f"{short.get(c, c)}\n(n={ref[c]['n_pos']:,})" for c in cats]
    rows = [short_name(t) for t in order]
    panels = [('Missed true labels (1 − TPR)', lambda v: 1 - v['tpr'], BLUE_RAMP),
              ('Accepted false labels (FPR)', lambda v: v['fpr'], ORANGE_RAMP)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(order) + 2), sharey=True)
    for ax, (title, fn, ramp) in zip(axes, panels):
        matrix = np.array([[fn(models[t]['per_category'][c]) if c in models[t]['per_category'] else np.nan
                            for c in cats] for t in order])
        heatmap(ax, matrix, rows, col_labels, vmin=0, vmax=max(np.nanmax(matrix), 0.05), fmt='{:.1%}', ramp=ramp)
        ax.tick_params(axis='x', labelsize=8.5)
        ax.set_title(title, fontsize=11)
    skipped = [c for c in ref if c not in cats]
    note = f" ({', '.join(skipped)} left out: fewer than {min_labels} labels)" if skipped else ''
    fig.suptitle('Judge errors by label category', x=0.01, ha='left', fontsize=12, fontweight='semibold', color=INK)
    fig.text(0.01, 0.905, 'Lower is better; each panel has its own colour scale' + note, color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out_dir, 'category_errors')


def plot_agreement(summary, order, out_dir):
    """Lower triangle only: the matrix is symmetric."""
    idx = {t: i for i, t in enumerate(order)}
    matrix = np.full((len(order), len(order)), np.nan)
    for pair, a in summary['agreement'].items():
        x, y = pair.split(' | ')
        if x in idx and y in idx:
            i, j = sorted((idx[x], idx[y]), reverse=True)
            matrix[i, j] = a['kappa']
    matrix = matrix[1:, :-1]
    labels = [short_name(t) for t in order]
    fig, ax = plt.subplots(figsize=(0.8 * len(order) + 3, 0.55 * len(order) + 1.5))
    lo = np.floor(np.nanmin(matrix) * 20) / 20
    heatmap(ax, matrix, labels[1:], labels[:-1], vmin=lo, vmax=1.0, fmt='{:.2f}')
    ax.set_xticklabels(labels[:-1], rotation=35, ha='right')
    ax.set_title("Agreement between judges (Cohen's κ)", pad=12)
    save(fig, out_dir, 'agreement')


def write_table(models, order, out_dir):
    lines = ['| Judge | Params | TPR | FPR absent | FPR role swap | Precision | Bal. acc. | AUC | Best thr. '
             '| Samples fully correct |', '|---|---|---|---|---|---|---|---|---|---|']
    for t in order:
        m = models[t]
        k = m['fpr_by_kind']
        size = model_size(t)
        lines.append(f"| {short_name(t)} | {f'{size:g}B' if size else '-'} | {m['tpr']:.4f} | "
                     f"{k.get('absent', float('nan')):.4f} | {k.get('role_swap', float('nan')):.4f} | "
                     f"{m['precision']:.4f} | {m['balanced_accuracy']:.4f} | {m['auc']:.4f} | "
                     f"{m['best_threshold']:.2f} | {m['samples_fully_correct']:.4f} |")
    with open(os.path.join(out_dir, 'table.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--summary', default='geo/eval/reliability/summary.json')
    parser.add_argument('--out_dir', default=None, help='defaults to <summary dir>/plots')
    args = parser.parse_args()

    with open(args.summary, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    models = summary['models']
    order = sorted(models, key=lambda t: -models[t]['balanced_accuracy'])
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.summary), 'plots')
    os.makedirs(out_dir, exist_ok=True)

    style()
    plot_error_rates(models, order, out_dir)
    plot_quality_vs_size(models, order, out_dir)
    plot_category_errors(models, order, out_dir)
    plot_agreement(summary, order, out_dir)
    write_table(models, order, out_dir)
    print(f'figures written to {out_dir}')


if __name__ == '__main__':
    main()
