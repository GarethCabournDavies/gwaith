"""Plot match distributions from collated or individual bank verification reports."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def plot_report(report_path, output_dir, bins=50):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    rows = report['results']
    matches = np.asarray([r['match'] for r in rows if not r['error']], float)
    if not np.isfinite(matches).all():
        raise ValueError(f'Nonfinite matches in {report_path}')
    threshold = report['provenance']['min_match']
    failures = sum(bool(r['error']) for r in rows)
    below = f'{np.mean(matches < threshold):.2%}' if len(matches) else 'N/A'
    # Shared match bins make the shard histograms directly comparable.
    edges = np.linspace(0., max(1., float(matches.max()) if len(matches) else 1.), bins + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(matches, bins=edges, color='steelblue', edgecolor='white')
    axes[0].set(ylabel='Injection count', title='Match histogram')
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    if len(matches):
        sorted_matches = np.sort(matches)
        axes[1].step(np.r_[0., sorted_matches, edges[-1]],
                     np.r_[0., np.arange(1, len(matches) + 1) / len(matches), 1.],
                     where='post', color='steelblue')
    else:
        axes[0].text(.5, .5, 'No successful injections', ha='center', transform=axes[0].transAxes)
    axes[1].set(ylabel='Fraction of successful injections', ylim=(0., 1.), title='Cumulative match distribution')
    for ax in axes:
        ax.axvline(threshold, color='darkorange', linestyle='--', label=f'Minimal match = {threshold:g}')
        ax.set(xlabel='Best bank match (fitting factor)', xlim=(edges[0], edges[-1]))
        ax.legend(loc='upper left')
        ax.grid(alpha=.2)
    fig.suptitle(f'{report_path.stem}: {len(matches):,} recovered, {failures:,} failed; {below} below minimal match')
    output = Path(output_dir) / (report_path.stem + '.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)
    logger.debug('Plotted %d successful matches from %s', len(matches), report_path)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('reports', nargs='+')
    args = parser.parse_args()
    if args.bins < 1:
        parser.error('bins must be positive')
    stems = [Path(p).stem for p in args.reports]
    if len(set(stems)) != len(stems):
        parser.error('Report names must be unique within the output directory')
    for report in args.reports:
        print(plot_report(report, args.output_dir, args.bins))


if __name__ == '__main__':
    main()
