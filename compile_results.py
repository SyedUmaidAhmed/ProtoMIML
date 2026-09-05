#!/usr/bin/env python3
"""Compile all ProtoMIML results into a comprehensive comparison report."""

import json
import os
import glob
import numpy as np

RESULTS_DIR = 'results'

# Published baselines (best per metric)
BASELINES = {
    'scene': {
        'HammingLoss': (0.165, 'MIML-kNN', 'lower'),
        'OneError': (0.310, 'MIML-kNN', 'lower'),
        'Coverage': (0.896, 'MIML-kNN', 'lower'),
        'RankingLoss': (0.159, 'MIML-kNN', 'lower'),
        'AveragePrecision': (0.801, 'MIML-kNN', 'higher'),
    },
    'reuters': {
        'HammingLoss': (0.033, 'MimlSvm', 'lower'),
        'OneError': (0.055, 'MimlSvmmi', 'lower'),
        'Coverage': (0.284, 'MimlSvmmi', 'lower'),
        'RankingLoss': (0.020, 'MimlSvmmi', 'lower'),
        'AveragePrecision': (0.965, 'MimlSvmmi', 'higher'),
    },
    'birdsong': {
        'HammingLoss': (0.105, 'CLAM++', 'lower'),
        'OneError': (0.272, 'CLAM++', 'lower'),
        'Coverage': (3.984, 'CLAM++', 'lower'),
        'RankingLoss': (0.104, 'CLAM++', 'lower'),
        'AveragePrecision': (0.723, 'CLAM++', 'higher'),
    },
    'yeast': {
        'HammingLoss': (0.189, 'MIML-kNN', 'lower'),
        'OneError': (0.214, 'MIML-kNN', 'lower'),
        'Coverage': (6.275, 'MIML-kNN', 'lower'),
        'RankingLoss': (0.163, 'MIML-kNN', 'lower'),
        'AveragePrecision': (0.774, 'MIML-kNN', 'higher'),
    },
    'letter_carroll': {
        'HammingLoss': (0.132, 'RankLossSIM', 'lower'),
        'OneError': (0.119, 'MIMLfast', 'lower'),
        'Coverage': (9.880, 'MIMLfast', 'lower'),
        'RankingLoss': (0.130, 'MIMLfast', 'lower'),
        'AveragePrecision': (0.715, 'MIMLfast', 'higher'),
    },
    'letter_frost': {
        'HammingLoss': (0.136, 'MIMLfast', 'lower'),
        'OneError': (0.151, 'MIMLfast', 'lower'),
        'Coverage': (9.672, 'RankLossSIM', 'lower'),
        'RankingLoss': (0.134, 'MIMLfast', 'lower'),
        'AveragePrecision': (0.704, 'MIMLfast', 'higher'),
    },
    'MSCV2': {
        'HammingLoss': (0.084, 'MIMLSVM', 'lower'),
        'OneError': (0.295, 'MIMLfast', 'lower'),
        'Coverage': (5.474, 'MIMLfast', 'lower'),
        'RankingLoss': (0.107, 'RankLossSIM', 'lower'),
        'AveragePrecision': (0.688, 'MIMLfast', 'higher'),
    },
    'protein_azotobacter': {
        'HammingLoss': (0.073, 'CLAM++', 'lower'),
        'OneError': (0.557, 'CLAM++', 'lower'),
        'Coverage': (13.570, 'CLAM++', 'lower'),
        'RankingLoss': (0.130, 'CLAM++', 'lower'),
        'AveragePrecision': (0.510, 'CLAM++', 'higher'),
    },
    'protein_geobacter': {
        'HammingLoss': (0.086, 'CLAM++', 'lower'),
        'OneError': (0.468, 'CLAM++', 'lower'),
        'Coverage': (8.183, 'CLAM++', 'lower'),
        'RankingLoss': (0.127, 'CLAM++', 'lower'),
        'AveragePrecision': (0.584, 'CLAM++', 'higher'),
    },
    'protein_haloarcula': {
        'HammingLoss': (0.110, 'CLAM++', 'lower'),
        'OneError': (0.560, 'CLAM++', 'lower'),
        'Coverage': (5.963, 'CLAM++', 'lower'),
        'RankingLoss': (0.136, 'CLAM++', 'lower'),
        'AveragePrecision': (0.558, 'CLAM++', 'higher'),
    },
    'protein_pyrococcus': {
        'HammingLoss': (0.075, 'CLAM++', 'lower'),
        'OneError': (0.532, 'CLAM++', 'lower'),
        'Coverage': (14.360, 'CLAM++', 'lower'),
        'RankingLoss': (0.128, 'CLAM++', 'lower'),
        'AveragePrecision': (0.544, 'CLAM++', 'higher'),
    },
}

DATASET_INFO = {
    'scene': {'samples': 2000, 'labels': 5, 'features': 15, 'domain': 'Image'},
    'reuters': {'samples': 2000, 'labels': 7, 'features': 243, 'domain': 'Text'},
    'birdsong': {'samples': 257, 'labels': 19, 'features': 38, 'domain': 'Audio'},
    'yeast': {'samples': 2417, 'labels': 14, 'features': 103, 'domain': 'Biology'},
    'letter_carroll': {'samples': 166, 'labels': 26, 'features': 16, 'domain': 'Text'},
    'letter_frost': {'samples': 144, 'labels': 26, 'features': 16, 'domain': 'Text'},
    'MSCV2': {'samples': 591, 'labels': 23, 'features': 48, 'domain': 'Image'},
    'protein_azotobacter': {'samples': 407, 'labels': 46, 'features': 216, 'domain': 'Biology'},
    'protein_geobacter': {'samples': 379, 'labels': 28, 'features': 216, 'domain': 'Biology'},
    'protein_haloarcula': {'samples': 304, 'labels': 21, 'features': 216, 'domain': 'Biology'},
    'protein_pyrococcus': {'samples': 425, 'labels': 52, 'features': 216, 'domain': 'Biology'},
}

METRICS = ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision']
METRIC_SHORT = {'HammingLoss': 'HL', 'OneError': 'OE', 'Coverage': 'Cov',
                'RankingLoss': 'RL', 'AveragePrecision': 'AP'}
DATASET_ORDER = ['scene', 'reuters', 'yeast', 'birdsong', 'letter_carroll',
                 'letter_frost', 'MSCV2', 'protein_azotobacter', 'protein_geobacter',
                 'protein_haloarcula', 'protein_pyrococcus']


def load_results():
    """Load the most complete incremental results for each dataset."""
    results = {}
    for dataset in DATASET_ORDER:
        # Find the highest fold count for this dataset
        pattern = os.path.join(RESULTS_DIR, f'incremental_{dataset}_*folds.json')
        files = glob.glob(pattern)
        if not files:
            continue
        # Sort by fold count (extract number from filename)
        def get_folds(f):
            base = os.path.basename(f)
            return int(base.split('_')[-1].replace('folds.json', ''))
        files.sort(key=get_folds)
        best_file = files[-1]
        folds = get_folds(best_file)
        with open(best_file) as f:
            data = json.load(f)
        # Extract metrics
        key = dataset if dataset in data else list(k for k in data.keys() if k != 'per_fold')[0]
        if key in data:
            results[dataset] = {
                'metrics': data[key],  # {metric: [mean, std]}
                'folds': folds,
                'per_fold': data.get('per_fold', []),
            }
    return results


def compare(results):
    """Compare results to baselines and return summary."""
    summary = {}
    for dataset in DATASET_ORDER:
        if dataset not in results or dataset not in BASELINES:
            continue
        r = results[dataset]['metrics']
        b = BASELINES[dataset]
        beats = 0
        details = {}
        for m in METRICS:
            if m not in r or m not in b:
                continue
            val = r[m][0]
            std = r[m][1]
            baseline, method, direction = b[m]
            won = val < baseline if direction == 'lower' else val > baseline
            if won:
                beats += 1
            details[m] = {
                'ours': val, 'std': std, 'baseline': baseline,
                'method': method, 'direction': direction, 'won': won
            }
        summary[dataset] = {'beats': beats, 'total': len(details), 'details': details,
                            'folds': results[dataset]['folds']}
    return summary


def generate_markdown_report(results, summary):
    """Generate markdown comparison tables."""
    lines = []

    # Overall summary table
    lines.append("## ProtoMIML: Comprehensive Results Report\n")
    lines.append("### Overall Summary\n")
    lines.append("| Dataset | Domain | Samples | Labels | Folds | Beats |")
    lines.append("|---------|--------|---------|--------|-------|-------|")
    total_beats = 0
    total_metrics = 0
    for dataset in DATASET_ORDER:
        if dataset not in summary:
            continue
        info = DATASET_INFO.get(dataset, {})
        s = summary[dataset]
        total_beats += s['beats']
        total_metrics += s['total']
        lines.append(f"| {dataset} | {info.get('domain', '?')} | {info.get('samples', '?')} | "
                      f"{info.get('labels', '?')} | {s['folds']} | **{s['beats']}/{s['total']}** |")
    lines.append(f"| **Total** | | | | | **{total_beats}/{total_metrics}** |")
    lines.append("")

    # Detailed per-dataset tables
    lines.append("### Detailed Results\n")
    for dataset in DATASET_ORDER:
        if dataset not in summary:
            continue
        s = summary[dataset]
        info = DATASET_INFO.get(dataset, {})
        lines.append(f"#### {dataset} ({info.get('domain', '')} — {info.get('samples', '?')} samples, "
                      f"{info.get('labels', '?')} labels) — **{s['beats']}/{s['total']} BEATS**\n")
        lines.append(f"| Metric | ProtoMIML | Baseline | Method | Status |")
        lines.append(f"|--------|----------|----------|--------|--------|")
        for m in METRICS:
            if m not in s['details']:
                continue
            d = s['details'][m]
            arrow = '↓' if d['direction'] == 'lower' else '↑'
            status = '**BEAT**' if d['won'] else 'miss'
            ms = METRIC_SHORT[m]
            lines.append(f"| {ms}{arrow} | {d['ours']:.4f} ± {d['std']:.4f} | {d['baseline']:.3f} | "
                          f"{d['method']} | {status} |")
        lines.append("")

    # LaTeX table for paper
    lines.append("### LaTeX Table (for paper)\n")
    lines.append("```latex")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{ProtoMIML results compared to best published baselines. "
                  "\\textbf{Bold} indicates ProtoMIML outperforms the baseline. "
                  "$\\downarrow$ = lower is better, $\\uparrow$ = higher is better.}")
    lines.append("\\label{tab:results}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{l|ccccc|c}")
    lines.append("\\toprule")
    lines.append("Dataset & HL $\\downarrow$ & OE $\\downarrow$ & Cov $\\downarrow$ & "
                  "RL $\\downarrow$ & AP $\\uparrow$ & Beats \\\\")
    lines.append("\\midrule")

    for dataset in DATASET_ORDER:
        if dataset not in summary:
            continue
        s = summary[dataset]
        row_parts = [dataset.replace('_', '\\_')]
        for m in METRICS:
            if m not in s['details']:
                row_parts.append('—')
                continue
            d = s['details'][m]
            val_str = f"{d['ours']:.4f}"
            if d['won']:
                val_str = f"\\textbf{{{val_str}}}"
            base_str = f"{d['baseline']:.3f}"
            row_parts.append(f"{val_str} / {base_str}")
        row_parts.append(f"{s['beats']}/{s['total']}")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main():
    results = load_results()
    print(f"Loaded results for {len(results)} datasets: {list(results.keys())}")

    missing = [d for d in DATASET_ORDER if d not in results]
    if missing:
        print(f"Missing results for: {missing}")

    summary = compare(results)

    # Print console summary
    print("\n" + "=" * 70)
    print("PROTO-MIML RESULTS SUMMARY")
    print("=" * 70)
    total_beats = 0
    total_metrics = 0
    for dataset in DATASET_ORDER:
        if dataset not in summary:
            continue
        s = summary[dataset]
        total_beats += s['beats']
        total_metrics += s['total']
        status = "BEAT ALL!" if s['beats'] == s['total'] else f"{s['beats']}/{s['total']}"
        print(f"  {dataset:<25} {s['folds']:>2}-fold  {status}")
        for m in METRICS:
            if m not in s['details']:
                continue
            d = s['details'][m]
            flag = " <-- BEAT" if d['won'] else ""
            print(f"    {METRIC_SHORT[m]:<5} {d['ours']:.4f} ± {d['std']:.4f}  "
                  f"(baseline: {d['baseline']:.3f} {d['method']}){flag}")
    print(f"\n  TOTAL: {total_beats}/{total_metrics} metrics beaten across {len(summary)} datasets")

    # Generate and save report
    report = generate_markdown_report(results, summary)
    report_path = os.path.join(RESULTS_DIR, 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save structured JSON
    json_path = os.path.join(RESULTS_DIR, 'all_results_summary.json')
    json_data = {}
    for dataset in DATASET_ORDER:
        if dataset not in results:
            continue
        json_data[dataset] = {
            'folds': results[dataset]['folds'],
            'metrics': {m: {'mean': v[0], 'std': v[1]}
                        for m, v in results[dataset]['metrics'].items()},
        }
        if dataset in summary:
            json_data[dataset]['beats'] = summary[dataset]['beats']
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON summary saved to {json_path}")


if __name__ == '__main__':
    main()
