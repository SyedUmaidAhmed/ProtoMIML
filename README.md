# ProtoMIML

Official implementation of **"Semi-Parametric Multi-Instance Multi-Label Learning via
Prototypical Memory, Cross-Attention Label Pooling, and Sharpness-Aware Minimization"**
(Syed Umaid Ahmed, Muhammad Atif Tahir, Muhammad Waqas — under review).

ProtoMIML is a compact (130K–177K parameter) semi-parametric pipeline for
multi-instance multi-label (MIML) learning on small datasets (144–2,000 samples),
where classical non-parametric methods still hold the published state of the art
and unregularized neural models overfit.

## Method

```
input bag (B, N, D)
  → InstanceEncoder        residual MLP, applied per instance
  → ISAB                   induced set attention: instances exchange context in O(NM)
  → Cross-attention label pooling   one learnable query per label → label-specific z_l
  → LabelGCN               2-layer GCN over the label co-occurrence graph
  → Dual head              parametric per-label MLP  +  non-parametric EMA prototype bank
  → Gated fusion           per-label convex combination → prediction
```

Training: two-phase curriculum with SAM(AdamW), SWA, gradual fused-logit ramp,
multi-view consistency, 5-seed ensembling, and per-label threshold optimization.

## Results

10-fold cross-validation on five standard MIML benchmarks
(Scene, Reuters, Letter Carroll, Letter Frost, MSCV2):

| Comparison | Outcome |
|---|---|
| vs. best **published** baselines (MIML-kNN/SVM family, MIMLfast, RankLossSIM, KISAR) | improves or ties **22 / 25** metrics; clean sweeps on 4 datasets |
| vs. **ABMIL** (shared attention), identical folds | **25 / 25** |
| vs. **ABMIL** (per-label attention), identical folds | **23 / 25** |

The paper also reports two explicit negative results, both reproducible from this
repository: under a matched ablation protocol no individual module's contribution
is resolvable above fold-level noise, and the learned per-label fusion gate is
empirically inert (every gate stays within ~5 points of its 0.5 initialization).

## Setup

```bash
pip install -r requirements.txt   # torch>=2.0, numpy, scipy, scikit-learn
```

Datasets are the public MIML benchmarks: Scene and Reuters from Zhou et al.
(LAMDA, Nanjing University — http://www.lamda.nju.edu.cn/data_MIMLimage.ashx);
Letter Carroll, Letter Frost, and MSCV2 from Briggs et al. (KDD 2012).
Place the `.mat` files locally and edit the paths in `run_experiment.py` (`CONFIGS`).
Runs use MPS (Apple Silicon), CUDA, or CPU, auto-detected.

## Reproducing the paper

```bash
# Headline results: 10-fold CV, 5-seed ensemble (Table 4)
python run_experiment.py scene            # also: reuters, letter_carroll, letter_frost, MSCV2

# Split-matched neural baselines: ABMIL shared + per-label (Sec. 6.7)
python run_baselines.py --model all --dataset all --folds 10

# Matched-protocol ablations: matched control, fixed gate, no-SAM, module removals (Sec. 7)
python run_ablations.py --variant all

# Gate-value measurement (Table 8) and measured attention maps (Figs. 1 and 9)
python extract_gates.py scene reuters letter_carroll letter_frost MSCV2
python extract_attention.py

# Regenerate the paper's result tables and figures directly from the JSON artifacts
python compile_new_results.py && python gen_ablation_chart.py && python gen_attention_figure.py \
  && python gen_results_chart.py && python gen_gate_strip.py
```

## Archived artifacts

`results/` ships the exact per-fold JSON outputs behind every number in the paper —
main results, both ABMIL baselines, all ablation variants, Wilcoxon recomputations,
gate values, attention weights, and pathway-selection counts. The paper's result
tables and figures are generated programmatically from these files, not transcribed,
so every reported quantity can be traced to an artifact in this repository.

## Repository layout

| Path | Contents |
|---|---|
| `models/` | architecture components, incl. the `abmil.py` baselines |
| `training/` | SAM optimizer, two-phase trainer, CV harness |
| `metrics/` | the five standard MIML metrics + threshold optimization |
| `data/` | loaders for the standard and DD-format `.mat` benchmarks |
| `results/` | archived JSON artifacts for every reported experiment |

## Citation

The manuscript is under journal review; a citation entry will be added upon
publication.

## License

MIT — see [LICENSE](LICENSE).
