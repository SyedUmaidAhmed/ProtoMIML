# ProtoMIML

Official implementation of **"Label-Query Set Attention for Small-Sample Multi-Instance
Multi-Label Learning, with a Controlled Evaluation of Prototypical Memory"**
(Syed Umaid Ahmed, Muhammad Atif Tahir, Muhammad Waqas — under review, revised version).

ProtoMIML is a compact (131K–177K parameter) model for multi-instance multi-label (MIML)
learning on small datasets (144–2,000 bags). It pools each bag separately for every label
through learnable label queries. As a design hypothesis that the paper tests explicitly, it
also fuses its parametric classifier with a prototypical memory through learned per-label gates.

## Method

```
input bag (B, N, D)
  → InstanceEncoder        residual MLP, applied per instance
  → ISAB                   induced set attention: instances exchange context in O(NM)
  → Cross-attention label pooling   one learnable query per label → label-specific z_l
  → LabelGCN               2-layer GCN over the label co-occurrence graph
  → Dual head              parametric per-label MLP  +  EMA prototype bank
  → Gated fusion           per-label convex combination → prediction
```

Training: two-phase curriculum with SAM(AdamW), SWA, gradual fused-logit ramp,
multi-view consistency, 5-seed ensembling, and per-label threshold optimization.

## Results (revised manuscript)

10-fold cross-validation on Scene, Reuters, Letter Carroll, Letter Frost and MSCV2.
Significance is a paired Wilcoxon test over folds on identical partitions.

| Comparison | Outcome |
|---|---|
| vs. the published record (cross-validated ensemble protocol) | best or tied on **24 / 25** metrics |
| vs. four classical MIML methods re-run on identical folds | best mean on 25 / 25, significantly better on **19 / 25** |
| published 1500/500 protocol, 30 splits (Scene, Reuters) | ahead on all 5 Scene metrics; **no improvement on Reuters** |
| vs. ABMIL (shared / per-label attention) under the same training recipe | significantly better on 4 / 2 datasets; per-label ABMIL ties on 3 |

Negative results, all reproducible from this repository:
- Removing the prototypical memory changes 1 of 25 dataset–metric pairs significantly; removing SAM changes none.
- With every deployed model trained through the fused loss, the prototype pathway alone becomes competitive, but the model with the memory and the model without it differ significantly on **0 / 25** pairs.
- On Yeast (leak-free per-fold neighbourhood bags), every classical method beats every neural model on AP, One-Error and Ranking Loss.

## Setup

```bash
pip install -r requirements.txt   # torch>=2.0, numpy, scipy, scikit-learn
```

Datasets are the public MIML benchmarks: Scene and Reuters from Zhou et al.
(LAMDA, Nanjing University — http://www.lamda.nju.edu.cn/data_MIMLimage.ashx);
Letter Carroll, Letter Frost and MSCV2 from Briggs et al. (KDD 2012); Yeast from
Elisseeff and Weston (NIPS 2001). Place the `.mat` files locally and edit the paths in
`run_experiment.py` (`CONFIGS`).

## Reproducing the revised paper

All revision experiments run on CPU (2 threads per run). Every run archives the per-seed
validation and test scores of the parametric, prototype and fused pathways, so every
deployment policy can be re-scored without retraining.

```bash
# All neural runs listed in revision_jobs.txt, 6 in parallel (resumable)
PYTHON=python ./revision_queue.sh 6
#   single run, e.g.:  python run_revision.py --model protomiml --dataset scene --protocol cvens
#   models: protomiml, protomiml_noproto, protomiml_nosam, protomiml_fullfuse,
#           protomiml_noproto_fullfuse, abmil, abmil_per_label
#   protocols: standard, cvens (cross-validated ensemble), holdout (use --repeats 30)

# Classical baselines on identical folds, and on the 30 holdout splits
python run_classical_baselines.py --dataset all
python run_classical_baselines.py --dataset scene --protocol holdout --repeats 30
python run_classical_baselines.py --dataset reuters --protocol holdout --repeats 30

# Score every run -> results/revision/summary_<tag>.json
python revision_analysis.py

# Paired comparisons, checkpoint audit, timing
python revision_compare.py
python revision_checkpoint_audit.py
python revision_timing.py --threads 2

# Regenerate every result table of the paper from the JSON artifacts
python gen_revision_tables.py && python gen_results_chart.py
```

The original submission's scripts (`run_experiment.py`, `run_baselines.py`,
`run_ablations.py`, `extract_gates.py`, `extract_attention.py`, `compile_new_results.py`)
are kept unchanged for reference.

## Archived artifacts

- `results/revision/`: one directory per revision run, with per-fold `.npz` score
  archives and `run_config.json`; `summary_*.json` for the scored results;
  `classical_*.json`, `comparisons.json`, `checkpoint_audit.json`, `timing.json` and
  `paper_facts.json`, which holds every number quoted in the paper's prose.
- `results/` (top level): the artifacts of the original submission.

The paper's result tables are generated programmatically from these files, not
transcribed, so every reported quantity can be traced to an artifact in this repository.

## Repository layout

| Path | Contents |
|---|---|
| `models/` | architecture components, incl. the `abmil.py` baselines |
| `training/` | SAM optimizer, two-phase trainer (incl. `phase2_select_from`), CV harness |
| `metrics/` | the five standard MIML metrics + threshold optimization |
| `data/` | loaders for the standard and DD-format `.mat` benchmarks; leak-free Yeast bags |
| `results/` | archived artifacts for every reported experiment |

## Citation

The manuscript is under journal review; a citation entry will be added upon
publication.

## License

MIT — see [LICENSE](LICENSE).
