# Cost-Aware Evaluation of Ensemble Intrusion Detection on Drone Network Traffic

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![Notebook](https://img.shields.io/badge/Jupyter-Colab%20ready-F9AB00)
![License](https://img.shields.io/badge/License-add%20licence-lightgrey)

Official code, configuration and result files for the paper **"A Cost-Aware Evaluation of Ensemble Intrusion Detection on Drone Network Traffic"** (submitted to *Discover Informatics*).

The study compares **17 intrusion detectors** on the ISOT Drone IDS benchmark under one protocol. It reports accuracy, macro-F1, Matthews correlation, probability calibration (Brier score, ECE), single-row inference latency and training time. It also tests whether accuracy differences are real, and where they break.

## Table of contents

1. [Why this repository](#why-this-repository)
2. [Main findings](#main-findings)
3. [Detectors](#detectors)
4. [Evaluation protocol](#evaluation-protocol)
5. [Repository layout](#repository-layout)
6. [Requirements](#requirements)
7. [Data](#data)
8. [Quick start](#quick-start)
9. [Pipeline stages and run time](#pipeline-stages-and-run-time)
10. [From result files to paper tables and figures](#from-result-files-to-paper-tables-and-figures)
11. [Reproducibility notes](#reproducibility-notes)
12. [Known limitations](#known-limitations)
13. [Citation](#citation)
14. [License](#license)
15. [Contact](#contact)

## Why this repository

Ensemble detectors on drone traffic often report accuracy near 99.99 percent. When accuracy saturates, it no longer separates detectors. This repository gives a protocol that does:

* a leakage-aware evaluation with a duplicate audit, a grouped split, an identifier ablation and time-ordered splits,
* one single-row latency harness for every detector,
* calibration before and after post hoc recalibration,
* corrected significance tests, equivalence tests and repeated validation,
* a reweighting of the test set to the raw class proportions.

Everything in the paper can be regenerated from one notebook.

## Main findings

All numbers are from the paper and come from the result files in `results/`.

| Topic | Result |
|---|---|
| Data | 301,553 flows in 10 classes after removing 2 duplicates. 60 model features. The timestamp and the sequence number are never used as features. |
| Accuracy tie | On the grouped split, 16 of 17 detectors differ by 0.00012 in accuracy and 0.00040 in macro-F1. Across all 17 the spans are 0.00053 and 0.00166. |
| Significance | No pair among the 16 detectors differs after Holm or Benjamini-Hochberg correction (120 pairs). All 120 are equivalent within a margin of 0.0002. |
| Identifier shortcut | A decision tree that sees only the timestamp scores 1.0000 on the grouped split. Removing the identifiers changes accuracy by at most 0.00010. |
| Where the tie breaks | On the last time block of a time-ordered split, accuracy is 0.8196 for the decision tree, the random forest and the cascade, and 0.9697 for HGB. |
| Latency | Median single-row latency runs from 0.140 ms (decision tree) to 74.99 ms (DES), a factor of 536. |
| Training cost | Training time runs from 5.8 s to 1190 s, a factor of 206. |
| Calibration | AdaBoost has a Brier score of 0.89 at 0.99982 accuracy, because its output is close to uniform. After temperature scaling or isotonic regression, the ECE of every tested detector is below 0.003. |
| Cascade | The confidence cascade answers 99.99 percent of test flows at its first stage. Its median latency is 0.212 ms. It is not faster than the decision tree alone. Its first stage is over-confident under time-ordered evaluation. |
| Pareto front | Macro-F1 against median latency: decision tree, confidence cascade, bagging, feature-view ensemble. Only the cheap end of the front is stable under bootstrap. |

The tie on accuracy is specific to this benchmark and this split.

## Detectors

| Family | Detectors |
|---|---|
| Single learners | Decision tree (DT), random forest (RF), histogram gradient boosting (HGB) |
| Averaging | Hard voting, soft voting |
| Bagging and boosting | Bagging (DT x 100), Balanced Bagging, AdaBoost (SAMME) |
| Meta-learning | Stacking (LR meta-model), blending (LR meta-model) |
| Structured ensembles | Feature-view ensemble, hierarchical IDS, mixture of experts, specialist ensemble |
| Dynamic selection | DCS, DES |
| Cost-aware | Confidence cascade (DT, HGB, RF) with `tau_c = 0.995` |

The base pool for voting, stacking, blending, DCS and DES is GNB, DT, RF, ET and HGB. Full settings are in the notebook and in Table VIII of the paper.

## Evaluation protocol

| Component | Setting |
|---|---|
| Splits | Grouped 80/20 split (seed 42). Five repeated grouped splits (seeds 1 to 5). Time-ordered split per class (last 20 percent as test). Five blocked time folds. |
| Grouping | Feature vectors rounded to 3 decimals and hashed into pseudo-flow groups (283,402 groups). Each group stays on one side of a split. |
| Scaling | Standard scaler fitted on training rows only, in every split. |
| Latency | One thread. 20 untimed warm-up calls, then 520 timed single-row calls per detector. Median, 95th percentile and mean are reported. |
| Calibration | Temperature scaling and one-vs-rest isotonic regression, fitted on a 20 percent hold-out of the training rows. 15-bin ECE. 1,000 bootstrap resamples for intervals. |
| Statistics | Exact McNemar test with Holm and Benjamini-Hochberg correction. Paired bootstrap intervals. Equivalence margins 0.0002 and 0.0005. Wilson intervals. |
| Class priors | Test flows reweighted to the raw class proportions. Confidence thresholds and false-alarm counts reported. |
| Seed | 42 everywhere unless stated. |

## Repository layout

Recommended layout for this repository:

```text
.
|-- README.md
|-- Final_Drone_IDS_Experiments.ipynb   # full pipeline, runs top to bottom
|-- requirements.txt                    # see the Requirements section
|-- results/                            # result files written by the notebook
|   |-- prep.json
|   |-- main.json
|   |-- main_preds.npz
|   |-- stats.json
|   |-- ablation.json
|   |-- calib.json
|   |-- calib_probas.npz
|   |-- prior.json
|   `-- repeat.json
`-- figures/                            # paper figures (PNG and PDF)
    |-- fig_pareto.*
    |-- fig_acc_vs_brier.*
    |-- fig_reliability.*
    `-- fig_calibration_after.*
```

The dataset is not included. See [Data](#data).

## Requirements

* Python 3.10 or newer
* A CPU runtime is enough. No GPU is used.

Results in the paper were produced with:

| Package | Version |
|---|---|
| Python | 3.13 |
| scikit-learn | 1.6.1 |
| NumPy | 2.4.6 |
| pandas | 2.2.3 |
| SciPy | 1.16.3 |
| imbalanced-learn | 0.14.2 |

Hardware: a cloud virtual machine with 2 virtual CPU cores (x86-64, Linux 6.6, no GPU). The exact processor model was not recorded.

Suggested `requirements.txt`:

```text
numpy
pandas
scipy
scikit-learn
imbalanced-learn
threadpoolctl
matplotlib
jupyter
```

Install with:

```bash
pip install -r requirements.txt
```

## Data

The notebook expects a processed CSV of the **ISOT Drone Anomaly Detection Dataset** (University of Victoria) with:

* 62 numeric feature columns, including `ts` and `Sequence number`,
* an integer column named `Label` with classes 0 to 9.

| Label | Class |
|---|---|
| 0 | Benign |
| 1 | Denial of service |
| 2 | Injection |
| 3 | IP spoofing |
| 4 | Man-in-the-middle |
| 5 | Password cracking |
| 6 | Payload manipulation |
| 7 | Replay attack |
| 8 | Unauthorized UDP |
| 9 | Video interception |

Please download the original data from the dataset authors and follow their licence. Do not redistribute the data unless the licence allows it.

The majority classes were downsampled once before this study. The sampling seed was not recorded. The notebook reads the working set as it is. The raw class counts used for the reweighting analysis are stored in the notebook.

Dataset reference: Chen, Z., Traore, I., Mamun, M., Saad, S. *Drone Anomaly Detection: Dataset and Unsupervised Machine Learning.* Foundations and Practice of Security, LNCS 15532, Springer, 2025. doi: [10.1007/978-3-031-87499-4_12](https://doi.org/10.1007/978-3-031-87499-4_12)

## Quick start

### Option A: Google Colab

1. Upload the CSV to Google Drive.
2. Open `Final_Drone_IDS_Experiments.ipynb` in Colab.
3. Edit `CSV_PATH` and `OUT_DIR` in the settings cell.
4. Run the cells from top to bottom. Use one runtime for the whole run, because latency depends on the machine.

### Option B: local Jupyter

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt

export DRONE_IDS_CSV="/path/to/updated_drone_ids_dataset.csv"
export DRONE_IDS_OUT="./results"
jupyter lab Final_Drone_IDS_Experiments.ipynb
```

### Smoke test

To check that everything runs in a few minutes, set `SMOKE = True` in the settings cell and point `CSV_PATH` to a small CSV with the same columns. The smoke settings use small models and few repetitions. Their numbers are not meaningful.

## Pipeline stages and run time

Each stage writes its own result file and is skipped on a re-run unless `FORCE = True`. A disconnect therefore loses little work.

| Run cell | Stages | Result files | Approx. time (2 vCPU) |
|---|---|---|---|
| RUN 1 | `prep`, `ablation` | `prep.json`, `ablation.json` | 25 min |
| RUN 2 | `main`, `stats` | `main.json`, `main_preds.npz`, `stats.json` | 90 to 100 min |
| RUN 3 | `calib`, `prior` | `calib.json`, `calib_probas.npz`, `prior.json` | 20 min |
| RUN 4 | `repeat` | `repeat.json` | 25 min |
| RUN 5 | figures and archive | `figs/*.png`, `figs/*.pdf`, `revision_out.zip` | 1 min |

Total run time is about three hours on two virtual cores.

| Stage | What it does |
|---|---|
| `prep` | Duplicate removal, pseudo-flow groups, grouped split, time-ordered split checks, majority baselines. |
| `ablation` | Identifier ablation on 62, 60 and 58 features, on the grouped and the time-ordered split, plus single-field identity probes. |
| `main` | Trains the 17 detectors with train-only scaling and times each one in the single-row harness. |
| `stats` | Exact McNemar tests, Holm and BH correction, paired intervals, equivalence tests, per-class recall, Pareto front and its bootstrap stability. |
| `calib` | Post hoc calibration (temperature, isotonic), bin counts, and the cascade threshold sweep. |
| `prior` | Reweighting to the raw class proportions, confidence thresholds, false-alarm counts. |
| `repeat` | Five grouped seeds and five blocked time folds for DT, RF, HGB and the cascade. |

## From result files to paper tables and figures

| Paper item | Source file |
|---|---|
| Table V (class distribution), Table VII (leakage audit) | `prep.json` |
| Table X (unified benchmark) | `main.json` |
| Table XI (repeated grouped splits), Table XVI (blocked time folds) | `repeat.json` |
| Table XII (pairwise tests), Table XIII (per-class recall) | `stats.json` |
| Table XIV (identifier ablation), Table XV (identity probes) | `ablation.json` |
| Table XVII (post hoc calibration), Table XX (threshold sweep), Table XXI (bin counts) | `calib.json` |
| Table XVIII (raw class proportions), Table XIX (confidence thresholds) | `prior.json` |
| Fig. 4 (accuracy vs Brier), Fig. 5 (Pareto), Fig. 6 (reliability), Fig. 7 (after recalibration) | `main.json`, `main_preds.npz`, `calib.json`, `calib_probas.npz`; drawn by `stage_figs` |

The last cell of the notebook prints the benchmark table (accuracy, macro-F1, Brier, ECE, training time, median and 95th percentile latency) and the Pareto set.

## Reproducibility notes

* **Hardware dependence.** Latency in milliseconds depends on the machine. Compare detectors within one run. Do not compare absolute values across machines.
* **Library versions.** The notebook prints the row counts after loading. The paper values are 301,553 rows after de-duplication, 241,324 training flows and 60,229 test flows. A mismatch means a different CSV or a different library version.
* **Randomness.** The seed is 42. The repeated splits use seeds 1 to 5. Latency uses 520 randomly chosen test flows.
* **Timing fairness.** Every model is limited to one thread and called once per flow. Batch throughput is stored in `main.json` but is not used for ranking.
* **No test-set tuning.** Hyperparameters are fixed values. The calibrators are fitted on a hold-out of the training rows. The test set is used only for scoring.

## Known limitations

* The results describe one benchmark. They are not a claim about every intrusion-detection dataset.
* Each class comes from a single capture session, and session identifiers are not in the processed data. The grouped split is a proxy built from hashed feature vectors.
* The majority classes were downsampled once, with an unrecorded seed. The class-prior analysis assumes the draw was random within each class.
* Latency was measured on a cloud virtual machine, not on embedded drone hardware. Energy and memory were not measured.
* Repeated validation covers four detectors, not all 17.
* No adversarial perturbation and no unseen attack category is tested.

## Citation

If you use this code, please cite the paper. Replace the entry below after publication.

```bibtex
@article{hossain2026costaware,
  author  = {Hossain, Md. Alamgir},
  title   = {A Cost-Aware Evaluation of Ensemble Intrusion Detection on Drone Network Traffic},
  journal = {Discover Informatics},
  year    = {2026},
  note    = {Under review. Add volume, pages and DOI after publication.}
}
```

The related earlier study on the same dataset:

```bibtex
@article{hossain2026ensemblexai,
  author  = {Hossain, M. A. and Ishtiaq, W. and Islam, M. S.},
  title   = {A Comparative Analysis of Ensemble-Based Machine Learning Approaches With Explainable AI for Multi-Class Intrusion Detection in Drone Networks},
  journal = {Security and Privacy},
  year    = {2026},
  volume  = {9},
  number  = {1},
  doi     = {10.1002/spy2.70164}
}
```

## License

Add the licence of your choice here, for example MIT or Apache 2.0, and place the licence text in a `LICENSE` file. The dataset has its own licence from its authors.

## Contact

Md. Alamgir Hossain. Add your institution and e-mail address here. Please open a GitHub issue for questions about the code or the results.
