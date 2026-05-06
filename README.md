# System Call-Based Intrusion Detection with Sequence Modeling on ADFA-LD

[![CI](https://github.com/Nada-belarbi/test/actions/workflows/ci.yml/badge.svg)](https://github.com/Nada-belarbi/test/actions/workflows/ci.yml)

This repository implements an end-to-end intrusion detection pipeline on ADFA-LD using:

- Classical baselines: Logistic Regression and Random Forest on BoW/TF-IDF/bigrams.
- Sequential model: LSTM with pre/post truncation ablation and hyperparameter checks.
- Error analysis: false positives/false negatives, scenario/family breakdown, length effects, imbalance effects.
- Robustness checks: local syscall deletion and light substitution noise.

## Project Structure

- `configs/default.yaml`: frozen default experiment configuration.
- `data/raw/adfa-ld/ADFA-LD`: raw dataset.
- `data/metadata`: audit/split artifacts.
- `data/processed/protocol_a_main_ml1231_dualview_v1`: encoded datasets/features.
- `src/`: reusable pipeline modules.
- `notebooks/01_dataset_audit.ipynb`: dataset audit and protocol construction.
- `notebooks/02_preprocessing_and_encoding.ipynb`: preprocessing and feature encoding.
- `notebooks/03_baseline_models.ipynb`: baseline training/evaluation.
- `notebooks/04_lstm_model.ipynb`: LSTM ablation and best-model export.
- `notebooks/05_error_analysis.ipynb`: error analysis + robustness.
- `notebooks/06_attack_variation_experiments.ipynb`: cross-model comparison — recall by attack family/scenario, error by sequence length, LSTM robustness under perturbations, and final summary table.
- `results/tables`: CSV result tables.
- `results/confusion_matrices`: confusion matrix figures.
- `reports/figures`: exported plots.
- `reports/technical_report.pdf`: report artifact.

## Implemented Core Modules

- `src/datasets.py`
	- `SyscallSequenceDataset`
	- Dynamic-padding `collate_padded_batch`
	- Batch outputs include `input_ids`, `lengths`, `labels`, `sample_ids`
- `src/baseline_models.py`
	- Baseline suite runner for LR/RF over BoW/TF-IDF/bigrams
- `src/lstm_model.py`
	- `LSTMClassifier` binary sequence model
- `src/metrics.py`
	- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, FNR
	- Confusion matrix utilities
- `src/train.py`
	- LSTM training loop, validation, checkpointing, early stopping
- `src/evaluate.py`
	- Unified sklearn/LSTM evaluation and prediction frame builders

## Environment Setup

From repository root (`syscall-anomaly-detection`):

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn pyarrow pyyaml torch tqdm jupyter
```

Linux / macOS (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn pyarrow pyyaml torch tqdm jupyter
```

## Execution Order

1. `01_dataset_audit.ipynb`
2. `02_preprocessing_and_encoding.ipynb`
3. `03_baseline_models.ipynb`
4. `04_lstm_model.ipynb`
5. `05_error_analysis.ipynb`
6. `06_attack_variation_experiments.ipynb`

If notebook execution is unstable in UI, run the LSTM batch script:

```powershell
python scripts/run_lstm_ablation.py
```

## Main Outputs

Expected key tables in `results/tables`:

- `baseline_results_all_splits.csv`
- `baseline_results_test_ranked.csv`
- `predictions_baselines_test.csv`
- `lstm_ablation_results.csv`
- `lstm_best_run_summary.csv`
- `predictions_lstm_test.csv`
- `final_baseline_vs_lstm.csv`
- `error_analysis_false_positives.csv`
- `error_analysis_false_negatives.csv`
- `error_analysis_by_attack_scenario.csv`
- `error_analysis_by_attack_family.csv`
- `error_analysis_by_length_bin.csv`
- `error_analysis_imbalance_impact.csv`
- `robustness_stability_lstm.csv`
- `robustness_prediction_stability.csv`

Expected figures:

- `reports/figures/baseline_f1_comparison.png`
- `reports/figures/lstm_ablation_f1_top8.png`
- `reports/figures/error_analysis_and_robustness_summary.png`
- `reports/figures/attack_variation_recall_by_family.png`
- `reports/figures/attack_variation_error_by_scenario.png`
- `reports/figures/attack_variation_error_by_length.png`
- `reports/figures/attack_variation_robustness.png`

## Dataset Licence

This project uses the **ADFA-LD** dataset (Australian Defence Force Academy — Linux Dataset).
ADFA-LD was created by Creech & Hu (2013) at UNSW Canberra and is freely available for
non-commercial academic research.  Please cite the original paper when using this dataset:

> G. Creech and J. Hu, "Generation of a New IDS Test Dataset: Time to Retire the KDD
> Collection," *IEEE WCNC*, 2013.

## Notes on Protocol and Leakage Safety

- Protocol A is enforced with `sha1` disjointness checks across train/val/test.
- Both `trace_id` and `sha1` overlap checks are exported in `data/metadata`.
- Current processed version uses `max_len = p95 = 1231` and dual pre/post views.
