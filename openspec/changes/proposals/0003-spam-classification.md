# Change 0003 — (Archival) Logistic Regression draft (superseded)

Status: Superseded
Owner: <owner-handle>
Created: 2025-10-27

Summary
  This file contained an earlier draft proposing a Logistic Regression baseline. It has been superseded by `0001-spam-classification.md` (SVM baseline) per project direction. Keep this file as an archival note.

If you later decide to make Logistic Regression the canonical baseline, update `0001` or promote this file back to active status.

Notes
  - Active proposals: `openspec/changes/proposals/0001-spam-classification.md` (SVM baseline), `0002-spam-followups.md` (placeholders).
# Change 0003 — Spam email/SMS classification (baseline)

Status: Draft
Owner: <owner-handle>
Created: 2025-10-27

Summary
  Add a baseline spam classification feature that trains and evaluates a basic machine-learning model on the provided SMS/email spam dataset. The baseline model will use Logistic Regression (optionally evaluate an SVM for comparison in later iterations).

Motivation
  - Provide an initial ML capability to classify spam vs. ham (SMS/email).
  - Establish a reproducible pipeline and evaluation baseline for future improvements and deployment.

Data
  - Source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
  - Expected schema: CSV with two columns: label (spam/ham) and message (text). Implementation will verify and document the exact schema on ingestion.

Design
  - Phase 1 — Baseline
    - Ingest CSV from URL (and support a local-copy fallback).
    - Preprocess text: lowercasing, punctuation removal, tokenization, optional stopword removal (configurable).
    - Feature extraction: TF-IDF vectorization (configurable n-grams, min_df/max_df).
    - Baseline model: Logistic Regression (scikit-learn). Optionally train an SVM and compare metrics as a secondary experiment.
    - Train / validation split: stratified 80/20, with cross-validation for stable metrics.
    - Evaluation: accuracy, precision, recall, F1, ROC AUC, and confusion matrix.
    - Outputs: trained model artifact (joblib/pickle), evaluation report (JSON and Markdown), and a simple CLI or Jupyter notebook demonstrating inference.
  - Phase 2, Phase 3
    - (Placeholders — leave contents empty for now.)

Implementation notes
  - Language: Python 3.9+ (or align with repo primary language if different).
  - Libraries: scikit-learn, pandas, numpy, joblib (or pickle), optionally nltk or spaCy for richer preprocessing.
  - Recommended pinned dependency versions: include in a requirements.txt or pyproject.toml for reproducibility.
  - Scripts / notebooks:
    - scripts/data_ingest.py — download and validate the dataset schema.
    - scripts/preprocess.py — text preprocessing utilities.
    - scripts/train_baseline.py — training/evaluation pipeline for logistic regression and model export.
    - notebooks/experiment_baseline.ipynb — interactive exploration and reproducible run-through.
  - Tests: unit tests for preprocessing functions and a small smoke integration test that runs training on a tiny sample to ensure pipeline health.

Tasks
  - Create data ingestion and schema validation code.
  - Implement preprocessing and TF-IDF pipeline.
  - Implement logistic regression training script and model save/load.
  - Add evaluation reporting (JSON + human-readable Markdown report).
  - Add a Jupyter notebook demonstrating end-to-end reproducible experiment.
  - Add unit tests for preprocessing and a smoke test for training.
  - (Optional) Add CI smoke test to run a fast end-to-end training on a tiny sample.
  - Document dataset provenance and reproducibility steps in README or docs.

Acceptance criteria
  - Script or notebook can download the dataset and produce a trained model artifact.
  - Evaluation report produced with accuracy, precision, recall, F1 and ROC AUC.
  - Basic unit tests pass for preprocessing and a smoke training run.
  - README or docs include instructions to reproduce the baseline locally.

Risks & considerations
  - Data licensing/usage — confirm dataset license permits this use (public dataset; check provenance).
  - Class imbalance — use stratified splitting and appropriate metrics to mitigate skew.
  - Privacy — dataset is public; avoid adding additional PII handling beyond the dataset contents.
  - Dependency/versioning — pin library versions for reproducibility and add a minimal environment file.

Rollout plan
  - Implement Phase 1 baseline locally and verify metrics.
  - Add lightweight CI smoke test to ensure the baseline pipeline runs in CI (fast, small sample).
  - Iterate in Phase 2+ to improve features, model selection, and consider model packaging/deployment.

Notes
  - The baseline will use Logistic Regression by default (per your request). If you prefer the baseline to be an SVM instead, or want both trained and compared in Phase 1, tell me and I will update the proposal accordingly.
  - Please provide the owner handle and preferred Python/runtime versions so I can update metadata and `openspec/project.md`.
