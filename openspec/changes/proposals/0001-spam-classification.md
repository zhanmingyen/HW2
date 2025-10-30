# Change 0001 — Spam classification (Phase 1: SVM baseline)

Status: Draft
Owner: <owner-handle>
Created: 2025-10-27

Summary
  Phase 1: build a baseline spam classifier (SVM) using the provided SMS/email dataset. This proposal focuses on data ingestion, preprocessing, feature extraction, model training (SVM), evaluation, and producing a reproducible experiment artifact (script or notebook).

Motivation
  - Provide an initial ML capability to classify spam vs. ham (SMS/email).
  - Establish a reproducible pipeline and evaluation baseline to iterate on and improve.

Data
  - Source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
  - Expected schema: CSV with two columns: label (spam/ham) and message (text). Implementation will verify and document the exact schema on ingestion.

Design (Phase 1)
  - Ingest CSV from URL (and support a local-copy fallback).
  - Preprocess text: lowercasing, punctuation removal, tokenization, optional stopword removal (configurable).
  - Feature extraction: TF-IDF vectorization (configurable n-grams, min_df/max_df).
  - Baseline model: Support training a Support Vector Machine (SVM) classifier (scikit-learn). Optionally include a Logistic Regression model for comparison in a follow-up task.
  - Train / validation split: stratified 80/20, with cross-validation for stable metrics.
  - Evaluation: accuracy, precision, recall, F1, ROC AUC, and confusion matrix.
  - Outputs: trained model artifact (joblib/pickle), evaluation report (JSON and Markdown), and a simple CLI or Jupyter notebook demonstrating inference.

Implementation notes
  - Language: Python 3.9+ (or align with repo primary language if different).
  - Libraries: scikit-learn, pandas, numpy, joblib (or pickle), optionally nltk or spaCy for richer preprocessing.
  - Recommended pinned dependency versions: include in a requirements.txt or pyproject.toml for reproducibility.
  - Scripts / notebooks:
    - scripts/data_ingest.py — download and validate the dataset schema.
    - scripts/preprocess.py — text preprocessing utilities.
    - scripts/train_svm_baseline.py — training/evaluation pipeline for SVM and model export.
    - notebooks/experiment_svm_baseline.ipynb — interactive exploration and reproducible run-through.
  - Tests: unit tests for preprocessing functions and a small smoke integration test that runs training on a tiny sample to ensure pipeline health.

Tasks
  - Create data ingestion and schema validation code.
  - Implement preprocessing and TF-IDF pipeline.
  - Implement SVM training script and model save/load.
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
  - Implement Phase 1 (SVM baseline) locally and verify metrics.
  - Add lightweight CI smoke test to ensure the baseline pipeline runs in CI (fast, small sample).
  - Iterate in Phase 2+ to improve features, model selection, and consider model packaging/deployment.

Notes
  - This proposal implements SVM as the Phase 1 baseline per your instruction. If you later want Logistic Regression as the baseline instead, or both trained and compared, we can update or add a follow-up proposal.
  - Please provide the owner handle and preferred Python/runtime versions so I can update metadata and `openspec/project.md`.
  - CI: a GitHub Actions workflow (`.github/workflows/smoke-training.yml`) runs the baseline training and uploads artifacts:
    - `models/svm_baseline.joblib` — model bundle
    - `models/svm_baseline_evaluation.json` — evaluation metrics (JSON)
    - `models/svm_baseline_evaluation.md` — human-readable evaluation report
# Change 0001 — Spam classification (Phase 1: SVM baseline)

Status: Draft
Owner: <owner-handle>
Created: 2025-10-27

Summary
  Phase 1: build a baseline spam classifier (SVM) using the provided SMS/email dataset. This proposal focuses on data ingestion, preprocessing, feature extraction, model training (SVM), evaluation, and producing a reproducible experiment artifact (script or notebook).

Motivation
  - Provide an initial ML capability to classify spam vs. ham (SMS/email).
  - Establish a reproducible pipeline and evaluation baseline to iterate on and improve.

Data
  - Source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
  - Expected schema: CSV with two columns: label (spam/ham) and message (text). Implementation will verify and document the exact schema on ingestion.

Design (Phase 1)
  - Ingest CSV from URL (and support a local-copy fallback).
  - Preprocess text: lowercasing, punctuation removal, tokenization, optional stopword removal (configurable).
  - Feature extraction: TF-IDF vectorization (configurable n-grams, min_df/max_df).
  - Baseline model: Support training a Support Vector Machine (SVM) classifier (scikit-learn). Optionally include a Logistic Regression model for comparison in a follow-up task.
  - Train / validation split: stratified 80/20, with cross-validation for stable metrics.
  - Evaluation: accuracy, precision, recall, F1, ROC AUC, and confusion matrix.
  - Outputs: trained model artifact (joblib/pickle), evaluation report (JSON and Markdown), and a simple CLI or Jupyter notebook demonstrating inference.

Implementation notes
  - Language: Python 3.9+ (or align with repo primary language if different).
  - Libraries: scikit-learn, pandas, numpy, joblib (or pickle), optionally nltk or spaCy for richer preprocessing.
  - Recommended pinned dependency versions: include in a requirements.txt or pyproject.toml for reproducibility.
  - Scripts / notebooks:
    - scripts/data_ingest.py — download and validate the dataset schema.
    - scripts/preprocess.py — text preprocessing utilities.
    - scripts/train_svm_baseline.py — training/evaluation pipeline for SVM and model export.
    - notebooks/experiment_svm_baseline.ipynb — interactive exploration and reproducible run-through.
  - Tests: unit tests for preprocessing functions and a small smoke integration test that runs training on a tiny sample to ensure pipeline health.

Tasks
  - Create data ingestion and schema validation code.
  - Implement preprocessing and TF-IDF pipeline.
  - Implement SVM training script and model save/load.
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
  - Implement Phase 1 (SVM baseline) locally and verify metrics.
  - Add lightweight CI smoke test to ensure the baseline pipeline runs in CI (fast, small sample).
  - Iterate in Phase 2+ to improve features, model selection, and consider model packaging/deployment.

Notes
  - This proposal implements SVM as the Phase 1 baseline per your instruction. If you later want Logistic Regression as the baseline instead, or both trained and compared, we can update or add a follow-up proposal.
  - Please provide the owner handle and preferred Python/runtime versions so I can update metadata and `openspec/project.md`.
