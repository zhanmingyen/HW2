# Project Context

# Project Context

Project name: Spam Classification
Short description: Baseline project to classify SMS/email messages as spam or ham using classical ML models.

Repository root:
  c:\Users\EE715 - Acer\Desktop\HW3

Primary language(s):
  - Python

Tech stack:
  - Runtime / framework: Python 3.9+
  - Package manager: pip
  - Build system: none (simple scripts/notebooks)
  - Testing: pytest
  - CI: GitHub Actions (recommended)
  - Containerization: Optional, Docker for reproducible environments

Conventions
  - Branching: main / feature/*
  - Commit messages: Conventional Commits recommended (type:scope: subject)
  - PR template: use GitHub PR template (add in .github/PULL_REQUEST_TEMPLATE.md)
  - Code style: black + isort for Python formatting
  - Tests: unit + small integration smoke tests; aim for meaningful coverage on preprocessing and pipeline functions

Repository layout
  - /scripts - runnable scripts (data ingestion, training)
  - /notebooks - exploratory notebooks and reproducible experiments
  - /tests - test suites
  - /openspec - spec and change proposals
  - /docs - documentation and reproduction instructions

Openspec usage
  - Change proposal folder: openspec/changes/proposals/
  - How to create/apply: create a markdown file under the proposals folder and open a PR; follow the OpenSpec format in `openspec/AGENTS.md`
  - Validation command: `openspec validate ./openspec` (install via `npm i -g @fission-ai/openspec` if using the global tool)

Owners and contacts
  - Primary owner: <owner-handle>
  - Reviewers: <list>

Local dev
  - Python version: 3.9 or 3.10 recommended
  - Useful commands:
    - Create virtualenv: python -m venv .venv
    - Activate (PowerShell): .\.venv\Scripts\Activate.ps1
    - Install deps: python -m pip install -r requirements.txt
    - Run tests: pytest -q
    - Lint/format: black . && isort .

Recommended ML dependencies (example `requirements.txt`):
  - pandas==1.5.3
  - numpy==1.24.2
  - scikit-learn==1.2.2
  - joblib==1.2.0
  - nltk==3.8.1  # optional, if used for stopwords/tokenization

Release & deployment
  - Release process: manual tagging and uploading artifacts (if packaging models). Automate later in CI if needed.
  - Artifact registry: model artifacts stored under /artifacts or uploaded to an internal storage when deploying

Notes / TODO
  - Fill in `Primary owner` and `Reviewers` contact info.
  - Add `requirements.txt` and a minimal GitHub Actions workflow for CI smoke tests.

Owner: <owner-handle>
Created: 2025-10-27

Summary
  Add a spam classification feature that trains and evaluates a baseline machine-learning model on the provided SMS/spam dataset. The baseline model will be a logistic regression (optionally evaluate an SVM as a comparison). This change adds data ingestion, preprocessing, model training, evaluation, and a reproducible experiment (notebook or script).

Motivation
  - Provide an initial ML capability to classify spam vs. ham.
  - Establish a reproducible pipeline and evaluation baseline to iterate on.
  - Allow future phases to improve model, features, and deployment.

Dataset
  - Source: https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
  - Expected format: CSV, two columns (label, message) — verify and document exact schema on ingestion.

Design
  - Phase1 — Baseline
    - Ingest CSV from URL and local copy support.
    - Preprocess text: lowercasing, punctuation/tokenization, stopword removal (configurable).
    - Feature extraction: TF-IDF vectorization (configurable n-grams).
    - Baseline model: Logistic Regression (sklearn). Optionally train an SVM to compare.
    - Train / validation split (e.g., stratified 80/20) and cross-validation for metrics.
    - Evaluation: accuracy, precision, recall, F1, ROC AUC, confusion matrix.
    - Output: trained model artifact (pickle), evaluation report (JSON/Markdown), and a simple CLI or Jupyter notebook demonstrating inference.
  - Phase2, Phase3...
    - (Placeholders) — leave contents empty for now to be filled later.

Implementation notes
  - Language: Python 3.9+ (or use project's primary language if different).
  - Libraries: scikit-learn, pandas, numpy, joblib (or pickle), optionally nltk/spacy for preprocessing.
  - Scripts/notebooks:
    - scripts/data_ingest.py
    - scripts/train_baseline.py
    - notebooks/experiment_baseline.ipynb
  - Tests: unit tests for preprocessing and small integration test for end-to-end training on a tiny sample.

Tasks
  - Create data ingestion and schema validation code.
  - Implement preprocessing and TF-IDF pipeline.
  - Implement logistic regression training script and model save/load.
  - Add evaluation and reporting.
  - Add a Jupyter notebook demonstrating the workflow.
  - Add unit tests for preprocessing and a smoke test for training.
  - Add CI entry to run smoke test (optional).
  - Document dataset provenance and reproducibility steps in README or docs.

Acceptance criteria
  - Script or notebook can download the dataset and produce a trained model artifact.
  - Evaluation report produced with accuracy, precision, recall, F1 and ROC AUC.
  - Basic unit tests pass in CI for preprocessing and a smoke training run.
  - README or docs include instructions to reproduce the baseline locally.

Risks & considerations
  - Data licensing/usage — confirm dataset allowed for this use.
  - Class imbalance — require stratified splitting and metric selection.
  - Privacy — dataset is public SMS dataset; ensure no PII handling beyond dataset content.
  - Dependencies — pin library versions for reproducibility.

Rollout plan
  - Implement Phase1 baseline locally and verify metrics.
  - Add CI smoke test to ensure baseline pipeline runs.
  - Iterate in Phase2+ to improve features, models, and consider deployment.

Notes
  - User originally mentioned both logistic regression and SVM; baseline will use logistic regression unless you confirm SVM is preferred for Phase1.
  - Update project openspec/project.md to list ML dependencies, Python version, and commands to run the baseline.