# Spam classification — baseline

This repository contains a small reproducible baseline for spam classification (SMS/email). It includes data ingestion, a minimal training script for an SVM baseline, and CI to run a smoke training job.

How to run locally (Windows PowerShell)

1. Ensure you have Python 3.9/3.10 installed and available on PATH (or use the `py` launcher).

2. From the repo root:

```powershell
# create venv
py -3 -m venv .venv

# activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# download dataset
python .\scripts\data_ingest.py

# run smoke training
python .\scripts\train_svm_baseline.py --csv data/sms_spam_no_header.csv --out models/svm_baseline.joblib
```

Artifacts produced
- `models/svm_baseline.joblib` — model bundle containing the trained classifier and TF-IDF vectorizer
- `models/svm_baseline_evaluation.json` — machine-readable evaluation report
- `models/svm_baseline_evaluation.md` — human-readable evaluation summary

CI

The repository includes a GitHub Actions workflow `.github/workflows/smoke-training.yml` that runs the data download and training, and uploads the model and evaluation artifacts. You can run it manually from the Actions tab or it runs on pushes to `main` and on PRs.

If you want me to add automatic publishing of the evaluation report into `openspec/changes/` or to comment on PRs with metrics, say so and I'll add it.
