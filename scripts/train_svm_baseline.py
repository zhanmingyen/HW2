import os
import re
import joblib
import argparse
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, roc_auc_score


def simple_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # lowercase
    t = text.lower()
    # remove URLs/emails
    t = re.sub(r"https?://\S+|\S+@\S+", " ", t)
    # remove non-word characters
    t = re.sub(r"[^\w\s]", " ", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_data(path: str):
    # Expecting two columns: label, message
    df = pd.read_csv(path, header=None, names=["label", "message"], encoding="latin-1")
    df = df[["label", "message"]].dropna()
    df["message_prep"] = df["message"].map(simple_preprocess)
    return df


def train_and_evaluate(csv_path: str, model_out: str):
    df = load_data(csv_path)
    X = df["message_prep"].values
    y = df["label"].map(lambda v: 1 if str(v).lower().strip() == "spam" else 0).values

    tf = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    Xt = tf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, stratify=y, random_state=42)

    clf = LinearSVC(random_state=42, max_iter=5000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    report_str = classification_report(y_test, preds, target_names=["ham", "spam"]) 
    report_dict = classification_report(y_test, preds, target_names=["ham", "spam"], output_dict=True)

    # Try to compute ROC AUC if decision_function exists
    auc = None
    if hasattr(clf, "decision_function"):
        try:
            scores = clf.decision_function(X_test)
            auc = float(roc_auc_score(y_test, scores))
        except Exception:
            auc = None

    eval = {
        "classification": report_dict,
        "roc_auc": auc,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }

    # Ensure output dirs
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    # Save model bundle
    joblib.dump({"model": clf, "tfidf": tf}, model_out)
    print(f"Saved model bundle to {model_out}")

    # Save evaluation artifacts
    eval_json = os.path.splitext(model_out)[0] + "_evaluation.json"
    eval_md = os.path.splitext(model_out)[0] + "_evaluation.md"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(eval, f, indent=2)

    with open(eval_md, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write("## Classification report\n\n")
        f.write(report_str + "\n\n")
        if auc is not None:
            f.write(f"## ROC AUC: {auc:.4f}\n\n")
        f.write(f"Training size: {eval['n_train']}\n")
        f.write(f"Test size: {eval['n_test']}\n")

    print(f"Saved evaluation artifacts to {eval_json} and {eval_md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=os.path.join("data", "sms_spam_no_header.csv"))
    parser.add_argument("--out", default=os.path.join("models", "svm_baseline.joblib"))
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV file not found: {args.csv}. Run scripts/data_ingest.py first.")

    train_and_evaluate(args.csv, args.out)


if __name__ == "__main__":
    main()
