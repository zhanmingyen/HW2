import os
import csv
import math
import json
import random
from collections import defaultdict, Counter


def simple_tokenize(text):
    if not isinstance(text, str):
        return []
    t = text.lower()
    # keep words and numbers
    tokens = []
    word = []
    for ch in t:
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                tokens.append(''.join(word))
                word = []
    if word:
        tokens.append(''.join(word))
    return tokens


def load_csv(path):
    rows = []
    with open(path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            label = row[0].strip()
            text = row[1].strip()
            rows.append((label, text))
    return rows


def train_naive_bayes(train_rows):
    class_counts = Counter()
    token_counts = defaultdict(Counter)
    vocab = set()

    for label, text in train_rows:
        cls = 'spam' if label.lower() == 'spam' else 'ham'
        class_counts[cls] += 1
        tokens = simple_tokenize(text)
        for t in tokens:
            token_counts[cls][t] += 1
            vocab.add(t)

    total_docs = sum(class_counts.values())
    priors = {cls: class_counts[cls] / total_docs for cls in class_counts}

    # Precompute denominators for likelihood with Laplace smoothing
    denom = {}
    for cls in token_counts:
        denom[cls] = sum(token_counts[cls].values()) + len(vocab)

    model = {
        'priors': priors,
        'token_counts': {cls: dict(token_counts[cls]) for cls in token_counts},
        'vocab': list(vocab),
        'denom': denom,
    }
    return model


def predict(model, text):
    tokens = simple_tokenize(text)
    scores = {}
    vocab = set(model['vocab'])
    for cls in model['priors']:
        score = math.log(model['priors'][cls]) if model['priors'][cls] > 0 else -1e9
        tc = model['token_counts'].get(cls, {})
        denom = model['denom'].get(cls, 1)
        for t in tokens:
            count = tc.get(t, 0) + 1  # Laplace
            score += math.log(count / denom)
        scores[cls] = score
    # return class with max score
    return max(scores.items(), key=lambda x: x[1])[0]


def evaluate(model, rows):
    y_true = []
    y_pred = []
    for label, text in rows:
        cls_true = 'spam' if label.lower() == 'spam' else 'ham'
        cls_pred = predict(model, text)
        y_true.append(cls_true)
        y_pred.append(cls_pred)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 'spam' and p == 'spam')
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 'ham' and p == 'ham')
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 'ham' and p == 'spam')
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 'spam' and p == 'ham')

    accuracy = (tp + tn) / len(y_true) if y_true else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'n': len(y_true),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    data_path = os.path.join('data', 'sms_spam_no_header.csv')
    if not os.path.exists(data_path):
        print('Dataset not found:', data_path)
        return 1

    rows = load_csv(data_path)
    random.seed(42)
    random.shuffle(rows)

    split = int(0.8 * len(rows))
    train_rows = rows[:split]
    test_rows = rows[split:]

    model = train_naive_bayes(train_rows)

    eval_train = evaluate(model, train_rows)
    eval_test = evaluate(model, test_rows)

    os.makedirs('models', exist_ok=True)
    # Save model as JSON (simple)
    model_out = os.path.join('models', 'simple_nb_model.json')
    with open(model_out, 'w', encoding='utf-8') as f:
        json.dump({'priors': model['priors'], 'token_counts': model['token_counts']}, f)

    # Save evaluation
    out_json = os.path.join('models', 'simple_nb_evaluation.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'train': eval_train, 'test': eval_test}, f, indent=2)

    out_md = os.path.join('models', 'simple_nb_evaluation.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('# Simple NB Evaluation\n\n')
        f.write('## Train\n')
        f.write(json.dumps(eval_train, indent=2))
        f.write('\n\n## Test\n')
        f.write(json.dumps(eval_test, indent=2))

    print('Saved lightweight model and evaluation to models/')
    print('Test evaluation:', eval_test)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
