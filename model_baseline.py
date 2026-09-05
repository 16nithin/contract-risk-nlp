"""
model_baseline.py
Stage 1 of the modeling pipeline: TF-IDF + Logistic Regression baseline.

Usage:
    python src/model_baseline.py --datadir data/ --outdir outputs/
"""
import argparse
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    train = pd.read_csv(f"{args.datadir}/train.csv")
    val = pd.read_csv(f"{args.datadir}/val.csv")
    test = pd.read_csv(f"{args.datadir}/test.csv")

    # Combine train+val for the baseline (no early stopping needed for LR)
    full_train = pd.concat([train, val], ignore_index=True)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(full_train["text"])
    X_test = vectorizer.transform(test["text"])

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # important: clause categories are imbalanced
    )
    clf.fit(X_train, full_train["label"])

    preds = clf.predict(X_test)
    y_true = test["label"]

    acc = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average="macro")
    f1_weighted = f1_score(y_true, preds, average="weighted")
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)

    print(f"=== TF-IDF + Logistic Regression Baseline ===")
    print(f"Accuracy:      {acc:.3f}")
    print(f"F1 (macro):    {f1_macro:.3f}")
    print(f"F1 (weighted): {f1_weighted:.3f}\n")
    print(classification_report(y_true, preds, zero_division=0))

    # Save model + vectorizer for reuse, and metrics for the results chapter
    joblib.dump(clf, f"{args.outdir}/baseline_model.joblib")
    joblib.dump(vectorizer, f"{args.outdir}/baseline_vectorizer.joblib")

    results = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_report": report,
    }
    with open(f"{args.outdir}/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model, vectorizer, and metrics to {args.outdir}")


if __name__ == "__main__":
    main()
