"""
model_spacy.py
Stage 2 of the modeling pipeline: spaCy-based features + classifier.

This stage uses spaCy for linguistic feature extraction (entities, POS patterns,
noun chunks) combined with a simple classifier -- positioned as the "intermediate"
step between the TF-IDF baseline and full transformer fine-tuning.

Usage:
    python -m spacy download en_core_web_sm      # one-time setup
    python src/model_spacy.py --datadir data/ --outdir outputs/
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


def extract_spacy_features(texts, nlp):
    """
    Extracts linguistic features beyond plain TF-IDF:
    - count of named entities by type (MONEY, DATE, ORG, CARDINAL, etc.)
    - count of modal/obligation verbs (shall, must, may) -- legally meaningful
    - sentence length, average word length
    """
    rows = []
    obligation_words = {"shall", "must", "will", "may", "should"}
    for doc in nlp.pipe(texts, batch_size=32):
        ent_counts = {}
        for ent in doc.ents:
            ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
        modal_count = sum(1 for tok in doc if tok.text.lower() in obligation_words)
        rows.append({
            "n_tokens": len(doc),
            "n_entities": len(doc.ents),
            "n_money_ent": ent_counts.get("MONEY", 0),
            "n_date_ent": ent_counts.get("DATE", 0),
            "n_cardinal_ent": ent_counts.get("CARDINAL", 0),
            "n_org_ent": ent_counts.get("ORG", 0),
            "modal_count": modal_count,
            "avg_word_len": np.mean([len(t.text) for t in doc]) if len(doc) > 0 else 0,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    args = parser.parse_args()

    train = pd.read_csv(f"{args.datadir}/train.csv")
    val = pd.read_csv(f"{args.datadir}/val.csv")
    test = pd.read_csv(f"{args.datadir}/test.csv")
    full_train = pd.concat([train, val], ignore_index=True)

    print(f"Loading spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model)

    print("Extracting linguistic features (entities, modal verbs, structure)...")
    train_ling = extract_spacy_features(full_train["text"], nlp)
    test_ling = extract_spacy_features(test["text"], nlp)

    # Combine spaCy linguistic features with a light TF-IDF signal
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(full_train["text"])
    X_test_tfidf = vectorizer.transform(test["text"])

    scaler = StandardScaler()
    X_train_ling = scaler.fit_transform(train_ling)
    X_test_ling = scaler.transform(test_ling)

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_ling)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_ling)])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, full_train["label"])

    preds = clf.predict(X_test)
    y_true = test["label"]

    acc = accuracy_score(y_true, preds)
    f1_macro = f1_score(y_true, preds, average="macro")
    f1_weighted = f1_score(y_true, preds, average="weighted")
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)

    print(f"\n=== spaCy Features + Logistic Regression ===")
    print(f"Accuracy:      {acc:.3f}")
    print(f"F1 (macro):    {f1_macro:.3f}")
    print(f"F1 (weighted): {f1_weighted:.3f}\n")
    print(classification_report(y_true, preds, zero_division=0))

    joblib.dump(clf, f"{args.outdir}/spacy_model.joblib")
    joblib.dump(vectorizer, f"{args.outdir}/spacy_vectorizer.joblib")
    joblib.dump(scaler, f"{args.outdir}/spacy_scaler.joblib")

    results = {
        "model": "spaCy linguistic features + TF-IDF + Logistic Regression",
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_report": report,
    }
    with open(f"{args.outdir}/spacy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model and metrics to {args.outdir}")


if __name__ == "__main__":
    main()
