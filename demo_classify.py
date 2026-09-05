"""
================================================================================
 LIVE DEMO  --  Supply Chain Contract Risk Clause Classifier
 Pathi Nithin Sai (31260087)  |  ISM International School of Management
================================================================================

WHAT THIS DOES
    Loads your trained model and classifies contract clauses in real time.
    Your professor can paste in any clause and see the prediction instantly.

MODES (chosen automatically, best first)
    1. Fine-tuned Legal-BERT checkpoint   -- best, if you saved one
    2. Fine-tuned BERT checkpoint         -- good fallback
    3. TF-IDF baseline retrained on the fly (~30 seconds) -- always works
       provided data/train.csv exists

"""

import os
import sys
import glob
import time
import textwrap

# ------------------------------------------------------------------ settings
LABELS = [
    "Delivery_Obligations",
    "Force_Majeure",
    "Liability",
    "Payment_Terms",
    "Termination",
    "Warranty",
]

# Folders the script will look in for a saved model checkpoint.
CHECKPOINT_HINTS = [
    "outputs/legalbert", "outputs/legalbert_model", "outputs/legal-bert",
    "outputs/bert", "outputs/bert_model",
    "models/legalbert", "models/bert",
    "outputs", "models", ".",
]

DATA_HINTS = ["data/train.csv", "train.csv", "data/cuad_clauses.csv", "cuad_clauses.csv"]

# Clauses used for the automatic demo pass.
SAMPLES = [
    ("Force_Majeure",
     "Neither party shall be liable for any delay or failure to perform its "
     "obligations under this Agreement where such delay or failure results from "
     "events beyond its reasonable control, including acts of God, flood, "
     "earthquake, epidemic or pandemic, war, or any action taken by a government "
     "or public authority."),
    ("Liability",
     "In no event shall either party be liable for any indirect, incidental, "
     "special or consequential damages arising out of or related to this "
     "agreement, however caused and on any theory of liability, even if such "
     "party has been advised of the possibility of such damages."),
    ("Termination",
     "Either party may terminate this Agreement at any time, for any reason or "
     "no reason, upon sixty (60) days' prior written notice to the other party."),
    ("Warranty",
     "Seller warrants to Buyer that for a period of twenty-four (24) months from "
     "the date of delivery, the Product shall be free from defects in material "
     "and workmanship under normal use and service conditions."),
    ("Payment_Terms",
     "Customer shall purchase a minimum of five thousand (5,000) units of the "
     "Product in the aggregate during each calendar year during the Term of this "
     "Agreement."),
    ("Delivery_Obligations",
     "For a period of twelve (12) months following the expiration or termination "
     "of this Agreement for any reason, Supplier shall continue to provide the "
     "Services to Company on substantially the same terms as set forth herein."),
]


# ------------------------------------------------------------------ printing
def rule(ch="=", n=78):
    print(ch * n)


def banner():
    rule()
    print(" SUPPLY CHAIN CONTRACT RISK CLAUSE CLASSIFIER")
    print(" Pathi Nithin Sai (31260087)  |  ISM International School of Management")
    print(" Master's Thesis: Contract Risk Analysis in Supply Chain Using NLP")
    rule()
    print()


def show_prediction(text, label, confidence, all_scores=None, expected=None):
    print()
    rule("-")
    preview = textwrap.shorten(text.replace("\n", " "), width=300, placeholder=" ...")
    for line in textwrap.wrap(preview, width=76):
        print("  " + line)
    rule("-")
    mark = ""
    if expected:
        mark = "  [MATCHES EXPECTED]" if label == expected else f"  [expected {expected}]"
    print(f"  PREDICTION : {label}{mark}")
    if confidence is not None:
        bar = "#" * int(confidence * 40)
        print(f"  CONFIDENCE : {confidence:.1%}  {bar}")
    if all_scores:
        print("  ALL SCORES :")
        for lbl, sc in sorted(all_scores.items(), key=lambda x: -x[1]):
            b = "#" * int(sc * 30)
            print(f"      {lbl:<24} {sc:6.1%}  {b}")
    rule("-")
    print()


# ------------------------------------------------------------- model loading
def find_checkpoint():
    """Look for a HuggingFace checkpoint folder (must contain a config.json)."""
    for hint in CHECKPOINT_HINTS:
        if not os.path.isdir(hint):
            continue
        if os.path.exists(os.path.join(hint, "config.json")):
            return hint
        # one level down, e.g. outputs/checkpoint-864
        for sub in sorted(glob.glob(os.path.join(hint, "*"))):
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "config.json")):
                return sub
    return None


def load_transformer():
    """Return (predict_fn, description) or (None, reason)."""
    ckpt = find_checkpoint()
    if ckpt is None:
        return None, "no saved checkpoint folder found"

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        return None, f"transformers/torch not installed ({e})"

    try:
        print(f"  Loading model from: {ckpt}")
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        model.eval()
    except Exception as e:
        return None, f"could not load checkpoint at {ckpt} ({e})"

    # Prefer label names stored in the checkpoint config, else fall back.
    id2label = getattr(model.config, "id2label", None) or {}
    if id2label and not all(str(v).startswith("LABEL_") for v in id2label.values()):
        names = [id2label[i] for i in sorted(id2label)]
    else:
        names = LABELS

    def predict(text):
        enc = tok(text, return_tensors="pt", truncation=True,
                  padding=True, max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        scores = {names[i]: probs[i] for i in range(min(len(names), len(probs)))}
        best = max(scores, key=scores.get)
        return best, scores[best], scores

    kind = "Legal-BERT" if "legal" in ckpt.lower() else "BERT"
    return predict, f"{kind} (fine-tuned checkpoint: {ckpt})"


def load_tfidf():
    """Retrain the TF-IDF baseline on the fly. Always available if data exists."""
    data_path = next((p for p in DATA_HINTS if os.path.exists(p)), None)
    if data_path is None:
        return None, "no training CSV found (looked for data/train.csv, train.csv, ...)"

    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as e:
        return None, f"pandas/scikit-learn not installed ({e})"

    try:
        print(f"  Training TF-IDF baseline from: {data_path}  (about 30 seconds)")
        df = pd.read_csv(data_path)

        text_col = next((c for c in ("text", "clause", "clause_text", "sentence")
                         if c in df.columns), None)
        label_col = next((c for c in ("label", "category", "class", "target")
                          if c in df.columns), None)
        if text_col is None or label_col is None:
            return None, (f"could not identify text/label columns in {data_path}; "
                          f"found columns: {list(df.columns)[:10]}")

        df = df.dropna(subset=[text_col, label_col])

        pipe = make_pipeline(
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            sublinear_tf=True, stop_words="english"),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        t0 = time.time()
        pipe.fit(df[text_col].astype(str), df[label_col].astype(str))
        print(f"  Trained on {len(df):,} clauses in {time.time() - t0:.1f}s")
    except Exception as e:
        return None, f"training failed ({e})"

    classes = list(pipe.classes_)

    def predict(text):
        probs = pipe.predict_proba([text])[0]
        scores = {classes[i]: float(probs[i]) for i in range(len(classes))}
        best = max(scores, key=scores.get)
        return best, scores[best], scores

    return predict, f"TF-IDF + Logistic Regression (trained live from {data_path})"


def get_model():
    print("Looking for a trained model...")
    print()
    predict, desc = load_transformer()
    if predict is not None:
        print(f"  OK -- using {desc}")
        print()
        return predict, desc
    print(f"  Transformer unavailable: {desc}")
    print("  Falling back to the TF-IDF baseline...")
    print()

    predict, desc = load_tfidf()
    if predict is not None:
        print(f"  OK -- using {desc}")
        print()
        return predict, desc

    print(f"  TF-IDF unavailable: {desc}")
    print()
    rule()
    print(" COULD NOT LOAD ANY MODEL")
    rule()
    print(" Run this script from inside your thesis_pipeline folder, where")
    print(" either a saved checkpoint (outputs/... with config.json) or a")
    print(" training CSV (data/train.csv) is present.")
    print()
    print(f" Current folder: {os.getcwd()}")
    print(f" Contents: {sorted(os.listdir('.'))[:15]}")
    rule()
    sys.exit(1)


# -------------------------------------------------------------------- modes
def run_samples(predict):
    print()
    rule()
    print(" AUTOMATIC DEMO -- six clauses, one per risk category")
    rule()
    correct = 0
    for expected, text in SAMPLES:
        label, conf, scores = predict(text)
        if label == expected:
            correct += 1
        show_prediction(text, label, conf, scores, expected=expected)
    rule()
    print(f" DEMO RESULT: {correct}/{len(SAMPLES)} matched the expected category")
    rule()
    print()


def run_interactive(predict, desc):
    rule()
    print(" INTERACTIVE MODE")
    rule()
    print(" Paste a contract clause, then press Enter on an empty line.")
    print(" Type 'samples' to re-run the demo clauses, or 'quit' to exit.")
    print()

    while True:
        try:
            print("Clause> ", end="", flush=True)
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            text = " ".join(lines).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            return
        if text.lower() == "samples":
            run_samples(predict)
            continue

        label, conf, scores = predict(text)
        show_prediction(text, label, conf, scores)


def main():
    banner()
    predict, desc = get_model()
    run_samples(predict)
    print(f" Model in use: {desc}")
    print()
    run_interactive(predict, desc)


if __name__ == "__main__":
    main()
