"""
model_bert.py
Fine-tune BERT or Legal-BERT for clause classification.

Usage:
    python src/model_bert.py --datadir data/ --outdir outputs/ \
        --model_name bert-base-uncased --epochs 4 --tag bert

    python src/model_bert.py --datadir data/ --outdir outputs/ \
        --model_name nlpaueb/legal-bert-base-uncased --epochs 4 --tag legalbert
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)


class ClauseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True, max_length=max_length
        )
        # FIX: cast to torch.long (int64) — required by PyTorch cross-entropy loss
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy"   : accuracy_score(labels, preds),
        "f1_macro"   : f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",     required=True)
    parser.add_argument("--outdir",      required=True)
    parser.add_argument("--model_name",  default="bert-base-uncased")
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--tag",         default="bert")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    train = pd.read_csv(os.path.join(args.datadir, "train.csv"))
    val   = pd.read_csv(os.path.join(args.datadir, "val.csv"))
    test  = pd.read_csv(os.path.join(args.datadir, "test.csv"))

    # Encode string labels -> integers
    le = LabelEncoder()
    le.fit(pd.concat([train["label"], val["label"], test["label"]]))
    train_y = le.transform(train["label"])
    val_y   = le.transform(val["label"])
    test_y  = le.transform(test["label"])
    num_labels = len(le.classes_)

    print(f"\nModel   : {args.model_name}")
    print(f"Train   : {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Classes : {list(le.classes_)}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )

    train_ds = ClauseDataset(train["text"], train_y, tokenizer)
    val_ds   = ClauseDataset(val["text"],   val_y,   tokenizer)
    test_ds  = ClauseDataset(test["text"],  test_y,  tokenizer)

    training_args = TrainingArguments(
        output_dir              = os.path.join(args.outdir, f"{args.tag}_checkpoints"),
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate           = args.lr,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1_macro",
        logging_steps           = 20,
        report_to               = "none",
        save_total_limit        = 1,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...\n")
    trainer.train()

    # ── Final evaluation on held-out test set ──────────────────────────────────
    print("\nEvaluating on test set...")
    test_output = trainer.predict(test_ds)
    test_preds  = np.argmax(test_output.predictions, axis=-1)

    acc        = accuracy_score(test_y, test_preds)
    f1_macro   = f1_score(test_y, test_preds, average="macro")
    f1_weighted= f1_score(test_y, test_preds, average="weighted")
    report     = classification_report(
        test_y, test_preds, target_names=le.classes_,
        output_dict=True, zero_division=0
    )

    print(f"\n=== {args.model_name} | TEST SET RESULTS ===")
    print(f"Accuracy      : {acc:.3f}")
    print(f"F1 (macro)    : {f1_macro:.3f}")
    print(f"F1 (weighted) : {f1_weighted:.3f}\n")
    print(classification_report(test_y, test_preds, target_names=le.classes_, zero_division=0))

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "model"          : args.model_name,
        "tag"            : args.tag,
        "accuracy"       : acc,
        "f1_macro"       : f1_macro,
        "f1_weighted"    : f1_weighted,
        "per_class_report": report,
    }
    results_path = os.path.join(args.outdir, f"{args.tag}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    model.save_pretrained(os.path.join(args.outdir, f"{args.tag}_model"))
    tokenizer.save_pretrained(os.path.join(args.outdir, f"{args.tag}_model"))
    print(f"Saved results -> {results_path}")
    print(f"Saved model   -> {args.outdir}\\{args.tag}_model\\")


if __name__ == "__main__":
    main()
