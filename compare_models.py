"""
compare_models.py
Loads the *_results.json file from each model stage and produces a single
comparison table (CSV + printed) for your Chapter 5 results section.

Usage (run after baseline, spacy, bert, legalbert have all been run):
    python src/compare_models.py --outdir outputs/
"""
import argparse
import glob
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    result_files = sorted(glob.glob(f"{args.outdir}/*_results.json"))
    if not result_files:
        print(f"No *_results.json files found in {args.outdir}. Run the model scripts first.")
        return

    rows = []
    for f in result_files:
        with open(f) as fh:
            r = json.load(fh)
        rows.append({
            "Model": r.get("model", r.get("tag", f)),
            "Accuracy": round(r["accuracy"], 3),
            "F1 (macro)": round(r["f1_macro"], 3),
            "F1 (weighted)": round(r["f1_weighted"], 3),
        })

    df = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
    print("\n=== Model Comparison (Stage 1 -> Stage 4) ===\n")
    print(df.to_string(index=False))

    df.to_csv(f"{args.outdir}/model_comparison.csv", index=False)
    print(f"\nSaved comparison table to {args.outdir}/model_comparison.csv")
    print("Use this table directly in Chapter 5 (Modeling & Results) of your thesis.")


if __name__ == "__main__":
    main()
