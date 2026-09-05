"""
convert_cuad.py
STARTER SCRIPT — you must adjust this after downloading the real CUAD dataset.

CUAD is distributed in a SQuAD-style JSON format:
{
  "data": [
    {
      "title": "<contract filename>",
      "paragraphs": [
        {
          "context": "<full contract text>",
          "qas": [
            {
              "question": "Highlight the parts (if any) of this contract related to 'Payment Terms' ...",
              "id": "...",
              "answers": [
                {"text": "<clause text>", "answer_start": 1234}
              ],
              "is_impossible": false
            },
            ...
          ]
        }
      ]
    }
  ]
}

Each "question" encodes a clause category (e.g. "Payment Terms", "Termination
For Convenience", "Cap On Liability", "Uncapped Liability", "Force Majeure",
"Warranty Duration", "Covenant Not To Sue", etc — CUAD has ~41 of these).

This script's job: flatten that nested structure into a simple CSV with
columns `text` and `label`, where `label` is your chosen supply-chain-relevant
category (you'll map several of CUAD's 41 raw categories into your six
research categories — see CATEGORY_MAP below).

USAGE (after downloading CUAD's JSON, e.g. CUAD_v1.json):
    python data/convert_cuad.py --input data/CUAD_v1.json --output data/cuad_clauses.csv

NOTE: You MUST open the actual downloaded JSON and confirm these field names
match (CUAD has had a couple of minor format revisions). Print
`list(raw_json.keys())` and inspect one example record before trusting this
script blindly.
"""
import argparse
import json
import re
import pandas as pd


# Map CUAD's exact question categories to your six supply-chain-relevant
# research categories. ADJUST these strings to match the exact category names
# in the CUAD version you download -- inspect a few "question" fields first.
CATEGORY_MAP = {
    "Payment Terms": "Payment_Terms",
    "Cap On Liability": "Liability",
    "Uncapped Liability": "Liability",
    "Limitation Of Liability": "Liability",
    "Termination For Convenience": "Termination",
    "Termination": "Termination",
    "Force Majeure": "Force_Majeure",
    "Covenant Not To Sue": "Delivery_Obligations",   # example mapping; verify against real category list
    "Delivery": "Delivery_Obligations",
    "Warranty Duration": "Warranty",
    "Warranty": "Warranty",
}


def extract_clause_question(question_text: str) -> str:
    """
    CUAD's 'question' field is a long templated sentence, e.g.:
    "Highlight the parts (if any) of this contract related to 'Payment Terms'
    that should be reviewed by a lawyer."
    This pulls out the quoted category name.
    """
    match = re.search(r"'([^']+)'", question_text)
    return match.group(1) if match else question_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CUAD JSON file")
    parser.add_argument("--output", required=True, help="Path to write flattened CSV")
    args = parser.parse_args()

    with open(args.input) as f:
        raw = json.load(f)

    print("Top-level keys in CUAD JSON:", list(raw.keys()))
    print("If 'data' is not among them, inspect the file manually and adjust this script.\n")

    rows = []
    skipped_categories = set()

    for contract in raw["data"]:
        for paragraph in contract["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa.get("is_impossible", True):
                    continue  # no answer for this category in this contract
                category_raw = extract_clause_question(qa["question"])
                category = CATEGORY_MAP.get(category_raw)
                if category is None:
                    skipped_categories.add(category_raw)
                    continue
                for ans in qa["answers"]:
                    text = ans["text"].strip()
                    if len(text) > 10:  # skip near-empty spans
                        rows.append({"text": text, "label": category})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df)} clauses to {args.output}")
    print(df["label"].value_counts())
    print(f"\nSkipped {len(skipped_categories)} CUAD categories not in CATEGORY_MAP "
          f"(this is expected -- CUAD has ~41 categories, you only want ~6).")
    print("First 10 skipped category names (for reference):", list(skipped_categories)[:10])


if __name__ == "__main__":
    main()
