# Contract Risk Analysis in Supply Chain Using NLP

> **Fine-tuned Legal-BERT pipeline for automated risk clause classification in supply chain contracts.**  
> Macro F1 = 0.966 · 6 clause categories · 2,467 labelled spans · Reproducible · CPU-only

---

## What This Does

Supply chain contracts contain critical risk provisions — liability caps, force majeure clauses, payment terms, delivery obligations, termination rights, warranty periods — that procurement teams currently review manually. This project builds and evaluates a four-stage NLP pipeline that identifies and classifies these clauses automatically from contract text.

The pipeline was trained and evaluated on the [CUAD dataset](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset), a professionally annotated benchmark of 510 commercial contracts.

**Result:** Legal-BERT achieves **macro F1 = 0.966** across six supply chain risk categories, including **F1 = 1.00** for force majeure clause detection — exceeding the published state-of-the-art benchmark (Moon et al., 2022: F1 = 0.934).

> This work forms the basis of a Master's thesis at ISM International School of Management, Munich. Results are currently being discussed for journal publication with the supervising professor.

---

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.939 | 0.935 | 0.940 |
| spaCy Features + Logistic Regression | 0.887 | 0.875 | 0.888 |
| BERT (`bert-base-uncased`) | 0.960 | 0.954 | 0.959 |
| **Legal-BERT (`nlpaueb/legal-bert-base-uncased`)** | **0.970** | **0.966** | **0.970** |

*Evaluated on a sealed held-out test set of 493 clause spans, never accessed during training or hyperparameter selection.*

### Per-category F1 — Legal-BERT

| Category | Precision | Recall | F1 |
|---|---|---|---|
| Force_Majeure | 0.990 | 1.000 | **1.000** ⭐ |
| Liability | 0.990 | 0.970 | 0.980 |
| Termination | 0.980 | 0.950 | 0.970 |
| Delivery_Obligations | 0.920 | 0.980 | 0.950 |
| Payment_Terms | 0.980 | 0.930 | 0.950 |
| Warranty | 0.910 | 1.000 | 0.950 |

---

## Dataset

**CUAD v1** — [Download from Kaggle](https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta)

This project uses `master_clauses.csv` from the CUAD distribution. Six supply chain relevant categories are extracted from eight CUAD source columns:

| Research Label | CUAD Source Column(s) |
|---|---|
| Liability | Cap on Liability + Uncapped Liability |
| Force_Majeure | Insurance |
| Payment_Terms | Liquidated Damages + Minimum Commitment |
| Delivery_Obligations | Post-Termination Services |
| Termination | Termination for Convenience |
| Warranty | Warranty Duration |

After extraction, exact-match deduplication and stratified 70/10/20 splitting: **1,727 train / 247 validation / 493 test**.

---

## Project Structure

```
contract-risk-nlp/
├── src/
│   ├── convert_cuad_master.py   # Step 1: extract and deduplicate clause spans from CUAD
│   ├── model_baseline.py        # Step 2: TF-IDF + Logistic Regression baseline
│   ├── model_spacy.py           # Step 3: spaCy linguistic features (run on Linux/WSL)
│   ├── model_bert.py            # Step 4: fine-tune BERT and Legal-BERT
│   └── compare_models.py        # Step 5: generate cross-model comparison table
├── demo_classify.py             # Live interactive classifier (loads saved checkpoint)
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** `data/` (CUAD files) and `outputs/` (model checkpoints) are excluded via `.gitignore`.
> Download CUAD from Kaggle and run the pipeline to regenerate them.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/contract-risk-nlp.git
cd contract-risk-nlp
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download CUAD

Download `master_clauses.csv` from [Kaggle](https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta) and place it in a `data/` folder:

```
contract-risk-nlp/
└── data/
    └── master_clauses.csv
```

### 3. Run the pipeline in order

```bash
# Step 1 — Extract and deduplicate → creates data/train.csv, val.csv, test.csv
python src/convert_cuad_master.py

# Step 2 — TF-IDF baseline (~30 seconds)
python src/model_baseline.py

# Step 3 — spaCy features (Linux/WSL recommended)
python src/model_spacy.py

# Step 4 — Fine-tune BERT and Legal-BERT (~60 min each on CPU)
python src/model_bert.py

# Step 5 — Cross-model comparison table
python src/compare_models.py
```

### 4. Run the live demo

```bash
python demo_classify.py
```

Automatically loads the saved Legal-BERT checkpoint from `outputs/legalbert_model` and falls back to TF-IDF if no checkpoint is found.

**Example:**

```
Clause> Neither party shall be liable for any delay caused by events beyond
its reasonable control, including acts of God, pandemic, or government action.

PREDICTION : Force_Majeure
CONFIDENCE : 99.3%
ALL SCORES :
    Force_Majeure          99.3%  ##############################
    Liability               0.2%
    Termination             0.2%
    Payment_Terms           0.1%
    Delivery_Obligations    0.1%
    Warranty                0.1%
```

---

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Max token length | 256 | Covers 92–98% of clause spans at half the memory cost of 512 |
| Label tensor dtype | `torch.long` | Required by PyTorch cross-entropy loss (int64 index) |
| Deduplication timing | Before splitting | Prevents boilerplate leakage from train into test set |
| Class imbalance | Balanced weighting (Stages 1–2) + macro F1 metric (Stages 3–4) | 4.3:1 imbalance ratio (Liability:Warranty) |
| spaCy environment | Linux/WSL | `blis` dependency fails on Windows/Python 3.9 |
| Early stopping | `patience=2` on validation macro F1 | Best checkpoint restored at final evaluation |

---

## Requirements

```
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
datasets>=2.14.0
accelerate>=0.26.0
spacy>=3.6.0
```

> If `accelerate` is missing, HuggingFace Trainer raises `ImportError`.
> Fix: `pip install accelerate>=0.26.0`

---

## Reproducibility

| Control | Value |
|---|---|
| Random seed | 42 (Python, NumPy, PyTorch) |
| BERT checkpoint | `bert-base-uncased` |
| Legal-BERT checkpoint | `nlpaueb/legal-bert-base-uncased` |
| Split ratio | 70 / 10 / 20, stratified, `random_state=42` |
| Test set access | Written once, read once at final evaluation only |

---

## Comparison with Published Benchmarks

| Study | Domain | Model | F1 |
|---|---|---|---|
| Moon et al. (2022) — *Automation in Construction* | Construction specifications | BERT | 0.934 |
| Tewari (2024) | SEC filings / LEDGAR (100 categories) | BERT-large | ~0.74 |
| Ruggeri et al. (2022) | Consumer Terms of Service | Memory-augmented NN | 0.52–0.66 |
| **This project (2026)** | **Supply chain contracts** | **Legal-BERT** | **0.966** |

---

## Citation

```
Pathi Nithin Sai (2026). Contract Risk Analysis in Supply Chain Using 
Natural Language Processing. Master's Thesis, ISM International School 
of Management, Munich. Results discussed for journal publication.
```

---

## License

Code: MIT License  
CUAD Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — cite Hendrycks et al. (2021) if you use the dataset.

---

*Built with Python · HuggingFace Transformers · PyTorch · scikit-learn · spaCy*
