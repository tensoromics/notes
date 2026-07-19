# nano-Geneformer (IBD)

A from-scratch, laptop-sized **Geneformer**: a tiny BERT encoder that reads a
single cell as a **rank-ordered list of gene tokens** and is pretrained by
**masked gene prediction**. Then we probe what it learned about IBD (ulcerative
colitis) — disease state and in-silico gene knockouts — the way you'd probe a
real single-cell foundation model, but small enough to train on an M-series Mac.

Karpathy-style: every piece is here to read (~1–5M params, plain PyTorch). It
runs on `mps` (Apple Metal), `cuda`, or `cpu`, auto-detected.

---

## 0. Setup

- **Environment:** Python 3.11 or 3.12. Create a virtual environment and install
  the dependencies:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
  Everything runs on `mps` (Apple Metal), `cuda`, or `cpu`, auto-detected.
- **Data:** download the Smillie et al. 2019 UC atlas from the Broad Single Cell
  Portal ([SCP259](https://singlecell.broadinstitute.org/single_cell/study/SCP259))
  into a `smillie/` folder next to the scripts — the three compartment matrices
  (Epi/Fib/Imm) plus `Smillie_meta2.txt`. `prepare.py` auto-detects that layout.
  The atlas is ~5 GB, so it is not included in this repo.
- **Optional:** `all.meta2.txt` from SCP259 adds batch/sex covariates;
  `load_smillie.py` merges its columns automatically if a valid copy is present.

---

## 1. Open in VSCode

1. Open this folder in VSCode.
2. Command Palette → **Python: Select Interpreter** → pick `./.venv/bin/python`.
3. Open the **Run and Debug** panel (⇧⌘D). The dropdown has four stages plus a
   fast subsampled prep — press ▶ to run any of them, or use the terminal
   commands below.

> In the integrated terminal, the venv auto-activates, so `python prepare.py`
> works. Outside it, prefix with the venv: `.venv/bin/python prepare.py`.

---

## 2. Run the pipeline

Run in order. Stage 1 writes to `data/`, stage 2 writes to `checkpoints/`.

| # | Run-panel entry | Terminal command | What it does |
|---|---|---|---|
| 1 | `1 · prepare (Smillie)` | `python prepare.py` | rank-encode all 365k cells |
| 1b | `1b · prepare (fast 20k)` | `python prepare.py --subsample 20000` | same, ~20k cells (quick smoke test) |
| 2 | `2 · train` | `python train.py --epochs 5` | masked-gene-prediction pretraining |
| 3 | `3 · probe` | `python probe.py` | embedding vs PCA at reading disease state |
| 4 | `4 · perturb` | `python perturb.py` | in-silico gene knockouts |

**Do the fast pass first** (`1b` → `2 --epochs 1` → `3` → `4`) to confirm the
whole loop works end-to-end in a few minutes, then run the full thing.

### Stage 1 — `prepare.py`
Loads the three compartments, QC-filters, CP10k-normalizes, keeps the top
`n_hvg` (2048) highly-variable genes as the vocabulary, applies the Geneformer
per-gene-median normalization, ranks genes within each cell, and writes token
sequences plus a deterministic **patient-level** split and a 50-dim PCA baseline.

- **Runtime:** roughly **10–30 min** on the full atlas — dominated by
  `scipy.io.mmread` grinding through ~387M non-zeros. Memory-heavy; if it stalls
  or OOMs, use `--subsample`.
- **Outputs (`data/`):** `tokens.npy`, `pca.npy`, `gene_median.npy`, `meta.npz`,
  `vocab.json`.
- **Success looks like:** it prints the final cell/gene counts, the patient split
  (≈ N patients, ~20% held out), and `Label classes: ['Healthy', 'Inflamed',
  'Non-inflamed']`.

### Stage 2 — `train.py`
Masked gene prediction (15% masked, BERT-style). Pretrains on the **train**
patients only so the val patients stay clean for probing.

- **Runtime:** ballpark **30–90 min for 5 epochs** on M-series; start with
  `--epochs 1`. Watch the loss fall in the tqdm bar.
- **Output:** `checkpoints/model.pt`.

### Stage 3 — `probe.py`
Freezes the encoder, mean-pools a cell embedding, and trains logistic regression
to predict disease state — **evaluated on held-out patients** — against a 50-dim
PCA baseline.

- **What to look for:** the nano-Geneformer embedding **beating PCA** on macro-F1
  = pretraining learned disease-relevant structure. If it *doesn't* beat PCA,
  that's a real result (see caveat 3), not a bug.

### Stage 4 — `perturb.py`
Deletes each candidate gene's token from inflamed cells and ranks genes by how
far they push the cell toward the healthy centroid.

- **What to look for:** **TNF / OSM / IL1B** near the top (flagged `<-- watchlist`)
  = a computational echo of what anti-TNF does. OSM predicting anti-TNF
  *non-response* is real biology (West et al. 2017, Nat Med), so an OSM hit is
  literature-validated, not a toy.

---

## 3. Config knobs — `config.py`

Everything is in one dataclass. Common edits:

- `n_hvg`, `max_len` — vocabulary size and sequence length (smaller = faster).
- `d_model`, `n_layers`, `n_heads` — model size.
- `epochs`, `batch_size`, `lr` — training.
- `val_every` — patient-split ratio (every Nth patient → val; 5 ≈ 20%).
- `patient_col` / `label_col` / `celltype_col` — already set to Smillie's
  `Subject` / `Health` / `Cluster`. If you switch datasets and `prepare.py` can't
  find a column, it prints the columns your file actually has.

---

## 4. Caveats — read before believing any result

1. **Split by patient, never by cell.** Disease signal is heavily confounded with
   patient/batch identity; a model can "predict disease" by memorizing batch.
   This repo holds out whole patients. Be suspicious of anything too clean —
   and once you re-download `all.meta2.txt`, you can also hold out / stratify by
   **batch** directly.
2. **Small patient n.** 365k cells still come from only ~tens of patients.
   Cell-level tasks are fine; patient-level clinical prediction is underpowered.
3. **Rank encoding discards magnitude,** and inflammation lives in fold-changes.
   If the embedding doesn't beat PCA, that may be *why* — try a value-aware
   tokenization next.
4. **A nano model won't beat real Geneformer/scGPT.** Not the goal. The goal is to
   build every part and watch real IBD biology emerge.

---

## 5. Files

| file | role |
|------|------|
| `config.py` | all hyperparameters + column names |
| `load_smillie.py` | assemble the 3 SCP259 compartments + metadata into one AnnData |
| `prepare.py` | AnnData → rank-encoded tokens + patient split + PCA baseline |
| `model.py` | tiny BERT encoder + masked-LM head + cell embedding |
| `train.py` | masked-gene-prediction pretraining loop |
| `probe.py` | frozen-embedding vs PCA disease-state probe (held-out patients) |
| `perturb.py` | in-silico gene deletion, ranked by healthward shift |

---

## 6. Next experiments

- Merge `all.meta2.txt` (batch/sex) and add them as covariates / stratifiers.
- Value-aware tokenization (bin expression) vs rank encoding — does magnitude help?
- GPT flavor: autoregressive next-gene prediction (Cell2Sentence-style).
- Transfer: pretrain on Smillie UC, probe on Martin CD (GSE134809).
- UMAP the embeddings — do inflammation-associated fibroblasts (IAFs) fall out
  unsupervised?
