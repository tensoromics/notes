# nano-Geneformer — a hand-held walkthrough

Running notes as we read the project together. Each section is a small, self-contained
chunk. Code snippets are trimmed to the essential lines; file/line references point at
the real source.

---

## 0. The big picture

You're building a **tiny language model — but for cells instead of sentences.** It's a
from-scratch, laptop-sized copy of the real single-cell model *Geneformer*.

The one idea everything rests on:

- a **cell** plays the role of a *sentence*,
- a **gene** plays the role of a *word*,
- the word "order" is the **ranking of genes by how strongly the cell expresses them**
  (relative to how strongly cells normally express them).

So each cell becomes a ranked list of gene tokens — a "sentence." We teach the model the
BERT way: **hide some genes and make it guess them from the rest** (masked gene
prediction). To guess well it must learn which genes travel together = real co-expression
biology, with nobody labeling a single cell.

### The four stages (run in order)

| Stage | File | Plain-language job | Output |
|------|------|--------------------|--------|
| 1 | `prepare.py` | raw cell×gene counts → ranked gene "sentences" | `data/` |
| 2 | `train.py`   | teach the model by masked-gene prediction | `checkpoints/model.pt` |
| 3 | `probe.py`   | does the learned fingerprint beat plain PCA at reading disease? | printed scores |
| 4 | `perturb.py` | delete genes in-silico; which push cells toward "healthy"? | printed ranking |

`config.py` = all knobs. `model.py` = the ~1–5M-param transformer. `load_smillie.py`
assembles the raw dataset.

---

## Stage 1 — `prepare.py`

Turns raw data into ranked gene sentences. On the fast 20k-cell run it did this:

- Loaded 365,492 cells × 18,172 genes (genes shared by all 3 tissue compartments).
- Skipped `all.meta2.txt` (a 220-byte error file — the known gap; harmless).
- Kept 20,000 random cells (fast test set).
- QC: dropped genes seen in <10 cells → 17,374 left; no cells dropped.
- Chose a **vocabulary** = the 2,048 most-variable genes.
- Rank-encoded all 20,000 cells.
- Split by patient: 30 patients, 6 held out → 16,044 train / 3,956 val.

**The payoff, concrete.** Cell 0 is a real *healthy CD8⁺ T cell*. Rank-encoded, the model
reads it as:

```
GZMA > CCL5 > TRBC2 > CD3E > CD69 > …
```

GZMA (granzyme), CCL5, the T-cell-receptor gene TRBC2, CD3E/CD69 — textbook cytotoxic
T-cell markers rose to the top on their own. Nobody labeled the cell; ranking its genes
surfaced its identity. (It expresses only 61 of 2,048 vocab genes — cells are sparse — so
the rest of its 1,024 slots are `PAD`, which the model ignores.)

Files written to `data/`: `tokens.npy` (20,000 × 1,024 — the sentences), `pca.npy` (the
baseline), `gene_median.npy`, `meta.npz`, `vocab.json`.

### 1a. QC filter + normalize (`prepare.py:78–84`)

```python
sc.pp.filter_cells(adata, min_genes=cfg.min_genes_per_cell)   # min_genes = 200
sc.pp.filter_genes(adata, min_cells=cfg.min_cells_per_gene)   # min_cells = 10
sc.pp.normalize_total(adata, target_sum=1e4)                  # CP10k
```

- drop cells expressing <200 genes (dying cells / empty droplets),
- drop genes seen in <10 cells (too rare to learn from),
- rescale every cell to sum to 10,000 counts, so a deeply- and a shallowly-sequenced cell
  compare fairly.

### 1b. Choose the 2,048-gene vocabulary + PCA baseline (`prepare.py:86–91`)

```python
adata_log = adata.copy()
sc.pp.log1p(adata_log)                                     # squash big numbers
sc.pp.highly_variable_genes(adata_log, n_top_genes=2048)  # the 2048 most-varying genes
sc.pp.pca(adata_log, n_comps=50, use_highly_variable=True)
X_pca = adata_log.obsm["X_pca"]                            # -> pca.npy
```

"Highly variable" = genes whose expression **swings a lot cell-to-cell**. Housekeeping
genes (ACTB, B2M, MALAT1) are high *everywhere*, barely vary, so they lose and never enter
the vocabulary — which is exactly what you want a cell "language" built from. `sc.pp.pca`
squeezes each cell into 50 numbers → `pca.npy`, the baseline Stage 3 tries to beat.

### 1c. Rank-encode — gene medians (`prepare.py:100–107`)

```python
Xcsc = adata.X.tocsc()
gene_median = np.ones(n_genes, dtype=np.float32)
for j in range(n_genes):                                  # each gene
    col = Xcsc.data[Xcsc.indptr[j]:Xcsc.indptr[j+1]]      # its nonzero values
    if col.size:
        gene_median[j] = np.median(col)                   # its typical level
```

Each gene's "normal" level = median of its nonzero values. (GZMA = 11.83; across the vocab
they range 0.52 → 64.29.)

**Why the slice is nonzero:** the matrix is stored **sparse** — zeros are never stored.
`.data` holds only nonzero values; `indptr[j]:indptr[j+1]` says where gene *j*'s chunk of
`.data` starts and ends. Tiny CSC example:

```
        g0  g1
cell0    5   0
cell1    0   3
cell2    2   4

data    = [5, 2, 3, 4]     # nonzero values only
indices = [0, 2, 1, 2]     # which cell each came from
indptr  = [0, 2, 4]        # g0 = data[0:2], g1 = data[2:4]
```

### 1d. Rank-encode — rank each cell into tokens (`prepare.py:109–121`)

```python
Xcsr = adata.X.tocsr()                                    # CSR: row = cell
tokens = np.zeros((n_cells, max_len), dtype=np.int64)     # all PAD (0) to start
for i in range(n_cells):                                  # each cell
    idx  = Xcsr.indices[lo:hi]                            # genes this cell expresses
    vals = Xcsr.data[lo:hi] / gene_median[idx]            # count ÷ that gene's baseline
    order = np.argsort(-vals)[:max_len]                   # sort high→low, keep top 1024
    toks  = idx[order] + cfg.N_SPECIAL                    # gene column -> token id (+2)
    tokens[i, :toks.size] = toks
```

`vals = counts / gene_median` is the whole trick: a gene ranks high **only if it's loud
for itself**, not just loud in absolute terms. `+ N_SPECIAL` (=2) turns a gene column into
a token id, because ids 0 and 1 are reserved for `PAD` and `MASK`.

**Worked example.** `gene_median = [2, 5, 10, 1]` for g0–g3; cell expresses:

```
idx  = [0, 2, 3]      data = [6, 20, 1]
gene_median[idx] = [2, 10, 1]
vals = [6/2, 20/10, 1/1] = [3.0, 2.0, 1.0]
order = argsort(-vals) = [0, 1, 2]
idx[order] = [0, 2, 3]              # g0 > g2 > g3
toks = [0,2,3] + 2 = [2, 4, 5]
tokens[i] = [2, 4, 5, 0, 0, ...]   # rest stays PAD
```

**The median flip:** g2 had the biggest raw count (20) but a high baseline (10), so it
drops *below* g0 (count 6, baseline 2). By raw counts it'd be g2 > g0; after ÷median it's
g0 > g2. That reordering is the point. (Across all 20k cells the most frequent rank-1
genes were IGKC, CD74, IGLC3, TFF3 — real cell-type markers, not housekeeping noise.)

### 1e. Patient-level split (`prepare.py:123–129`)

```python
patients = adata.obs["Subject"].astype(str).values          # each cell's patient
uniq = sorted(set(patients))                                 # unique patients, sorted
val_patients = set(uniq[::cfg.val_every])                    # every 5th -> validation
split = np.array(["val" if p in val_patients else "train"    # label each cell
                  for p in patients])
```

**Worked example** (6 patients):

```
uniq = [P0, P1, P2, P3, P4, P5]
uniq[::5] = [P0, P5]                        # val_patients
patients (per cell) = [P0, P2, P5, P1, P0]
split               = [val, train, val, train, val]
```

The split is decided **per patient, then stamped on that patient's cells** — so no patient
is ever split across train and val. Why it matters (README caveat #1): cells from one
patient share genetics/batch quirks; a random cell-split lets the model "predict disease"
by secretly recognizing *which patient* a cell came from. Holding out whole patients makes
Stage 3's score honest.

---

## Stage 2 — `train.py`

### 2a. What "masked gene prediction" is

Take a cell's ranked sentence, hide a few genes, make the model guess them from the genes
still visible.

```
full:            GZMA CCL5 TRBC2 CD3E  CD69 ...
shown to model:  GZMA CCL5 TRBC2 ____  CD69 ...     # CD3E hidden
must guess:      CD3E
```

To fill that blank the model must learn that GZMA/CCL5/TRBC2/CD69 travel with CD3E — real
co-expression biology, learned with zero labels.

### 2b. How the model "guesses" a gene

It doesn't pull CD3E from thin air. At the blank position it outputs a **score for every
gene in the vocabulary** (all 2,050); the guess is whichever scores highest.

```
gene    score   -> probability (softmax)
CD3E     4.0            0.84
CD8A     2.0            0.11
GZMK     1.0            0.04
FABP1   -1.0            0.006   (unrelated epithelial gene)
```

- scores come from the transformer reading the visible genes,
- **softmax** turns scores into probabilities summing to 1,
- guess = the top one (CD3E).

In training we know the truth is CD3E, so the loss checks the probability placed on CD3E
and nudges the weights to push it up. The `mlm_head` in `model.py`
(`Linear(d_model → vocab_size)`) is the layer that emits those 2,050 scores.

### 2c. This is BERT-style (masked), not GPT-style (next-token)

- **GPT / autoregressive:** show tokens 1…n, predict n+1. Left context only. ("n-gram →
  n+1 gram.")
- **BERT / masked:** show the whole sentence with a random 15% blanked, predict the blanks
  using context on **both sides**. ← this repo.

```
GPT-style:    GZMA CCL5 TRBC2 ____           -> next: CD3E   (left only)
BERT-style:   GZMA CCL5 TRBC2 ____ CD69 ...  -> blank: CD3E  (both sides)
```

The model always sees the whole sequence at once (up to 1,024 genes) — we don't slide
n-grams by hand. Masked fits cells because a cell's gene order is only a **ranking**, not
grammar — there's no rule that CD3E must follow TRBC2 — so using every other expressed gene
(both directions) to fill a blank is the natural choice. (The GPT/next-gene flavor is a
real alternative, listed in the README's "Next experiments.")

### 2d. What the model actually learns

Not a memorized twin cell, and not the exact same order. It learns from **thousands** of
cells **which genes keep each other's company**. Like filling "the cat sat on the ___" →
*mat* without ever seeing that exact sentence: across many T cells, whenever
`{GZMA, CCL5, TRBC2, CD69, …}` appear together, CD3E is usually in the crowd. Rank/position
matters only loosely (via `pos_emb`). Because it learned the *pattern*, it fills the blank
on a **held-out cell it never saw** — which is exactly why a probe (Stage 3) can work.

### 2e. `mask_tokens` — pick which genes to predict (`train.py`, top of `mask_tokens`)

```python
labels = tokens.clone()
prob = torch.full(tokens.shape, cfg.mlm_prob)   # 0.15 chance at every position
prob[tokens < cfg.N_SPECIAL] = 0.0              # never pick PAD/MASK
chosen = torch.bernoulli(prob).bool()           # flip a 15%-coin per position
labels[~chosen] = -100                           # positions ignored in the loss
```

```
tokens =  [GZMA, CCL5, TRBC2, CD3E, CD69, PAD, PAD]
prob   =  [0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0]
chosen =  [ F,    F,    F,    T,    F,    F,   F ]
labels =  [-100, -100, -100, CD3E, -100, -100, -100]
```

`labels` keeps the true gene only where chosen; `-100` means "don't score this."
`cross_entropy(..., ignore_index=-100)` later skips those, so the model is graded only on
the genes it was asked to guess.

### 2f. `mask_tokens` — corrupt the chosen genes 80/10/10 (`train.py`, rest of the fn)

```python
inp = tokens.clone()
to_mask = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & chosen
inp[to_mask] = cfg.MASK                      # 80% of chosen -> [MASK]
to_rand = (torch.bernoulli(torch.full(tokens.shape, 0.5)).bool()
           & chosen & ~to_mask)              # of the leftover 20%, half -> random
rand = torch.randint(cfg.N_SPECIAL, vocab_size, tokens.shape)
inp[to_rand] = rand[to_rand]                 # 10% of chosen -> a random gene
return inp, labels                           # (last 10% of chosen left untouched)
```

If 100 genes were chosen: 80 → `[MASK]`, 10 → a random gene, 10 → left as-is. For **all
100**, `labels` still records the true gene. Three fates of chosen gene CD3E:

```
80%  ->  GZMA CCL5 TRBC2 [MASK] CD69     # blanked (normal case)
10%  ->  GZMA CCL5 TRBC2 FABP1  CD69     # swapped for a random gene
10%  ->  GZMA CCL5 TRBC2 CD3E   CD69     # left as-is
```

Why not always `[MASK]`? At probe/inference time (Stages 3–4) there are **no `[MASK]`
tokens** — only real genes. The 10%-random teaches the model not to blindly trust every
gene it sees; the 10%-kept forces a right answer even when the token already looks correct.

### 2g. The training loop (`train.py:87–101`)

```python
model.train()
for epoch in range(args.epochs):                     # one epoch = one full pass over train cells
    for (batch,) in dl:                              # dl serves 64 cells at a time
        batch = batch.to(device)                     # move to GPU (mps)
        inp, labels = mask_tokens(batch, vocab_size, cfg)   # corrupt (80/10/10)
        logits = model(inp)                          # transformer -> scores for every gene
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               labels.view(-1), ignore_index=-100)   # how wrong?
        opt.zero_grad()                              # clear old gradients
        loss.backward()                              # backprop: blame each weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # cap huge updates
        opt.step()                                   # nudge every weight down-hill
        sched.step()                                 # adjust the learning rate
```

Each batch of 64 cells: mask → predict → measure wrongness → nudge weights. Fresh random
masking every time, so a cell gets different blanks each epoch. Over 5 epochs the loss
falls = the model gets better at predicting hidden genes from context = it's learning gene
co-occurrence. Result saved to `checkpoints/model.pt`.

---

### 2h. We ran it — the loss curve (1 epoch, 20k-cell data)

```
device: mps      model: 4.47M params      pretraining on 16044 cells (train)
250 steps, ~13.5 min (3.2 s/step)  ->  5 epochs ≈ ~1 hour

step   0   loss = 7.77     ← pure chance (ln 2050 = 7.63)
step  25   loss = 7.01
step  50   loss = 6.73
step  75   loss = 6.60
step 250   loss = 6.62     ← plateaued
```

Reading it:

- **7.77 at step 0** = knows nothing. A uniform guess over 2,050 genes scores `ln(2050)=7.63`.
- **~6.6 by step 75** = a real drop. Effective guessing narrowed from 1-in-2050 to ~1-in-735
  (`e^6.6`). It learned gene co-occurrence from masking alone.
- **Plateau afterward** is expected: one epoch, tiny model, and the LR schedule decays to 0
  by the end (`lr=0.0e+00`). More epochs / the full 365k cells push it lower. The 6.5–6.8
  bounce is batch-to-batch noise (different cells + random masks each step).

Output: `checkpoints/model.pt` (17.9 MB) — the frozen weights Stages 3–4 will probe.

---

---

## Stage 3 — `probe.py`

Freeze the trained model, turn each cell into one fingerprint, and test whether a simple
linear classifier can read disease from it — vs the PCA baseline, on held-out patients.

### 3a. Cell fingerprint (`model.py`, `cell_embedding`)

```python
@torch.no_grad()
def cell_embedding(self, tokens):
    h = self.encode(tokens)                          # (B, L, 256) — one vector per gene
    mask = (~tokens.eq(self.pad_id)).unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)     # average over real genes
```

The model outputs a 256-vector per gene; we average them (ignoring PAD) → one 256-number
fingerprint per cell. "Frozen" = `@torch.no_grad()`, no `.backward()`; it's now a fixed
feature extractor.

### 3b. Linear probe (`probe.py`, `report`)

```python
clf = LogisticRegression(max_iter=2000, C=1.0)   # a deliberately SIMPLE classifier
clf.fit(Xtr, ytr)                                # train on train-patient cells
pred = clf.predict(Xva)                          # score on held-out val patients
f1 = f1_score(yva, pred, average="macro")        # per-class F1, averaged equally
```

We keep the classifier weak on purpose: if even a straight-line boundary reads disease from
the fingerprints, the fingerprints themselves are well-organized. macro-F1 averages the
three classes equally so the model can't win by predicting the biggest class.

### 3c. Bug we hit + fix

`LogisticRegression(..., multi_class="auto")` crashed: scikit-learn **1.9.0** removed the
`multi_class` argument (multiclass is automatic now). Fix = delete the argument.
**Lesson:** library drift, not a model bug — pinned versions in `requirements.txt` prevent
this.

### 3d. Result (1-epoch model, 20k cells)  →  `probe_result.png`

```
PCA-50 baseline         acc = 0.753   macro-F1 = 0.739
nano-Geneformer embed   acc = 0.609   macro-F1 = 0.607
chance (3 classes)      ≈ 0.333
```

**PCA won — and that's the README's predicted outcome, not a failure.** Both beat chance,
so both carry disease signal; but PCA is clearly better. Why:

1. **Rank encoding discards magnitude, and inflammation lives in magnitude** (fold-changes).
   PCA keeps it; our tokens keep only gene *ordering*. We handed PCA the best signal.
2. **The model is barely trained** — 1 epoch, 20k cells, loss still 6.6. Real Geneformer
   sees millions of cells for many epochs.
3. **Still above chance**, so pretraining did learn disease-relevant structure — just not
   enough to beat a strong classical baseline yet.

Next things that would likely help the embedding: more epochs, the full 365k cells, and/or
a value-aware tokenization that keeps some magnitude (README "Next experiments").

---

---

## Experiment — does more data close the gap to PCA?

Test: prepare a **100k-cell** subsample, train **2 epochs**, re-probe. Everything else held
identical to the 20k/1-epoch run (batch 64, same LR), so only *data size + epochs* changed.
Runtime ≈ 3 h (train ~2h55m for 2 epochs, ~3.5 s/step). Plot: `probe_result_100k.png`.

| | 20k · 1 epoch | 100k · 2 epochs | change |
|---|---|---|---|
| Training loss | 6.6 | **5.03** | −24% |
| Embedding macro-F1 | 0.607 | **0.643** | +0.036 |
| Embedding accuracy | 0.609 | 0.646 | +0.037 |
| PCA-50 macro-F1 | 0.739 | 0.748 | ~flat |
| **Gap (PCA − embed)** | 0.132 | **0.105** | narrowed ~20% |

**Verdict: more data helped the embedding but did not overtake PCA.** The gap narrowed
modestly.

**The key diagnostic — a disproportion:** loss fell 24% but disease-reading rose only
+0.036. If *undertraining* were the bottleneck, both would jump together; they didn't. The
model got much better at its *training task* (masked-gene prediction) while barely improving
at *separating disease*. That isolates the bottleneck as the **tokenization ceiling** — rank
encoding discarded the expression magnitude that inflammation lives in, and more data can't
put it back. Visible in the plot: the 100k embedding forms real defined clusters (vs the 20k
"arch"), but disease colors still smear across them.

**Implication for next steps:** scaling data/compute gives diminishing returns *on this
task*. The higher-leverage move is to **raise the ceiling** — a value-aware tokenization that
keeps some magnitude (bin expression instead of pure rank), per the README's "Next
experiments." More scale moves you *toward* the ceiling; value-aware tokens move the ceiling.

*Artifacts:* `probe_result_20k.png` (baseline), `probe_result_100k.png` (this run). The
`data/` + `checkpoints/model.pt` on disk are now the **100k/2-epoch** versions.

---

*(continued as we go)*
