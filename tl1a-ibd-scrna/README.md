# TL1A axis in IBD — single-cell expression analysis

Where the **TL1A–DR3 axis** acts in the gut at single-cell resolution, and how it splits into
an **inflammation arm** (DR3 on Th17/ILC3) and a **fibrosis arm** (DR3 on fibroblasts, incl.
the anti-TNF-resistant IAF program). Companion to the genetics work in the TL1A target dossier
(`tensoromics/tensortarget → docs/TL1A_target_dossier.md`), which established that the IBD risk
variant raises TL1A specifically in **myeloid** cells. This folder takes the **expression** side.

- **TL1A** = `TNFSF15` (ligand; source = myeloid) · **DR3** = `TNFRSF25` (receptor)
- **IAF** signature = `IL13RA2, IL11, OSMR, TNFRSF11B, CHI3L1, IL24` (inflammation-associated fibroblast / anti-TNF resistance)

## Setup (Python 3.12 — scanpy's stack has no 3.14 wheels)
```bash
cd tl1a-ibd-scrna
uv venv --python 3.12 .venv                 # or: python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt             # or: uv pip install -r requirements.txt
```

## Get data (public cellxgene atlases, resumable; ~3 GB for the light path)
```bash
bash scripts/download_data.sh primary       # UC+CD Gut Cell Atlas compartments (mesenchymal, myeloid, T, endothelial)
bash scripts/download_data.sh xcheck        # 264 MB CD terminal-ileum stromal cross-check
```
Full provenance + accessions + sizes: [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md).

## Run
```bash
python scripts/analyze_tl1a_expression.py --input data/gca_mesenchymal_77k.h5ad   # DR3 / IAF (fibrosis arm)
python scripts/analyze_tl1a_expression.py --input data/gca_myeloid_52k.h5ad       # TNFSF15 source
python scripts/analyze_tl1a_expression.py --input data/gca_t_nk_263k.h5ad         # DR3 inflammation arm
```
Each run writes a per-(cell_type × disease) table to `results/` and a dotplot to `figures/`,
and scores the IAF signature where fibroblasts are present.

## What to look for (the open questions, from the dossier §5.2)
1. Does the **myeloid-source / Th17-ILC3-receptor** pattern hold in **UC** (the indication with the positive trial)?
2. Use the richer UC stromal sampling to test the **fibrosis arm** — DR3 on inflammatory/fibrotic fibroblasts.
3. Compare **UC vs CD** (the atlas has both in one harmonized annotation space) and **inflamed vs normal**.

## Layout
```
tl1a-ibd-scrna/
  README.md
  requirements.txt          # scanpy stack (py3.12)
  scripts/
    download_data.sh        # cellxgene UC/CD atlas downloader (tiered, resumable)
    analyze_tl1a_expression.py
  data/
    DATA_SOURCES.md         # accessions, URLs, sizes, gene lists
    *.h5ad                  # downloaded (gitignored)
  figures/  results/        # generated (gitignored)
```
Data, `.venv`, and outputs are gitignored — recreate with the steps above on any machine.
