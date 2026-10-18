# Inflammatory fibroblasts and response to Crohn's drugs

Analysis code for a TensorOmics note in progress. The question: inflammatory fibroblasts build up in
Crohn's ulcers and, in culture, stay inflamed when the cytokines that made them are blocked. Do
patients with more of them respond less to ustekinumab or infliximab, or does the signal only track
how damaged the gut already is?

An 8-gene signature for these cells is built from the Koplev et al. Xenium atlas
([Nature Immunology, 2026](https://doi.org/10.1038/s41590-026-02617-0)) without using any patient
outcomes, then tested in biopsies taken before treatment.

The note is not published yet. This folder holds the analysis plan and the scripts. Data and results
are not included.

## The plan

`PLAN.md` was written before any response data were used. Each later extension was added to it
before its own outcomes were used, and the plan records what was already known at each point.
Extensions 3 (vedolizumab, GSE73661) and 4 (IBDome) are committed here before either has been run,
so the commit history shows that those plans came first.

## Setup (Python 3.10 or later)

```bash
cd ustekinumab-fibroblasts
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the data listed in [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) into `data/`.

## Scripts

Run them in order from this folder. Each writes to `results/`.

| Script | What it does |
|---|---|
| `common.py` | Loaders for GEO series matrices, probe annotations and HGNC symbols |
| `01_atlas_pseudobulk.py` | Sums Xenium counts by cell type and tissue section |
| `02_atlas_signatures.py` | Chooses the signature genes (rule in `PLAN.md`, Amendment 1) |
| `03_array_scores.py` | Scores the UNITI biopsies and runs the checks before outcomes |
| `04_primary.py` | Primary test: week-0 score vs week-8 symptom response on ustekinumab |
| `05_secondary.py` | Placebo arm, change during treatment, week 44, rectal biopsies, sensitivity analyses |
| `06_infliximab.py` | Extension 1: infliximab (GSE16879) |
| `07_infliximab_checks.py` | Extension 1b: inflammation, fibroblast and normalization checks |
| `08_uniti_endoscopy.py` | Extension 2: endoscopic response on ustekinumab |
| `09_published_markers.py` | Where published inflammatory-fibroblast markers sit in the atlas |

## License

MIT, see `LICENSE` at the repo root.
