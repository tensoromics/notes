# Data sources — TL1A / DR3 in IBD single-cell analysis

Question: where the TL1A–DR3 axis acts in the gut, and why anti-TL1A may work where
anti-TNF fails — with the inflammation arm (DR3 on T/ILC) vs the fibrosis arm
(DR3 on fibroblasts, incl. the anti-TNF-resistant IAF program) resolved by cell type.

Genes: TL1A = `TNFSF15` (ENSG00000181634) · DR3 = `TNFRSF25` (ENSG00000215788).
IAF (inflammation-associated fibroblast) signature: `IL13RA2, IL11, OSMR, TNFRSF11B, CHI3L1, IL24`.

## Primary — UC + CD in one harmonized annotation space
**Gut Cell Atlas, "Single-cell integration reveals metaplasia in inflammatory gut diseases"** (Nature).
cellxgene collection. UC + CD + normal, compartment-split, standardized obs (cell_type, disease, tissue).
Note: harmonized integration — IAFs identified by signature scoring, not a prebuilt label.
Caveat: mucosal biopsies under-sample deep (submucosal/muscularis) fibrosis.

| File | Cells | Role |
|---|---|---|
| `gca_mesenchymal_77k.h5ad` | 77,050 | DR3 fibrosis arm + IAF program |
| `gca_myeloid_52k.h5ad` | 52,404 | TL1A (`TNFSF15`) source |
| `gca_endothelial_60k.h5ad` | 60,411 | secondary TL1A source |
| `gca_t_nk_263k.h5ad` | 262,642 | DR3 inflammation arm |

## CD cross-check (light) — terminal ileum stromal
**"The landscape of immune dysregulation in Crohn's disease"** (cellxgene). `cd_ti_stromal_75k.h5ad`, 75,695 cells, 264 MB.

## CD cross-check (premium) — IBDverse
**IBDverse** — terminal ileal biopsies, 1.1M cells, 111 CD + 232 HC.
- Expression: ArrayExpress `E-MTAB-16999` → `Full_cohort.h5ad` (25.5 GB, TI-only, CD vs healthy). 24 GB RAM: use `backed='r'` + chunked streaming, do not load whole.
- cellxgene mirror: 32 GB (gut + blood) — `10.1038/s41586-026-10627-z`.
- **Cell-type eQTLs + DE:** BioStudies `S-BSST2944` (`DGE`). The IBDverse value-add: test whether the `TNFSF15` risk variant raises TL1A in a *specific* cell type (genetics-first, cell-type-resolved).
- Nat Genet CD-signatures paper: `10.1038/s41588-026-02634-7`. Raw reads: EGA `EGAD00001015692` (restricted). Code: github.com/andersonlab/sc_ti_atlas. Portal: https://www.ibdverse.info/

## Clinical anchor (citations, not data)
- Tulisokibart (anti-TL1A) positive Phase 2 in UC **with a companion genetic responder test** — NEJM 2024 (ARTEMIS-UC).
- Anti-TNF non-responders enriched for IL13RA2⁺/IL11⁺ IAFs — Smillie et al., Cell 2019.
- TL1A–DR3 signals directly on fibroblasts → fibrosis in vivo; anti-TL1A reverses it (↓CTGF/TGFβ1) — Sci Rep 2020.
