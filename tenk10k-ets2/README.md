# TenK10K meets ETS2

Figure code for the TensorOmics note
[How much of the ETS2 story can a chromatin atlas recover?](https://tensoromics.com/notes/tenk10k-chromatin-atlas-ets2-ibd/),
on Xue et al.,
[Genetic variants shape single-cell chromatin accessibility and disease risk](https://www.nature.com/articles/s41586-026-11173-4)
(Nature, 2026). The note compares the TenK10K chromatin atlas with what Stankey et al.
([Nature, 2024](https://www.nature.com/articles/s41586-024-07501-1)) showed at the ETS2 locus in IBD.

Every number comes from public summary statistics or published papers. The per-donor TenK10K data are under
controlled access, so no figure shows individual donors.

## Setup (Python 3.10 or later)

```bash
cd tenk10k-ets2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild the figures

```bash
./make_figures.sh            # writes figures/fig1-...png to figures/fig7-...png
```

Figure 1 reads `data/ets2_peak_cd14_cis_pairs.csv`. To rebuild that file from the source:

```bash
python extract_ets2_peak.py data/ets2_peak_cd14_cis_pairs.csv
```

It reads only the parts of a 940 MB parquet file on Hugging Face that hold the ETS2 peak.

## Scripts

| Script | Figure | Inputs |
|---|---|---|
| `extract_ets2_peak.py` | (data) | TenK10K `caQTL_common_TensorQTL/CD14_Mono/TenK10K.cis_qtl_pairs.chr21.parquet` |
| `fig1_caqtl_example.py` | What a caQTL looks like: regional plot of all 7,356 variants tested against the ETS2 enhancer peak in CD14 monocytes, and a model-based boxplot at rs2836882 | `data/ets2_peak_cd14_cis_pairs.csv` |
| `fig2_ets2_locus.py` | The chr21q22 locus: ETS2, the enhancer, the Hi-C contact and the gene desert | Ensembl GRCh38 coordinates (in the script) |
| `fig3_smr.py` | How SMR divides the variant's effect on IBD by its effect on openness | TenK10K SMR file; IBD GWAS values match Liu et al. 2023 (values in the script) |
| `fig4_search_windows.py` | The ±100 kb and ±1 Mb search windows around ETS2 | Ensembl GRCh38 coordinates; Cuomo et al. and Xue et al. Methods |
| `fig5_caqtl_peak.py` | The ETS2 enhancer peak across the 26 cell types TenK10K tested | TenK10K caQTL lead table and SMR files (values in the script) |
| `fig6_direction_chain.py` | The direction of each step from the risk allele to IBD risk | Stankey et al. 2024, TenK10K SMR file and Liu et al. 2023 (values in the script) |
| `fig7_eqtl_distance.py` | How far IBDverse eQTLs are from their gene, by resolution | Alegbe et al. 2026 (values in the script) |

Each script's docstring lists its sources and how each value was derived.

## The model-based boxplot (figure 1, right)

TenK10K posts per-allele effects, standard errors and carrier counts. The boxplot is built from them:

- Genotype counts: N = ma_count / (2 × af); minor homozygotes = ma_count − ma_samples; heterozygotes =
  ma_samples − minor homozygotes. For rs2836882 in CD14 monocytes: AA 68, AG 355, GG 499.
- Group medians: k × β for k copies of G, with β = 0.952709 per G allele.
- Spread: the residual SD comes from the slope's standard error, σ = SE × √(N × var(k)) ≈ 0.49. This ignores the
  covariate adjustment of genotype, so it is approximate. It is the spread after the model's covariates are removed;
  each peak is standardized across donors, so the unadjusted within-genotype SD would be about √(1 − 0.36) ≈ 0.8.
- Boxes show median ± 0.6745σ (middle 50%) and whiskers median ± 1.6449σ (middle 90%).

## Data and licenses

- TenK10K multiome summary statistics: [Hugging Face, anglixue/TenK10K_multiome](https://huggingface.co/datasets/anglixue/TenK10K_multiome),
  CC BY 4.0. `data/ets2_peak_cd14_cis_pairs.csv` is an extract of that dataset (Xue et al., Nature 2026).
- Code: MIT, see `LICENSE` at the repo root.
