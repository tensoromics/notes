# Data sources

| What | Where | Size | Access |
|---|---|---|---|
| Crohn's ileum, 10x Xenium Prime 5K, 16 sections from 8 patients (Koplev *et al.*, *Nature Immunology* 2026, [doi:10.1038/s41590-026-02617-0](https://doi.org/10.1038/s41590-026-02617-0)) | [Broad Single Cell Portal, SCP3259](https://singlecell.broadinstitute.org/single_cell/study/SCP3259), file `xenium5k_scVI_CellCharter_batch2_reannot_SCP.h5ad` | about 9 GB | free, Google sign-in |

What the file holds: 2,131,098 cells × 5,001 genes; raw counts in `layers/counts`; a per-cell table
(`obs`) with centroids, cell and nucleus areas, transcript and control-probe counts, the segmentation
method, the authors' cell-type labels (`megagut_lvl1_majority_voting`, `megagut_lvl3_majority_voting`,
`annot_new`) and CellCharter niches (`spatial_cluster15`; niche 11 = mucosal ulcer in the paper).

Not shared by the authors: the per-transcript file (`transcripts.parquet`), the cell and nucleus
outlines (`cell_boundaries.parquet`, `nucleus_boundaries.parquet`) and the morphology images.

Note: `donor_id` has 7 values; 0060032 holds two patients, told apart by specimen (`gi9662`, `gi10056`).

Caution on `annot_new`: one of its labels, "transit amplifying epithelium", covers 691,764 cells (about a
third of the atlas), and most of them are not epithelial in the broad labels. The note uses `annot_new`
only for "M COL7A1hi inflammatory fibroblast" (6,325 cells, 80% Mesenchymal in the broad labels).
