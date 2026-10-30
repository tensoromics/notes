# Spatial transcriptomics of Crohn's disease (10x Xenium)

Companion code for the TensorOmics series on spatial transcriptomics of Crohn's disease. The series
uses the 10x Xenium Prime 5K data from Koplev *et al.*,
[*Nature Immunology* 2026](https://doi.org/10.1038/s41590-026-02617-0): 2.1 million cells from
16 sections of Crohn's ileum, an inflamed and an adjacent piece from each of 8 patients.

1. [*A first look*](https://tensoromics.com/notes/spatial-transcriptomics-crohns-xenium/): load the
   atlas in Python and zoom from whole sections into an ulcer.
2. [*Cell segmentation errors in Xenium data*](https://tensoromics.com/notes/spatial-transcriptomics-crohns-cell-overlap/):
   how close the cells are packed, and whether genes from one cell type show up in the cells next to it.

## Setup (Python 3.9 or later)
```bash
cd spatial-xenium-crohns
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data
One AnnData file, `xenium5k_scVI_CellCharter_batch2_reannot_SCP.h5ad` (about 9 GB), from the
[Broad Single Cell Portal, study SCP3259](https://singlecell.broadinstitute.org/single_cell/study/SCP3259).
The download is free but needs a Google sign-in, and the file cannot be redistributed, so it is not
in this repo. Put it in `data/`. Details: [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md).

Part 2, Figure 5 also uses the public
[Xenium Prime FFPE Human Skin](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-skin)
sample from 10x Genomics: `Xenium_Prime_Human_Skin_FFPE_cells.parquet`,
`Xenium_Prime_Human_Skin_FFPE_cell_boundaries.parquet`, and the `morphology_focus/` folder from
`Xenium_Prime_Human_Skin_FFPE_xe_outs.zip` (3.1 GB). Put them in `data/Xenium_Prime_Human_Skin_FFPE/`.

## Run
Each note has a notebook that runs its code blocks in order:
```bash
jupyter lab spatial-transcriptomics-crohns-xenium.ipynb          # part 1, about 10-15 seconds once the file is cached
jupyter lab spatial-transcriptomics-crohns-cell-overlap.ipynb    # part 2, under a minute
```

The finished figures come from the scripts (they write to `figures/`).

**Part 1**

| Figures in the note | Script |
|---|---|
| 1, 2 | `python scripts/pipeline_fig.py` (made-up dots, no data needed) |
| 3 | `python scripts/crohn_summary.py data/xenium5k_scVI_CellCharter_batch2_reannot_SCP.h5ad` |
| 4–7, 9 | `spatial-transcriptomics-crohns-xenium.ipynb` |
| 8 | `python scripts/extract_crohn_obs.py data/xenium5k_...SCP.h5ad data/crohn_obs.csv.gz`, then `python scripts/crohn_zoom.py data/crohn_obs.csv.gz` |

`scripts/choose_ulcer_patch.py data/crohn_obs.csv.gz` shows how the zoom targets were chosen: the
150 µm patch with the most COL7A1hi inflammatory fibroblasts, and a 1 mm square around it.

**Part 2** (needs matplotlib 3.10 or later; Figure 5 also needs `pyarrow`, `shapely`, `tifffile`, and `zarr`)

| Figures in the note | Script |
|---|---|
| 1, 3, 7, 10 | `python scripts/p2_schematic.py`, `p2_method_schematic.py`, `p2_test_schematic.py`, `p2_de_schematic.py` (made-up cells, no data needed) |
| 2 | `python scripts/p2_matrix_fig.py data/xenium5k_scVI_CellCharter_batch2_reannot_SCP.h5ad` |
| 4, 6, 8, 9, 11 | `spatial-transcriptomics-crohns-cell-overlap.ipynb` |
| 5 | `python scripts/p2_stain_fig.py data` (reads `data/Xenium_Prime_Human_Skin_FFPE/`) |

## Layout
```
spatial-xenium-crohns/
  README.md
  requirements.txt
  spatial-transcriptomics-crohns-xenium.ipynb          # part 1's code, in order
  spatial-transcriptomics-crohns-cell-overlap.ipynb    # part 2's code, in order
  scripts/
    pipeline_fig.py         # figures 1-2 (schematic)
    crohn_summary.py        # figure 3
    extract_crohn_obs.py    # h5ad -> compact cell table (crohn_obs.csv.gz), matrix never loaded
    choose_ulcer_patch.py   # how the zoom square and patch were chosen
    crohn_zoom.py           # figure 8
    p2_schematic.py         # part 2: figure 1
    p2_matrix_fig.py        # part 2: figure 2
    p2_method_schematic.py  # part 2: figure 3
    p2_stain_fig.py         # part 2: figure 5 (skin sample)
    p2_test_schematic.py    # part 2: figure 7
    p2_de_schematic.py      # part 2: figure 10
  data/
    DATA_SOURCES.md
  figures/                  # generated, not tracked
```
