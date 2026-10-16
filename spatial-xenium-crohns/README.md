# Spatial transcriptomics of Crohn's disease (10x Xenium)

Companion code for the TensorOmics series on spatial transcriptomics of Crohn's disease. The first note is
[*Spatial transcriptomics of Crohn's disease: zooming into an ulcer*](https://tensoromics.com/notes/spatial-transcriptomics-crohns-xenium/).
It takes a first look at the
10x Xenium Prime 5K data from Koplev *et al.*,
[*Nature Immunology* 2026](https://doi.org/10.1038/s41590-026-02617-0): 2.1 million cells from
16 sections of Crohn's ileum, an inflamed and an adjacent piece from each of 8 patients.

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

## Run
The notebook runs the note's code blocks in order (about 10–15 seconds in total once the file is cached; the 16-section map is the slowest step):
```bash
jupyter lab spatial-transcriptomics-crohns-xenium.ipynb
```

The finished figures come from the scripts (they write to `figures/`):

| Figures in the note | Script |
|---|---|
| 1, 2 | `python scripts/pipeline_fig.py` (made-up dots, no data needed) |
| 3 | `python scripts/crohn_summary.py data/xenium5k_scVI_CellCharter_batch2_reannot_SCP.h5ad` |
| 4–7, 9 | `spatial-transcriptomics-crohns-xenium.ipynb` |
| 8 | `python scripts/extract_crohn_obs.py data/xenium5k_...SCP.h5ad data/crohn_obs.csv.gz`, then `python scripts/crohn_zoom.py data/crohn_obs.csv.gz` |

`scripts/choose_ulcer_patch.py data/crohn_obs.csv.gz` shows how the zoom targets were chosen: the
150 µm patch with the most COL7A1hi inflammatory fibroblasts, and a 1 mm square around it.

## Layout
```
spatial-xenium-crohns/
  README.md
  requirements.txt
  spatial-transcriptomics-crohns-xenium.ipynb          # the note's code, in order
  scripts/
    pipeline_fig.py         # figures 1-2 (schematic)
    crohn_summary.py        # figure 3
    extract_crohn_obs.py    # h5ad -> compact cell table (crohn_obs.csv.gz), matrix never loaded
    choose_ulcer_patch.py   # how the zoom square and patch were chosen
    crohn_zoom.py           # figure 8
  data/
    DATA_SOURCES.md
  figures/                  # generated, not tracked
```
