"""Central config for nano-Geneformer (IBD scRNA-seq).

Edit the OBS COLUMN NAMES to match YOUR AnnData file. If prepare.py can't find
a column, it prints the columns your file actually has so you can fix these.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- data ----------------------------------------------------------------
    # Path to your scRNA-seq data. Override on the CLI: `python prepare.py --data ...`
    # Accepts: .h5ad | a 10x mtx directory | .h5 (10x) | .loom | .csv/.tsv
    data_path: str = "smillie"     # Smillie SCP259 folder (compartment layout auto-detected)
    out_dir: str = "data"          # prepared tensors + vocab land here

    # obs column names in YOUR file. Defaults follow the Smillie et al. 2019 UC
    # atlas naming; EDIT to match what you actually have.
    patient_col: str = "Subject"   # one value per patient/donor (used for the split)
    label_col: str = "Health"      # e.g. Healthy / Non-inflamed / Inflamed
    celltype_col: str = "Cluster"  # cell-type annotation (used only for probing/plots)

    # ---- tokenization (rank-value encoding, Geneformer-style) -----------------
    n_hvg: int = 2048              # keep top-N highly-variable genes -> vocab size
    max_len: int = 1024            # max ranked genes per cell (sequence length)
    min_genes_per_cell: int = 200  # QC filter
    min_cells_per_gene: int = 10   # QC filter

    # Geneformer-style adaptive QC (Genecorpus-30M criteria): keep cells within
    # +-qc_nsd SD of the per-"dataset" mean on BOTH total counts and mito %.
    # Tissue-aware by construction: each sample's own distribution sets the bar,
    # so intrinsically high-mito colon epithelium survives while the dying tail
    # (extreme mito %) is removed. See geneformer_qc_impact.py.
    mito_qc: bool = True
    qc_nsd: float = 3.0
    mito_prefix: str = "MT-"
    qc_group_col: str = "Sample"   # per-"dataset" unit; falls back to whole set if absent

    # patient-level split: every Nth patient (sorted) goes to validation.
    # Keeps whole patients out of training -> honest probing. See README caveats.
    val_every: int = 5             # ~20% of patients held out

    # ---- model ---------------------------------------------------------------
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # ---- training ------------------------------------------------------------
    batch_size: int = 64
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    mlm_prob: float = 0.15
    pretrain_on: str = "train"     # "train" (held-out val stays clean) or "all"
    seed: int = 0

    # special token ids (genes start at N_SPECIAL)
    PAD: int = 0
    MASK: int = 1
    N_SPECIAL: int = 2

    def to_dict(self):
        return asdict(self)


CONFIG = Config()
