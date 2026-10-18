"""Steps 2 and 3: process both arrays, compute scores, run the sanity checks.

No response labels are used here.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

from common import gene_matrix, probe_to_symbol, read_series_matrix

rng = np.random.default_rng(20261017)
signatures = json.load(open("results/signatures.json"))
SCORES = ["IF", "IF_strict", "IF_loose", "pan_fibroblast", "inflammatory_myeloid"]
probe_map = probe_to_symbol()


def load(gse):
    expr, samples = read_series_matrix(f"data/{gse}_series_matrix.txt.gz")
    print(f"\n{gse}: {expr.shape[0]} probes x {expr.shape[1]} samples; "
          f"values {expr.values.min():.2f} to {expr.values.max():.2f}, median {np.median(expr.values):.2f}")
    genes = gene_matrix(expr, probe_map)
    print(f"{genes.shape[0]} genes after mapping probes to HGNC symbols")
    return genes, samples


def score(genes, reference, gene_list):
    """Mean z-score of the genes, with z taken against the reference samples."""
    g = [x for x in gene_list if x in genes.index]
    ref = genes.loc[g, reference]
    z = genes.loc[g].sub(ref.mean(axis=1), axis=0).div(ref.std(axis=1), axis=0)
    return z.mean(axis=0), g


def rank_score(genes, gene_list):
    pct = genes.rank(axis=0, pct=True)
    return pct.loc[[x for x in gene_list if x in genes.index]].mean(axis=0)


def coherence(genes, reference, gene_list, n_random=1000):
    """Mean pairwise correlation of the genes, against random sets matched on mean expression."""
    sub = genes[reference]
    g = [x for x in gene_list if x in sub.index]

    def mean_r(idx):
        c = np.corrcoef(sub.loc[idx].values)
        return c[np.triu_indices_from(c, 1)].mean()

    observed = mean_r(g)
    bins = pd.qcut(sub.mean(axis=1), 20, labels=False)
    pools = [bins.index[bins == bins[x]].tolist() for x in g]
    null = [mean_r([rng.choice(p) for p in pools]) for _ in range(n_random)]
    return observed, float(np.mean(np.array(null) >= observed))


# ---------- ileum (GSE112366)
ileum, s_il = load("GSE112366")
s_il = s_il.rename(columns={"treatment_induction": "arm", "i-wk8 response": "response",
                            "gi involvement": "location", "ileum ses-cd score": "ses_cd"})
s_il["cd"] = s_il["diagnosis"].str.startswith("Crohn")
base_il = s_il.index[s_il["cd"] & (s_il["visit"] == "I-WK0")]
for name in SCORES:
    s_il[name], used = score(ileum, base_il, signatures[name])
    print(f"  {name}: {len(used)}/{len(signatures[name])} genes on array")
s_il["IF_rank"] = rank_score(ileum, signatures["IF"])

# ---------- rectum (GSE207022)
rectum, s_re = load("GSE207022")
s_re = s_re.rename(columns={"donor id": "subject", "mucosal healing at week 8": "healing",
                            "inflamed area at week 0": "location"})
s_re["cd"] = s_re["diagnosis"].str.startswith("Crohn")
base_re = s_re.index[s_re["cd"]]
for name in SCORES:
    s_re[name], _ = score(rectum, base_re, signatures[name])
s_re["IF_rank"] = rank_score(rectum, signatures["IF"])

# ---------- QC: sex genes must agree across a patient's visits
sex = ileum.loc[["XIST", "RPS4Y1"]].T
s_il["male"] = (sex["RPS4Y1"] - sex["XIST"]) > 0
mixed = s_il[s_il["cd"]].groupby("subject")["male"].nunique()
print(f"\nQC: {int((mixed > 1).sum())} patients whose samples disagree on sex genes "
      f"(of {len(mixed)}); male call rate {s_il['male'].mean():.2f}")
print("sex-gene separation (RPS4Y1 - XIST):",
      np.round(np.percentile(sex["RPS4Y1"] - sex["XIST"], [5, 25, 50, 75, 95]), 2))

# ---------- Step 3 sanity checks
print("\n=== Step 3 sanity checks (ileum) ===")
b = s_il.loc[base_il].copy()
ctrl = s_il[~s_il["cd"]]
for name in ["IF", "pan_fibroblast", "inflammatory_myeloid"]:
    u = stats.mannwhitneyu(b[name], ctrl[name])
    print(f"{name:22s} CD baseline median {b[name].median():+.2f} vs control {ctrl[name].median():+.2f}  "
          f"(Mann-Whitney p = {u.pvalue:.1e})")

ileal = b["location"].isin(["Ileum only", "Ileum and colon"])
colon_only = b["location"] == "Colon only"
u = stats.mannwhitneyu(b.loc[ileal, "IF"], b.loc[colon_only, "IF"])
print(f"IF ileal involvement median {b.loc[ileal, 'IF'].median():+.2f} vs colon-only "
      f"{b.loc[colon_only, 'IF'].median():+.2f}  (p = {u.pvalue:.1e})")

has_ses = b["ses_cd"] != "."
ses = b.loc[has_ses, "ses_cd"].astype(int)
for name in ["IF", "pan_fibroblast", "inflammatory_myeloid"]:
    r = stats.spearmanr(ses, b.loc[has_ses, name])
    print(f"{name:22s} vs ileal SES-CD: Spearman rho {r.statistic:+.2f} (p = {r.pvalue:.1e}, n = {has_ses.sum()})")
r = stats.spearmanr(b["IF"], b["inflammatory_myeloid"])
print(f"IF vs monocyte-macrophage score: rho {r.statistic:+.2f}")
r = stats.spearmanr(b["IF"], b["pan_fibroblast"])
print(f"IF vs pan-fibroblast score: rho {r.statistic:+.2f}")

for name in ["IF", "IF_loose", "pan_fibroblast", "inflammatory_myeloid"]:
    obs, p = coherence(ileum, base_il, signatures[name])
    print(f"coherence {name:22s} mean pairwise r {obs:.2f}; random sets as high: {p:.3f}")

print("\nper-gene correlation with the IF score (baseline ileum):")
for g in signatures["IF"]:
    print(f"  {g:8s} r = {np.corrcoef(ileum.loc[g, base_il], b['IF'])[0, 1]:.2f}   "
          f"mean log2 {ileum.loc[g, base_il].mean():.1f}")

print("\n=== rectum ===")
br = s_re.loc[base_re]
u = stats.mannwhitneyu(br["IF"], s_re.loc[~s_re["cd"], "IF"])
print(f"IF CD median {br['IF'].median():+.2f} vs control {s_re.loc[~s_re['cd'], 'IF'].median():+.2f} (p = {u.pvalue:.1e})")
obs, p = coherence(rectum, base_re, signatures["IF"])
print(f"coherence IF mean pairwise r {obs:.2f}; random as high: {p:.3f}")

s_il.to_parquet("results/ileum_samples_scored.parquet")
s_re.to_parquet("results/rectum_samples_scored.parquet")
ileum.to_parquet("results/ileum_genes.parquet")
rectum.to_parquet("results/rectum_genes.parquet")
