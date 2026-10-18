"""Extension 1b in PLAN.md: inflammation, fibroblast and normalization checks on the infliximab result."""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import gene_matrix, probe_to_symbol, read_series_matrix

sig = json.load(open("results/signatures.json"))
expr, _ = read_series_matrix("data/GSE16879_series_matrix.txt.gz")
log2 = np.log2(expr.clip(lower=0) + 1)
s = pd.read_parquet("results/infliximab_samples_scored.parquet")
probe_map = probe_to_symbol("GPL570")


def quantile_normalize(m):
    ranks = m.rank(axis=0, method="first").astype(int) - 1
    mean_sorted = np.sort(m.values, axis=0).mean(axis=1)
    return pd.DataFrame(mean_sorted[ranks.values], index=m.index, columns=m.columns)


def scorer(genes):
    base = s[s.time == "Before"]

    def score(gene_list):
        out = pd.Series(index=s.index, dtype=float)
        for tissue in ["Colon", "Ileum"]:
            ref = base.index[base.tissue == tissue]
            idx = s.index[s.tissue == tissue]
            sub = genes.loc[[g for g in gene_list if g in genes.index]]
            z = sub[idx].sub(sub[ref].mean(axis=1), axis=0).div(sub[ref].std(axis=1), axis=0)
            out[idx] = z.mean(axis=0)
        return out
    return score


def per_sd(x):
    return (x - x.mean()) / x.std()


def or_line(label, fit, term):
    b, (lo, hi), p = fit.params[term], fit.conf_int().loc[term], fit.pvalues[term]
    print(f"{label:58s} {term:8s} OR per SD {np.exp(b):.2f} ({np.exp(lo):.2f} to {np.exp(hi):.2f}), p = {p:.3f}")


genes = gene_matrix(log2, probe_map)
score = scorer(genes)
s["MY"] = score(sig["inflammatory_myeloid"])
s["FIB"] = score(sig["pan_fibroblast"])
base = s[s.time == "Before"].copy()
print("genes present: myeloid", sum(g in genes.index for g in sig["inflammatory_myeloid"]), "of 25;",
      "pan-fibroblast", sum(g in genes.index for g in sig["pan_fibroblast"]), "of 4")
print(f"baseline correlation, IF vs monocyte-macrophage: rho {stats.spearmanr(base.IF, base.MY).statistic:+.2f}; "
      f"IF vs pan-fibroblast: rho {stats.spearmanr(base.IF, base.FIB).statistic:+.2f}")

for label, d in [("Crohn's", base[base.disease == "CD"].copy()), ("UC", base[base.disease == "UC"].copy())]:
    print(f"\n=== {label} (n = {len(d)}, {d.resp.sum()} responders) ===")
    d["IFz"], d["MYz"], d["FIBz"] = per_sd(d.IF), per_sd(d.MY), per_sd(d.FIB)
    d["ileum"] = (d.tissue == "Ileum").astype(int)
    tissue = " + ileum" if label == "Crohn's" else ""
    print(f"AUC for non-response: IF {roc_auc_score(1 - d.resp, d.IF):.2f}, monocyte-macrophage "
          f"{roc_auc_score(1 - d.resp, d.MY):.2f}, pan-fibroblast {roc_auc_score(1 - d.resp, d.FIB):.2f}")
    or_line("monocyte-macrophage alone", smf.logit(f"resp ~ MYz{tissue}", data=d).fit(disp=0), "MYz")
    f = smf.logit(f"resp ~ IFz + MYz{tissue}", data=d).fit(disp=0)
    or_line("IF + monocyte-macrophage", f, "IFz")
    or_line("IF + monocyte-macrophage", f, "MYz")
    f = smf.logit(f"resp ~ IFz + FIBz{tissue}", data=d).fit(disp=0)
    or_line("IF + pan-fibroblast", f, "IFz")
    or_line("IF + pan-fibroblast", f, "FIBz")

print("\n=== Crohn's colon and ileum (Mann-Whitney, AUC) ===")
for grp in ["CD colon", "CD ileum"]:
    d = base[base.group == grp]
    for name, col in [("IF", "IF"), ("monocyte-macrophage", "MY")]:
        u = stats.mannwhitneyu(d[col][d.resp == 0], d[col][d.resp == 1], alternative="two-sided")
        print(f"{grp:9s} {name:20s} AUC for non-response {roc_auc_score(1 - d.resp, d[col]):.2f}, Mann-Whitney p = {u.pvalue:.4f}")

print("\n=== Quantile-normalized arrays ===")
qgenes = gene_matrix(quantile_normalize(log2), probe_map)
qscore = scorer(qgenes)
s["IFq"] = qscore(sig["IF"])
d = s[(s.time == "Before") & (s.disease == "CD")].copy()
d["x"] = per_sd(d.IFq)
d["ileum"] = (d.tissue == "Ileum").astype(int)
or_line("primary test, quantile-normalized", smf.logit("resp ~ x + ileum", data=d).fit(disp=0), "x")
print(f"AUC {roc_auc_score(1 - d.resp, d.IFq):.2f}; correlation with the original score r = "
      f"{np.corrcoef(s.IF, s.IFq)[0, 1]:.3f}")
