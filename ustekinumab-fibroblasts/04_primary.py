"""Step 4: primary analysis, as fixed in PLAN.md.

Baseline ileal biopsies, ustekinumab arm, patients with ileal involvement.
Week-8 response ~ IF score (per SD). Finding = CI excludes 1 toward lower
response AND the score beats >= 95% of 1,000 matched random gene sets.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20261017)
s = pd.read_parquet("results/ileum_samples_scored.parquet")
genes = pd.read_parquet("results/ileum_genes.parquet")
signatures = json.load(open("results/signatures.json"))

ILEAL = ["Ileum only", "Ileum and colon"]
base = s[s["cd"] & (s["visit"] == "I-WK0")]
pop = base[(base["arm"] == "Ust") & base["location"].isin(ILEAL)].copy()
y = (pop["response"] == "Y").astype(int)
print(f"Primary population: {len(pop)} patients, {y.sum()} responders, {len(y) - y.sum()} non-responders")


def logit(y, X):
    fit = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    return fit


def per_sd(x):
    return (x - x.mean()) / x.std()


def report(label, y, X, term):
    fit = logit(y, X)
    b, (lo, hi), p = fit.params[term], fit.conf_int().loc[term], fit.pvalues[term]
    print(f"{label:46s} OR per SD {np.exp(b):.2f} (95% CI {np.exp(lo):.2f} to {np.exp(hi):.2f}), "
          f"p = {p:.3f}, n = {len(y)}")
    return fit


def auc_ci(y, x, n_boot=2000):
    """AUC of a high score for NON-response, with a bootstrap CI."""
    nonresp = 1 - y.values
    point = roc_auc_score(nonresp, x)
    boots = []
    idx = np.arange(len(y))
    while len(boots) < n_boot:
        i = rng.choice(idx, len(idx))
        if nonresp[i].min() != nonresp[i].max():
            boots.append(roc_auc_score(nonresp[i], x.values[i]))
    return point, np.percentile(boots, [2.5, 97.5])


print("\n=== Primary ===")
x = per_sd(pop["IF"])
fit = report("IF score", y, x.rename("IF"), "IF")
auc, (alo, ahi) = auc_ci(y, pop["IF"])
print(f"AUC of a high IF score for non-response: {auc:.2f} (95% CI {alo:.2f} to {ahi:.2f})")
print(f"response rate by IF tertile: "
      + ", ".join(f"{k}: {v:.0%}" for k, v in y.groupby(pd.qcut(pop['IF'], 3, labels=['low', 'mid', 'high'])).mean().items()))

# random gene sets matched on mean expression, same size as the IF signature
ref = genes[base.index]
bins = pd.qcut(ref.mean(axis=1), 20, labels=False)
pools = [bins.index[bins == bins[g]].tolist() for g in signatures["IF"]]
pop_ref_mean = ref.mean(axis=1)
pop_ref_sd = ref.std(axis=1)


def set_score(gene_list):
    z = genes.loc[gene_list, pop.index].sub(pop_ref_mean[gene_list], axis=0).div(pop_ref_sd[gene_list], axis=0)
    return z.mean(axis=0)


observed_b = fit.params["IF"]
null_b = []
for _ in range(1000):
    rs = set_score([rng.choice(p) for p in pools])
    null_b.append(logit(y, per_sd(rs).rename("r")).params["r"])
null_b = np.array(null_b)
frac = np.mean(null_b > observed_b)  # share of random sets with a weaker (less negative) link
print(f"IF coefficient {observed_b:+.3f}; random sets with a coefficient at least as negative: "
      f"{np.mean(null_b <= observed_b):.3f}  (IF beats {frac:.1%} of random sets)")
print(f"random-set coefficients: median {np.median(null_b):+.3f}, 5th-95th percentile "
      f"{np.percentile(null_b, 5):+.3f} to {np.percentile(null_b, 95):+.3f}")

ci_excludes = fit.conf_int().loc["IF", 1] < 0
print(f"\nPre-set finding criteria: CI excludes 1 toward lower response: {ci_excludes}; "
      f"beats >= 95% of random sets: {frac >= 0.95}")

print("\n=== Adjusted models ===")
has = pop["ses_cd"] != "."
yy, sub = y[has], pop[has]
X = pd.DataFrame({"IF": per_sd(sub["IF"]), "ses_cd": sub["ses_cd"].astype(int)})
report("IF score, SES-CD subset, unadjusted", yy, X[["IF"]], "IF")
report("IF score + ileal SES-CD", yy, X, "IF")
X = pd.DataFrame({"IF": per_sd(pop["IF"]), "fib": per_sd(pop["pan_fibroblast"])})
report("IF score + pan-fibroblast score", y, X, "IF")

print("\n=== Other scores, same population (context) ===")
for name in ["pan_fibroblast", "inflammatory_myeloid", "IF_strict", "IF_loose", "IF_rank"]:
    report(name, y, per_sd(pop[name]).rename(name), name)
X = pd.DataFrame({"IF": per_sd(pop["IF"]), "myeloid": per_sd(pop["inflammatory_myeloid"])})
f2 = report("IF score + monocyte-macrophage score", y, X, "IF")
print(f"{'   (monocyte-macrophage term)':46s} OR per SD {np.exp(f2.params['myeloid']):.2f}, p = {f2.pvalues['myeloid']:.3f}")

pd.DataFrame({"null_coef": null_b}).to_csv("results/primary_random_sets.csv", index=False)
