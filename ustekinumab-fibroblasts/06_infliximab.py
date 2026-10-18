"""Extension 1 in PLAN.md: the frozen 8-gene score in infliximab-treated IBD (GSE16879)."""
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score

from common import gene_matrix, probe_to_symbol, read_series_matrix

rng = np.random.default_rng(20261017)
signature = json.load(open("results/signatures.json"))["IF"]

expr, s = read_series_matrix("data/GSE16879_series_matrix.txt.gz")
expr = np.log2(expr.clip(lower=0) + 1)
genes = gene_matrix(expr, probe_to_symbol("GPL570"))
print(f"GSE16879: {genes.shape[0]} genes x {genes.shape[1]} samples; log2 values {genes.values.min():.1f} to {genes.values.max():.1f}")
missing = [g for g in signature if g not in genes.index]
print("signature genes missing from GPL570:", missing or "none")

s = s.rename(columns={"response to infliximab": "response", "before or after first infliximab treatment": "time"})
s["time"] = s["time"].str.split().str[0]
s["patient"] = s["title"].str.replace(r"_(before|after)T$", "", regex=True)
s["resp"] = (s["response"] == "Yes").astype(int)
s["group"] = np.select([s.disease == "Control", s.disease == "UC", s.tissue == "Ileum"],
                       ["control", "UC", "CD ileum"], "CD colon")
base = s[s["time"] == "Before"]


def score(gene_list):
    """Mean z-score, with each tissue z-scored against its own baseline IBD biopsies."""
    out = pd.Series(index=s.index, dtype=float)
    for tissue in ["Colon", "Ileum"]:
        ref = base.index[base.tissue == tissue]
        idx = s.index[s.tissue == tissue]
        sub = genes.loc[gene_list]
        z = sub[idx].sub(sub[ref].mean(axis=1), axis=0).div(sub[ref].std(axis=1), axis=0)
        out[idx] = z.mean(axis=0)
    return out


s["IF"] = score(signature)
base = s[s["time"] == "Before"].copy()

# ---------------------------------------------------------------- checks before outcomes
print("\n=== Checks (no response labels) ===")
ctrl = s[s.disease == "Control"]
for tissue in ["Colon", "Ileum"]:
    a, c = base.IF[base.tissue == tissue], ctrl.IF[ctrl.tissue == tissue]
    print(f"{tissue}: IBD baseline median {a.median():+.2f} (n = {len(a)}) vs control {c.median():+.2f} "
          f"(n = {len(c)}), Mann-Whitney p = {stats.mannwhitneyu(a, c).pvalue:.1e}")
bins = pd.qcut(genes[base.index].mean(axis=1), 20, labels=False)
pools = [bins.index[bins == bins[g]] for g in signature]


def coherence(gl):
    c = np.corrcoef(genes.loc[gl, base.index].values)
    return c[np.triu_indices_from(c, 1)].mean()


obs = coherence(signature)
null = [coherence([rng.choice(p) for p in pools]) for _ in range(1000)]
print(f"gene coherence: mean pairwise r {obs:.2f}; random sets as high: {np.mean(np.array(null) >= obs):.3f}")
for g in signature:
    print(f"  {g:8s} r with score {np.corrcoef(genes.loc[g, base.index], base.IF)[0, 1]:.2f}, "
          f"mean log2 {genes.loc[g, base.index].mean():.1f}")


def per_sd(x):
    return (x - x.mean()) / x.std()


def or_line(label, fit, term):
    b, (lo, hi), p = fit.params[term], fit.conf_int().loc[term], fit.pvalues[term]
    print(f"{label:50s} OR per SD {np.exp(b):.2f} (95% CI {np.exp(lo):.2f} to {np.exp(hi):.2f}), p = {p:.3f}")
    return b


# ---------------------------------------------------------------- primary: Crohn's disease
print("\n=== Primary: Crohn's disease, baseline ===")
cd = base[base.disease == "CD"].copy()
cd["x"] = per_sd(cd.IF)
cd["ileum"] = (cd.tissue == "Ileum").astype(int)
print(f"{len(cd)} patients, {cd.resp.sum()} responders")
fit = smf.logit("resp ~ x + ileum", data=cd).fit(disp=0)
coef = or_line("score + tissue", fit, "x")
print(f"AUC of a high score for non-response: {roc_auc_score(1 - cd.resp, cd.IF):.2f}")
print("response by score tertile:", cd.resp.groupby(pd.qcut(cd.IF, 3, labels=["low", "mid", "high"]),
                                                     observed=True).mean().round(2).to_dict())

null_b = []
for _ in range(1000):
    cd["r"] = per_sd(score([rng.choice(p) for p in pools])[cd.index])
    null_b.append(smf.logit("resp ~ r + ileum", data=cd).fit(disp=0).params["r"])
null_b = np.array(null_b)
beats = np.mean(null_b > coef)
print(f"random sets with a coefficient at least as negative: {np.mean(null_b <= coef):.3f} (score beats {beats:.1%})")
print(f"pre-set criteria: CI excludes 1 toward lower response: {fit.conf_int().loc['x', 1] < 0}; "
      f"beats >= 95% of random sets: {beats >= 0.95}")

# ---------------------------------------------------------------- secondary
print("\n=== Secondary ===")
uc = base[base.disease == "UC"].copy()
uc["x"] = per_sd(uc.IF)
fit_uc = smf.logit("resp ~ x", data=uc).fit(disp=0)
coef_uc = or_line(f"1. UC positive control (n = {len(uc)}, {uc.resp.sum()} responders)", fit_uc, "x")
print(f"   AUC for non-response {roc_auc_score(1 - uc.resp, uc.IF):.2f}")
null_uc = []
for _ in range(1000):
    uc["r"] = per_sd(score([rng.choice(p) for p in pools])[uc.index])
    null_uc.append(smf.logit("resp ~ r", data=uc).fit(disp=0).params["r"])
print(f"   random sets at least as negative: {np.mean(np.array(null_uc) <= coef_uc):.3f}")
uc_wo = uc.copy()
uc_wo["x"] = per_sd(score([g for g in signature if g != "IL11"])[uc.index])
or_line("   UC, signature without IL11", smf.logit("resp ~ x", data=uc_wo).fit(disp=0), "x")

for grp in ["CD ileum", "CD colon"]:
    d = base[base.group == grp].copy()
    d["x"] = per_sd(d.IF)
    try:
        or_line(f"2. {grp} (n = {len(d)}, {d.resp.sum()} responders)", smf.logit("resp ~ x", data=d).fit(disp=0), "x")
    except Exception as e:
        print(f"2. {grp}: model failed ({e})")
    print(f"   AUC for non-response {roc_auc_score(1 - d.resp, d.IF):.2f}; median score responders "
          f"{d.IF[d.resp == 1].median():+.2f}, non-responders {d.IF[d.resp == 0].median():+.2f}")

print("\n3. Change before -> after infliximab (paired)")
w = s[s.disease != "Control"].pivot_table(index="patient", columns="time", values="IF", aggfunc="first")
info = base.set_index("patient")[["group", "resp"]]
pc = w.join(info, how="inner").dropna(subset=["Before", "After"])
pc["change"] = pc["After"] - pc["Before"]
print(pc.groupby(["group", "resp"]).agg(n=("change", "size"), before=("Before", "mean"),
                                         after=("After", "mean"), change=("change", "mean")).round(2).to_string())
f = smf.ols("After ~ Before + resp + C(group)", data=pc).fit()
print(f"after-infliximab score, responders vs non-responders, given baseline: {f.params['resp']:+.2f} "
      f"({f.conf_int().loc['resp', 0]:+.2f} to {f.conf_int().loc['resp', 1]:+.2f}), p = {f.pvalues['resp']:.1e}, n = {len(pc)}")
for grp, d in pc.groupby("group"):
    for r, dd in d.groupby("resp"):
        p = stats.wilcoxon(dd.change).pvalue if len(dd) >= 5 else float("nan")
        print(f"   {grp:9s} {'responders' if r else 'non-responders':15s} n = {len(dd):2d}, mean change "
              f"{dd.change.mean():+.2f}, Wilcoxon p = {p:.3f}")

s.to_parquet("results/infliximab_samples_scored.parquet")
