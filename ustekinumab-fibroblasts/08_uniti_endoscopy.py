"""Extension 2 in PLAN.md: does the baseline score predict endoscopic response to ustekinumab?

Endoscopic response = ileal SES-CD falls by at least 50% from week 0 to week 8,
in ustekinumab patients with ileal involvement and baseline ileal SES-CD >= 3.
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20261018)
s = pd.read_parquet("results/ileum_samples_scored.parquet")
genes = pd.read_parquet("results/ileum_genes.parquet")
signatures = json.load(open("results/signatures.json"))
s["ses"] = pd.to_numeric(s["ses_cd"], errors="coerce")

ILEAL = ["Ileum only", "Ileum and colon"]
base = s[s["cd"] & (s["visit"] == "I-WK0")]
week8 = s[s["cd"] & (s["visit"] == "I-WK8")].set_index("subject")["ses"].rename("ses8")
b = base[base["location"].isin(ILEAL)].join(week8, on="subject")
b["cdai_resp"] = (b["response"] == "Y").astype(int)
b["endo_resp"] = (b["ses8"] <= b["ses"] / 2).astype(int)


def eligible(arm):
    return b[(b["arm"] == arm) & (b["ses"] >= 3) & b["ses8"].notna()].copy()


def per_sd(x):
    return (x - x.mean()) / x.std()


def report(label, fit, term):
    est, (lo, hi), p = fit.params[term], fit.conf_int().loc[term], fit.pvalues[term]
    print(f"{label:52s} OR per SD {np.exp(est):.2f} (95% CI {np.exp(lo):.2f} to {np.exp(hi):.2f}), p = {p:.3f}")
    return fit


pop = eligible("Ust")
pop["x"] = per_sd(pop["IF"])
print(f"Primary population: {len(pop)} patients, {pop.endo_resp.sum()} endoscopic responders, "
      f"{len(pop) - pop.endo_resp.sum()} non-responders")
print(f"baseline ileal SES-CD median {pop.ses.median():.0f} (range {pop.ses.min():.0f} to {pop.ses.max():.0f}); "
      f"week 8 median {pop.ses8.median():.0f}")

print("\n=== Primary ===")
fit = report("IF score", smf.logit("endo_resp ~ x", data=pop).fit(disp=0), "x")
print(f"AUC of a high IF score for endoscopic non-response: {roc_auc_score(1 - pop.endo_resp, pop.IF):.2f}")
print("response by IF tertile:", pop.endo_resp.groupby(pd.qcut(pop.IF, 3, labels=["low", "mid", "high"]),
                                                     observed=True).agg(["mean", "size"]).round(2).to_dict("index"))

# random gene sets matched on mean expression, scored against the same reference as the IF score
ref = genes[base.index]
bins = pd.qcut(ref.mean(axis=1), 20, labels=False)
pools = [bins.index[bins == bins[g]].tolist() for g in signatures["IF"]]
ref_mean, ref_sd = ref.mean(axis=1), ref.std(axis=1)


def set_score(gene_list, idx):
    z = genes.loc[gene_list, idx].sub(ref_mean[gene_list], axis=0).div(ref_sd[gene_list], axis=0)
    return z.mean(axis=0)


print(f"check: rebuilt IF score matches the saved one: "
      f"{np.allclose(set_score(signatures['IF'], pop.index), pop.IF)}")
observed = fit.params["x"]
null = []
for _ in range(1000):
    pop["r"] = per_sd(set_score([rng.choice(p) for p in pools], pop.index))
    null.append(sm.Logit(pop.endo_resp, sm.add_constant(pop[["r"]])).fit(disp=0).params["r"])
null = np.array(null)
beats = np.mean(null > observed)
print(f"random sets with a coefficient at least as negative: {np.mean(null <= observed):.3f} (IF beats {beats:.1%})")

lo, hi = np.exp(fit.conf_int().loc["x"])
print(f"\nPre-set criteria: CI excludes 1 toward less improvement: {hi < 1}; beats >= 95% of random sets: {beats >= 0.95}")
print(f"CI lower limit {lo:.2f} above 0.37 (an infliximab-sized effect ruled out): {lo > 0.37}")

print("\n=== Secondary ===")
pop["ses_z"] = per_sd(pop["ses"])
f1 = report("1. IF score + baseline SES-CD", smf.logit("endo_resp ~ x + ses_z", data=pop).fit(disp=0), "x")
print(f"{'   (baseline SES-CD term, per SD)':52s} OR {np.exp(f1.params['ses_z']):.2f}, p = {f1.pvalues['ses_z']:.3f}")

cont = b[(b["arm"] == "Ust") & b["ses"].notna() & b["ses8"].notna()].copy()
cont["x"] = per_sd(cont["IF"])
f2 = smf.ols("ses8 ~ ses + x", data=cont).fit()
print(f"2. week-8 SES-CD ~ baseline SES-CD + IF score (n = {len(cont)}): {f2.params['x']:+.2f} points per SD "
      f"({f2.conf_int().loc['x', 0]:+.2f} to {f2.conf_int().loc['x', 1]:+.2f}), p = {f2.pvalues['x']:.3f}")

pbo = eligible("Pbo")
pbo["x"] = per_sd(pbo["IF"])
print(f"3. placebo: {len(pbo)} patients, {pbo.endo_resp.sum()} endoscopic responders")
try:
    report("   IF score, placebo", smf.logit("endo_resp ~ x", data=pbo).fit(disp=0), "x")
except Exception as e:
    print(f"   model failed ({e})")
if 0 < pbo.endo_resp.sum() < len(pbo):
    print(f"   AUC for endoscopic non-response {roc_auc_score(1 - pbo.endo_resp, pbo.IF):.2f}")

pop["m"] = per_sd(pop["inflammatory_myeloid"])
report("4. monocyte-macrophage score", smf.logit("endo_resp ~ m", data=pop).fit(disp=0), "m")
print(f"   AUC for endoscopic non-response {roc_auc_score(1 - pop.endo_resp, pop.inflammatory_myeloid):.2f}")

print("\n5. CDAI response vs endoscopic response, primary population")
print(pd.crosstab(pop.cdai_resp.map({1: "CDAI responder", 0: "CDAI non-responder"}),
                  pop.endo_resp.map({1: "endoscopic responder", 0: "endoscopic non-responder"})).to_string())
