"""Step 5: secondary (exploratory) and sensitivity analyses, as listed in PLAN.md."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

s = pd.read_parquet("results/ileum_samples_scored.parquet")
s = s[s["cd"]].copy()
s["resp"] = (s["response"] == "Y").astype(int)
s["ust"] = (s["arm"] == "Ust").astype(int)
ILEAL = ["Ileum only", "Ileum and colon"]
s["ileal"] = s["location"].isin(ILEAL)
base = s[s["visit"] == "I-WK0"].copy()


def per_sd(x):
    return (x - x.mean()) / x.std()


def or_line(label, fit, term):
    b, (lo, hi), p = fit.params[term], fit.conf_int().loc[term], fit.pvalues[term]
    print(f"{label:52s} OR {np.exp(b):.2f} (95% CI {np.exp(lo):.2f} to {np.exp(hi):.2f}), p = {p:.3f}")


# ---------------------------------------------------------------- 1. prognostic or predictive
print("=== 1. Prognostic or predictive (baseline, ileal involvement) ===")
b1 = base[base["ileal"]].copy()
b1["IFz"] = per_sd(b1["IF"])
fit = smf.logit("resp ~ IFz * ust", data=b1).fit(disp=0)
or_line(f"IF x ustekinumab interaction (n = {len(b1)})", fit, "IFz:ust")
for arm, label in [(1, "ustekinumab"), (0, "placebo")]:
    d = b1[b1["ust"] == arm]
    f = smf.logit("resp ~ IFz", data=d).fit(disp=0)
    or_line(f"IF per SD in {label} arm (n = {len(d)}, {d.resp.sum()} responders)", f, "IFz")
for arm, label in [(1, "ustekinumab"), (0, "placebo")]:
    d = b1[b1["ust"] == arm]
    t = pd.qcut(b1["IF"], 3, labels=["low", "mid", "high"])[d.index]
    print(f"  {label} response by IF tertile (tertiles over both arms): "
          + ", ".join(f"{k} {v:.0%} (n={n})" for (k, v), n in zip(d.resp.groupby(t, observed=False).mean().items(),
                                                                    d.resp.groupby(t, observed=False).size())))

# ---------------------------------------------------------------- 2. specificity: monocyte-macrophage
print("\n=== 2. Same tests for the monocyte-macrophage score ===")
b1["MYz"] = per_sd(b1["inflammatory_myeloid"])
fit = smf.logit("resp ~ MYz * ust", data=b1).fit(disp=0)
or_line("monocyte-macrophage x ustekinumab interaction", fit, "MYz:ust")
for arm, label in [(1, "ustekinumab"), (0, "placebo")]:
    f = smf.logit("resp ~ MYz", data=b1[b1["ust"] == arm]).fit(disp=0)
    or_line(f"monocyte-macrophage per SD in {label} arm", f, "MYz")

# ---------------------------------------------------------------- 3. pharmacodynamics
print("\n=== 3. Change from week 0 to week 8 (paired patients) ===")
wide = s.pivot_table(index="subject", columns="visit", values=["IF", "inflammatory_myeloid"], aggfunc="first")
info = base.set_index("subject")[["arm", "response", "ileal", "location"]]
pd_ = pd.DataFrame({
    "dIF": wide[("IF", "I-WK8")] - wide[("IF", "I-WK0")],
    "dMY": wide[("inflammatory_myeloid", "I-WK8")] - wide[("inflammatory_myeloid", "I-WK0")],
}).dropna().join(info, how="inner")
for scope, d in [("ileal involvement", pd_[pd_["ileal"]]), ("all patients", pd_)]:
    print(f"-- {scope} (n = {len(d)})")
    g = d.groupby(["arm", "response"])
    tab = g["dIF"].agg(["count", "mean", "std"])
    tab["p_vs_0"] = g["dIF"].apply(lambda x: stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan)
    print(tab.round(3).to_string())
    d = d.assign(ust=(d["arm"] == "Ust").astype(int), resp=(d["response"] == "Y").astype(int))
    f = smf.ols("dIF ~ ust * resp", data=d).fit()
    print("  OLS dIF ~ ust * resp: " + ", ".join(f"{k} {v:+.2f} (p {f.pvalues[k]:.3f})" for k, v in f.params.items()))
    g = d.groupby(["arm", "response"])["dMY"].mean()
    print("  monocyte-macrophage change: " + ", ".join(f"{a}/{r} {v:+.2f}" for (a, r), v in g.items()))

# mixed model on all three... week 0 and 8 rows, ileal involvement
long = s[s["visit"].isin(["I-WK0", "I-WK8"]) & s["ileal"]].copy()
long["wk8"] = (long["visit"] == "I-WK8").astype(int)
mm = smf.mixedlm("IF ~ wk8 * ust * resp", data=long, groups=long["subject"]).fit()
print("\nMixed model IF ~ wk8*ust*resp + (1|patient), ileal involvement:")
for k in ["wk8", "wk8:ust", "wk8:resp", "wk8:ust:resp"]:
    print(f"  {k:14s} {mm.params[k]:+.3f} (95% CI {mm.conf_int().loc[k, 0]:+.3f} to {mm.conf_int().loc[k, 1]:+.3f}), p = {mm.pvalues[k]:.3f}")

# ---------------------------------------------------------------- 4. week 44
print("\n=== 4. Week 44 ===")
w = s[s["visit"] == "M-WK44"].set_index("subject")
w = w.join(wide[("IF", "I-WK0")].rename("IF0"))
w["group"] = w["arm"] + " induction / " + w["response"].map({"Y": "responder", "N": "non-responder"}) + " / " + w["treatment_maintenance"]
w["dIF"] = w["IF"] - w["IF0"]
print(w.groupby("group").agg(n=("IF", "size"), IF_wk44=("IF", "mean"), change_from_wk0=("dIF", "mean"),
                             n_paired=("dIF", "count")).round(2).to_string())
r = w[(w["arm"] == "Ust") & (w["response"] == "Y")].copy()
r["kept_on_ust"] = r["treatment_maintenance"].isin(["Ust8W", "Ust12W"]).astype(int)
f = smf.ols("IF ~ kept_on_ust", data=r).fit()
print(f"Randomized withdrawal (ustekinumab responders): IF at week 44, kept on drug vs placebo "
      f"{f.params['kept_on_ust']:+.2f} (95% CI {f.conf_int().loc['kept_on_ust', 0]:+.2f} to "
      f"{f.conf_int().loc['kept_on_ust', 1]:+.2f}), p = {f.pvalues['kept_on_ust']:.3f}, n = {len(r)}")

# ---------------------------------------------------------------- 5. rectum
print("\n=== 5. Rectal biopsies (GSE207022), response labels joined from GSE112366 ===")
re = pd.read_parquet("results/rectum_samples_scored.parquet")
re = re[re["cd"]].set_index("subject")
re = re.join(info[["arm", "response"]], how="inner", rsuffix="_il")
re["resp"] = (re["response"] == "Y").astype(int)
re["ust"] = (re["arm"] == "Ust").astype(int)
print(f"{len(re)} rectal patients with ileal response labels; arms {re['arm'].value_counts().to_dict()}")
colonic = re[re["location"].isin(["Ileum and colon", "Colon only"])].copy()
colonic["IFz"] = per_sd(colonic["IF"])
for arm, label in [(1, "ustekinumab"), (0, "placebo")]:
    d = colonic[colonic["ust"] == arm]
    f = smf.logit("resp ~ IFz", data=d).fit(disp=0)
    or_line(f"rectal IF per SD, {label}, colonic disease (n = {len(d)})", f, "IFz")
both = colonic.join(base.set_index("subject")["IF"].rename("IF_ileum"), how="inner")
print(f"rectal vs ileal IF score, same patients: rho {stats.spearmanr(both['IF'], both['IF_ileum']).statistic:+.2f} (n = {len(both)})")
heal = re[re["healing"].isin(["Y", "N"])]
print("mucosal healing at week 8 by arm:", heal.groupby(["arm", "healing"]).size().to_dict())

# ---------------------------------------------------------------- 6. dose
print("\n=== 6. Dose (from GSE207022, ustekinumab patients in both datasets) ===")
dose = pd.read_parquet("results/rectum_samples_scored.parquet").set_index("subject")["treatment"]
u = base[base["ust"] == 1].set_index("subject").join(dose, how="inner")
u["dose"] = np.where(u["treatment"].str.contains("130"), "130 mg", "6 mg/kg")
print(u.groupby("dose")["resp"].agg(["count", "mean"]).round(2).to_string())
ui = u[u["ileal"]].copy()
ui["IFz"] = per_sd(ui["IF"])
f = smf.logit("resp ~ IFz + C(dose)", data=ui).fit(disp=0)
or_line(f"IF per SD adjusted for dose, ileal involvement (n = {len(ui)})", f, "IFz")

# ---------------------------------------------------------------- sensitivity
print("\n=== Sensitivity analyses (ustekinumab arm, baseline) ===")
ust = base[base["ust"] == 1].copy()
for label, d in [("all patients incl. colon-only", ust),
                 ("ileal involvement, SES-CD > 0", ust[ust["ileal"] & (ust["ses_cd"] != ".") & (ust["ses_cd"] != "0")]),
                 ("ileal involvement, SES-CD > 0 or missing", ust[ust["ileal"] & (ust["ses_cd"] != "0")])]:
    for name in ["IF", "IF_strict", "IF_loose", "IF_rank"]:
        d = d.copy()
        d["x"] = per_sd(d[name])
        f = smf.logit("resp ~ x", data=d).fit(disp=0)
        or_line(f"{label}: {name} (n = {len(d)})", f, "x")
