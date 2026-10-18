# Plan: inflammatory fibroblasts and ustekinumab response (GSE112366)

Written 2026-10-17, before looking at any link between expression and outcome. Steps 1 to 3 build the score and check that it behaves. The outcome tests in steps 4 and 5 are fixed here so that the result can't steer the method.

## Question

**Primary.** In UNITI-2, among patients given ustekinumab, does a stronger inflammatory-fibroblast signal in the baseline ileal biopsy go with a lower chance of clinical response at week 8?

**Secondary.**
- Does the signal predict response on placebo as well? If it does, it marks the course of the disease (prognostic), not the benefit of the drug (predictive).
- Does ustekinumab lower the signal by week 8, and does it fall only in the patients who respond?
- Does the signal stay down at week 44 in patients kept on ustekinumab?

## Data

- **GSE112366 (main).**
  - Ileal biopsies from UNITI-2 and IM-UNITI, on the Affymetrix GPL13158 microarray.
  - 362 Crohn's samples from 171 patients, plus 26 non-IBD controls.
  - Each sample records the induction arm, the visit (week 0, 8 or 44), week-8 response, the maintenance arm, disease location, ileal SES-CD and microvilli length.
  - Response is reported as a CDAI drop of at least 100 or CDAI below 150. Confirm this against the UNITI paper.
- **GSE207022 (secondary).**
  - Baseline rectal biopsies from UNITI-2: 125 Crohn's patients, 115 of whom are also in GSE112366, plus 23 controls. Same platform.
  - Gives the ustekinumab dose and mucosal healing at week 8 (12 yes, 89 no).
  - Too few healing events for a main test. Use it for dose labels and as a second tissue.
- **Koplev et al. Xenium atlas.** The 2.1-million-cell file from the Xenium note, already downloaded. Used only to build the signature.
- **GPL13158 annotation.** Maps array probes to genes.

**Baseline counts, ileal involvement only** (the primary population):

| Arm | Responders | Non-responders |
|---|---|---|
| Ustekinumab | 37 | 29 |
| Placebo | 12 | 32 |

## Prior work on these data

Cite these, and frame the note around what they did not test.

- **VanDussen et al. (Gastroenterology, 2018), and a follow-up on UNITI-2 biopsies.** Microvilli length predicted ustekinumab response: 65% vs 39% response in patients with normal microvilli.
- **Frontiers in Immunology (2024).** Found WGCNA hub genes MUC1, DUOX2, LCN2 and PDZK1IP1, which were lower in responders.
- **Pavlidis et al. (Cell Reports, 2022).** Used GSE207022 for cytokine networks in organoids.
- **Friedrich et al. (Nature Medicine, 2021).** Linked a fibroblast and neutrophil pattern at the ulcer bed to non-response across several therapies. Check whether they included ustekinumab data.

I found no analysis of an inflammatory-fibroblast signature in these two datasets. Search again before publishing.

## Step 1. Build the signature (no trial data used)

1. **Pseudobulk.** From the Xenium atlas, compute mean normalized expression per fine cell type, as pseudobulk per tissue section, so that one section can't dominate.
2. **Keep specific genes.** Keep a gene if the inflammatory fibroblast ("M COL7A1hi inflammatory fibroblast") is its highest-expressing cell type, at least 2× the next highest, and it is detected in at least 20% of inflammatory fibroblasts.
3. **Freeze the list.** Take the 25 genes with the highest ratio that are on the array, and save the list with the date before going further.
4. **Build two comparison scores the same way:**
   - **pan-fibroblast,** to separate "more fibroblasts" from "fibroblasts in the inflammatory state";
   - **ulcer-immune,** from neutrophil and inflammatory-monocyte genes, to test whether the result is about fibroblasts or just about inflammation.

The atlas comes from surgical resections in treatment-refractory patients, while the trial biopsies come from endoscopy. A signature that is specific in the atlas may not stay specific in bulk biopsies. Step 3 checks this.

### Amendment 1 (2026-10-17, before any trial outcome was used)

**Why the rule changed.** Only one gene, COL7A1, passed the original Step 1 rule, and the "top 25" filled up with housekeeping genes (BRD4, LUC7L, EWSR1). There were three causes:
- **Tiny cell types.** Types with very few cells (T_CD8_RM had 24 usable cells) set noisy "next highest" values.
- **The detection cutoff.** The 20% detection cutoff removed secreted factors such as CXCL5, which Xenium picks up at low counts per cell. CXCL5 is about 10× higher in inflammatory fibroblasts than in any other fibroblast.
- **Gene names.** The GPL13158 annotation uses old gene symbols (WISP1 for CCN4), so genes are now matched through Entrez IDs and HGNC.

**New rule for the inflammatory-fibroblast signature.** A gene is kept if all of these hold:
- the inflammatory fibroblast expresses it at 100 CPM or more;
- it is detected in at least 5% of inflammatory fibroblasts;
- it is at least 1.5× higher than in every other fibroblast type;
- it is at least 2× higher than in every non-fibroblast type.

Cell types with fewer than 1,000 cells, and unlabeled cells, are left out of the comparison.

**Resulting signatures** (saved in `results/signatures.json`):
- **Inflammatory fibroblast (8 genes):** CXCL5, MMP13, COL7A1, IL11, C1QTNF6, CCN4, INHBA, PTGES.
- **Strict version (5 genes, at least 2× on both comparisons):** CXCL5, MMP13, COL7A1, IL11, C1QTNF6.
- **Loose version (15 genes, at least 1.25× and 1.5×):** the 8 genes plus CCDC57, CXCL6, PDPN, FAP, TWIST1, ADAMTS2, CA12. These replace the 15- and 50-gene sensitivity versions.
- **Pan-fibroblast (4 genes):** STEAP1, SPON2, TWIST2, THBS2. This set is thin, because Xenium assigns some fibroblast transcripts to the macrophages next to them. So adjusting for it only partly removes the effect of fibroblast numbers.
- **Comparison score:** the atlas has no neutrophil label, so it uses monocytes and CD14hi macrophages against non-myeloid cells. That gives a 25-gene monocyte-macrophage score: VSIG4, C5AR1, TLR2, FCGR2A, CD163, CSF1R and others.

## Step 2. Process the array

- Check whether the series-matrix values are already log2 RMA, and log-transform if not.
- Map probes to genes with the GPL13158 annotation. Where several probes map to one gene, keep the probe with the highest mean.
- **Score:** the mean of the signature genes after z-scoring each gene across baseline Crohn's samples. As a robustness check, also compute a rank-based score.
- **QC:** look for outliers on PCA, check sex against XIST and RPS4Y1, and look for scan-date batches.

## Step 3. Sanity checks (the outcome analysis waits until these pass)

- The score is higher in Crohn's than in non-IBD controls.
- It is higher in patients with ileal involvement than in colon-only patients.
- It rises with ileal SES-CD.
- The signature genes correlate with each other in the biopsies more than random gene sets of the same size do.

If these fail, the signature does not carry over to bulk biopsies. In that case, stop and report that result.

## Step 4. Primary analysis

- **Population:** baseline biopsies from patients on ustekinumab with ileal involvement (n = 66).
- **Model:** logistic regression of week-8 response on the score (per SD).
- **Report:** the odds ratio with its 95% CI, and the AUC with a bootstrap CI.
- **Adjusted models:**
  - add ileal SES-CD (missing for some patients, so this model is smaller);
  - add the pan-fibroblast score instead.
- **Power:** with 66 patients there is 80% power only for an odds ratio of about 2 per SD, which is an AUC around 0.69. A null result here is weak evidence of no effect, and the write-up has to say so.

**What counts as a finding**, fixed in advance:
1. the CI of the primary odds ratio excludes 1, in the direction of lower response with a higher score;
2. the score beats at least 95% of 1,000 random gene sets matched for size and mean expression.

Anything short of both is reported as inconclusive.

## Step 5. Secondary analyses (exploratory)

1. **Prognostic or predictive.** Fit response ~ score × arm in all 110 patients with ileal involvement. Report the interaction with its CI; it is underpowered.
2. **Specificity.** Compare the score with the ulcer-immune score and with the random gene sets. If any inflammation score does as well, the finding is about inflammation.
3. **Pharmacodynamics.** In patients with both week-0 and week-8 samples, look at the change in score by arm and by response, using a mixed model with a patient random effect.
4. **Week 44.** Look at the score by maintenance arm and by week-8 response.
5. **Rectum.** Apply the same score to the GSE207022 rectal biopsies, joined to the GSE112366 response labels by patient ID, to see whether colonic tissue tells the same story.
6. **Dose.** Use the doses from GSE207022 for the patients who appear in both datasets.

**Sensitivity analyses:**
- all patients, including colon-only;
- 15- and 50-gene signatures;
- the rank-based score;
- dropping biopsies with SES-CD 0.

## Representativeness

Compare the response rates in the biopsy substudy with the published UNITI-2 rates. The substudy has 48 of 86 responders on ustekinumab and 19 of 55 on placebo at baseline.

## Caveats to carry into any write-up

- The response definition is clinical (CDAI), not endoscopic.
- Bulk microarray mixes the number of fibroblasts with their state.
- The GEO summary says patients were largely anti-TNF naive.
- The atlas patients (refractory, at surgery) differ from the trial patients.
- The two datasets share patients, so the rectal analysis is not an independent replication.

## Outputs

- **Code and results** go in `research/ustekinumab-fibroblasts/`:
  - numbered scripts;
  - `data/` for the downloads;
  - `results/` for tables and figures.
- **Results summary first.** I write a short summary of the results for NR before drafting any note.
- **Then the note.** A notebook in `notebooks/` and a note, if the result is worth writing up. A clean null result can also be written up.

## Extension 1: infliximab (GSE16879)

Added 2026-10-17, after the ustekinumab results and before any GSE16879 response label was used.

**Question.** Does the same frozen 8-gene score predict endoscopic and histologic response to a first infliximab infusion?

**Why this dataset.** It differs from UNITI in two ways that bear on the caveats above:
- Response is mucosal (endoscopic and histologic healing at 4–6 weeks), not CDAI.
- Biopsies were taken from actively inflamed mucosa.

**Data.** GSE16879 (Arijs et al., Leuven), on GPL570 (U133 Plus 2.0). It holds 61 refractory IBD patients:
- Crohn's ileitis: 18 patients, 8 responders.
- Crohn's colitis: 19 patients, 12 responders.
- UC: 24 patients, 8 responders.
- Controls: 12 non-IBD (6 colon, 6 ileum).

Biopsies were taken before infliximab and 4–6 weeks after.

**Processing, fixed now.**
- Values are linear (MAS5-like, up to about 68,000), so use log2(x + 1).
- Probes map to genes through Entrez IDs and HGNC, keeping the highest-mean probe per gene, as for GPL13158.
- Score: z-score each signature gene against the baseline IBD biopsies of the same tissue (colon: UC and Crohn's colitis; ileum: Crohn's ileitis), then average the 8 genes. Z-scoring within tissue removes the colon–ileum difference.

**Checks before outcomes.**
- Within each tissue, baseline IBD scores higher than controls.
- The genes are coherent, compared with matched random sets.

**Primary (Crohn's disease).**
- Population: baseline biopsies from all 37 Crohn's patients (20 responders, 17 non-responders).
- Model: logistic regression, response ~ score (per SD) + tissue.
- Finding criteria: the same as the ustekinumab test. The CI must exclude 1 toward lower response, and the score must beat at least 95% of 1,000 random gene sets matched for expression.
- Power: with 37 patients, there is 80% power only for an odds ratio of about 2.5 (or 0.4) per SD.

**Secondary analyses.**
1. **UC as a positive control (24 patients, 8 responders).** Arijs et al. (Gut, 2009) reported that IL13RA2 and IL11, both inflammatory-fibroblast genes, predicted infliximab response in UC. A higher score should go with less healing. IL11 is one of our 8 genes, so this is not independent of that earlier finding.
2. **Crohn's ileum and colon separately.** These are descriptive, with 18 and 19 patients.
3. **Change in score from before to after infliximab, by response.** Use the paired samples, with disease and tissue as covariates.
4. **Comparison with UNITI.** Is the direction the same, and how big is the effect next to the ustekinumab estimate?

**Results of the primary test.** In 37 Crohn's patients the odds ratio was 0.07 per SD (0.01 to 0.37), the AUC 0.91, and the score beat all 1,000 random gene sets, so both criteria are met. The full output is in `results/06_infliximab.log`.

### Extension 1b: checks added after seeing the infliximab result

Added 2026-10-17. These are secondary and were chosen after the primary result. They apply checks that were already part of the ustekinumab plan, and are not new hypotheses.

1. **Inflammation.** Score the 25-gene monocyte-macrophage set (`inflammatory_myeloid` in `signatures.json`) the same way, then fit:
   - response ~ monocyte-macrophage + tissue;
   - response ~ score + monocyte-macrophage + tissue.

   If the monocyte-macrophage score predicts as well, and the inflammatory-fibroblast score loses its effect next to it, the finding is about inflammation.
2. **Pan-fibroblast.** Add the pan-fibroblast score as a covariate.
3. **Normalization.** Quantile-normalize the log2 matrix across all 133 arrays, rescore, and rerun the primary test.
4. **Crohn's colon separates perfectly**, so the regression can't estimate an odds ratio there. Report the AUC and a Mann–Whitney test instead.

## Extension 2: endoscopic response within UNITI

Added 2026-10-18, after the infliximab result and before week-8 SES-CD was compared with any score.

**Question.** In the same trial, on the same drug, does the baseline score predict endoscopic improvement at week 8?

**Why.** The ustekinumab test used CDAI, which is mostly symptoms. The infliximab test used endoscopic and histologic healing, and found a strong link. If the score also predicts endoscopic improvement in UNITI, the CDAI endpoint is the likely reason for the null result. If it does not, the gap more likely comes from the drug or the biopsy.

**Data.** GSE112366 records SES-CD for the ileum (0 to 12) at each biopsy visit. Already known before this extension: the baseline score correlates with baseline SES-CD (rho = 0.54). Week-8 SES-CD has not been looked at against any score.

**Population, fixed now.**
- Ustekinumab arm with ileal involvement, the same as the primary test (66 patients).
- Baseline ileal SES-CD of 3 or more, so there is room to improve (34 patients).
- Week-8 ileal SES-CD recorded: 28 patients. Responders not yet counted.

**Endpoint.** Endoscopic response: ileal SES-CD falls by at least 50% from week 0 to week 8.

**Primary.**
- Model: logistic regression, endoscopic response ~ score (per SD).
- Finding criteria: the same as before. The CI must exclude 1 toward less improvement, and the score must beat at least 95% of 1,000 random gene sets matched for expression.
- Power: with 28 patients, only a large effect can be found.
- Reading, fixed now:
  - criteria met: the score predicts endoscopic response on ustekinumab;
  - CI lower limit above 0.37 (the upper limit of the infliximab estimate): an effect as large as the infliximab one is ruled out;
  - otherwise: inconclusive.

**Secondary analyses.**
1. **Baseline SES-CD as a covariate.** Endoscopic response ~ score + baseline SES-CD.
2. **Continuous outcome.** Week-8 SES-CD ~ baseline SES-CD + score, in all ustekinumab patients with ileal involvement and both values recorded (38 patients).
3. **Placebo arm.** The primary model in placebo patients who meet the same criteria (15 patients). Descriptive only.
4. **Monocyte-macrophage score.** The primary model with this score in place of the inflammatory-fibroblast score, as in Extension 1b.
5. **Agreement of endpoints.** Cross-tabulate CDAI response and endoscopic response in the primary population.

## Extension 3: vedolizumab in ulcerative colitis (GSE73661)

Added 2026-10-18, before any GSE73661 outcome was compared with any score. Looked at so far: sample labels, week-0 Mayo endoscopic subscores, which follow-up visits exist, the expression scale, and gene coverage.

**Question.** In a third drug and disease, does the same frozen 8-gene score predict endoscopic healing? Does the link survive allowing for baseline endoscopic severity?

**Data.** GSE73661 (Arijs et al., Gut 2018), GPL6244 (Affymetrix Human Gene 1.0 ST). Values are already log2 (2.0 to 14.6), so they are used as they are.
- Colonic biopsies from UC patients in GEMINI I (vedolizumab induction, biopsies at week 0 and week 6) and GEMINI LTS (open-label vedolizumab, biopsies at week 0 and week 12).
- Week-0 biopsies: 41 vedolizumab patients, 3 placebo patients, 23 infliximab patients, plus 12 non-IBD controls.
- The infliximab patients are probably the Leuven UC patients already in GSE16879. They are used only as part of the z-score reference, not tested.
- Every sample has a Mayo endoscopic subscore. Week-0 subscore in the vedolizumab patients: 16 at 2, 25 at 3.

**Processing, fixed now.**
- Probes map to genes as before, through Entrez IDs and HGNC, keeping the highest-mean probe per gene.
- New rule for this array only: a probe whose Gene ID lists one gene twice counts as single-gene. GPL6244 lists the INHBA probe as "3624///3624", and without this rule INHBA would be lost. The rule is not applied to the UNITI or Leuven arrays, so their frozen results do not change.
- Coverage: all 8 signature genes once INHBA is kept, and 24 of the 25 monocyte-macrophage genes (no FCGR1A).
- Score: z-score each gene against all week-0 UC biopsies in the dataset (vedolizumab, infliximab and placebo; 67 biopsies), then average the 8 genes.

**Checks before outcomes.**
- Week-0 UC scores higher than the non-IBD controls.
- Gene coherence against matched random sets.

**Population.** Vedolizumab patients with a week-0 biopsy and an endoscopy at week 6 (GEMINI I) or week 12 (GEMINI LTS): 40 patients (27 and 13). The placebo arm (3 patients) is too small to test.

**Endpoint.** Mucosal healing, defined as a Mayo endoscopic subscore of 0 or 1 at that visit (the dataset's own definition).

**Primary.**
- Model: logistic regression, healing ~ score (per SD).
- Finding criteria: the same as before. The CI must exclude 1 toward less healing, and the score must beat at least 95% of 1,000 random gene sets matched for expression on this array.

**Secondary analyses.**
1. **Severity.** Healing ~ score + week-0 Mayo subscore (3 vs 2). This is coarse, so if the score survives it, that is only weak evidence that it goes beyond severity.
2. **Study.** Healing ~ score + study (GEMINI I vs LTS), because healing was judged at different weeks.
3. **Monocyte-macrophage score.** Healing ~ monocyte-macrophage score, and both scores in one model.
4. **Week 52.** Healing at week 52, descriptive only, if enough patients have a biopsy.

**Reading, fixed now.**
- **Criteria met, and the score keeps most of its effect next to the week-0 Mayo subscore:** consistent with a signal beyond severity, within the limits of a 2-vs-3 severity scale.
- **Criteria met, but the effect largely goes away with the Mayo subscore:** consistent with severity, as in UNITI.
- **Criteria not met:** inconclusive for vedolizumab.

## Extension 4: how much of the score is severity? (IBDome)

Added 2026-10-18, before any score was computed in IBDome. Looked at so far: table structure, sample counts, gene coverage, and how many biopsies can be joined to endoscopy and histology scores.

**Question.** In Crohn's disease biopsies from an independent cohort, measured by RNA-seq, how much of the inflammatory-fibroblast score is explained by the inflammation and damage of the same biopsy site? Is there enough variation left over for a separate fibroblast signal to exist?

**Data.** IBDome v1.0.1 (local SQLite; Berlin and Erlangen cohorts).
- 197 Crohn's disease RNA-seq biopsies from 127 patients: 124 ileum, the rest mostly colon. Up to 4 biopsies per patient.
- **Histology:** a normalized Naini–Cortina score, matched on patient, date and tissue, for 191 of these biopsies. This score is specific to the biopsy site.
- **Endoscopy:** SES-CD on the same day for 86 biopsies (49 ileum, 37 colon). This is a whole-bowel score here, not an ileal one, so it is secondary.
- All 8 signature genes and all 25 monocyte-macrophage genes are present.
- No treatment outcomes are used. Only about 27 Crohn's biopsies were taken in the 90 days before a biologic was started, too few for a response test.

**Processing, fixed now.**
- log2(TPM + 1).
- Each gene z-scored against the Crohn's biopsies of the same coarse tissue (ileum or colon), then the 8 genes averaged, as for GSE16879.
- Anastomosis biopsies are left out.
- One biopsy per patient and tissue, the earliest. Sensitivity analysis: all biopsies, with standard errors clustered by patient.

**Checks before the main analysis.**
- The score is higher in Crohn's biopsies than in non-IBD biopsies of the same tissue.
- It is higher in biopsies labelled inflamed than in those labelled non-inflamed.
- The genes are coherent.

**Primary.**
- R² of score ~ histology score + tissue.
- Spearman rho between the score and the histology score.

**Secondary analyses.**
1. R² of score ~ histology + monocyte-macrophage score + tissue. This gives how much of the score is left after two measures of inflammation.
2. Partial correlation of the inflammatory-fibroblast and monocyte-macrophage scores, given histology and tissue.
3. Score vs SES-CD in the 86 biopsies with endoscopy, to replicate UNITI's rho of 0.54 with ileal SES-CD.

**Reading, fixed now.**
- **Histology and the monocyte-macrophage score explain 80% or more of the score's variance:** bulk tissue can't separate fibroblast state from inflammation, and response studies with bulk biopsies can't test the fibroblast hypothesis itself.
- **They explain less than 50%:** there is room for a separate fibroblast signal, and a future response study should test the leftover part.
- **In between:** report as partial.
