# A Bayesian go/no-go for a drug candidate, in PyTorch

Companion notebook for the TensorOmics note:
https://tensoromics.com/notes/bayesian-clinical-risk-pytorch/

A clinical go/no-go decision built as Bayesian logistic regression (variational
inference in PyTorch). With 250 historical trials the Bayesian and frequentist
point estimates agree and the credible interval is the payoff; with a 15-trial
rare-disease archive, a plain logistic regression separates and reports a false
100% GO while the prior keeps the Bayesian estimate finite and honest.

Run:
```
pip install torch numpy matplotlib seaborn
jupyter notebook bayesian-clinical-risk-pytorch.ipynb
```
