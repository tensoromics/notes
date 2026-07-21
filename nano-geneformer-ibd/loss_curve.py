"""Regenerate the masked-gene pretraining loss curve from logged losses."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

loss = np.load("checkpoints/loss_history.npy")
steps = np.arange(1, len(loss) + 1)
chance = math.log(2050)

def roll(x, w=25):
    if len(x) < w:
        return x, np.arange(len(x))
    k = np.ones(w) / w
    sm = np.convolve(x, k, mode="valid")
    return sm, np.arange(len(sm)) + w // 2

sm, smx = roll(loss, 25)
INK, MUTED, ACC = "#0f172a", "#64748b", "#2563eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(steps, loss, color=ACC, alpha=0.18, lw=0.8)
ax.plot(smx, sm, color=ACC, lw=2)
ax.axhline(chance, ls="--", color=MUTED, lw=1.3)
ax.text(len(loss) * 0.98, chance + 0.05, f"random guess = ln(2050) = {chance:.2f}",
        ha="right", va="bottom", color=MUTED, fontsize=9.5)
final = float(sm[-1])
ax.text(len(loss) * 0.99, final - 0.12, f"{final:.2f}", ha="right", va="top",
        color=ACC, fontsize=11, fontweight="bold")
ax.set_xlabel("training step")
ax.set_ylabel("masked-gene loss (cross-entropy)")
ax.set_title("nano-Geneformer masked-gene pretraining loss",
             fontweight="bold", color=INK, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cbd5e1")
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color="#eef2f7")
ax.set_axisbelow(True)
ax.set_ylim(min(loss.min() * 0.98, 4.6), chance + 0.35)
fig.tight_layout()
fig.savefig("loss-curve.png", dpi=150, facecolor="white", bbox_inches="tight")
print(f"saved loss-curve.png | start {float(loss[0]):.2f} -> final(smoothed) {final:.2f} "
      f"| {len(loss)} steps | e^final = 1-in-{math.exp(final):.0f}")
