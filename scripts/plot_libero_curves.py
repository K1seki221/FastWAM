import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps = [22000, 24000, 26000, 28000, 30000]
arms = {
    "Baseline (fixed tap)":  ([0.9975, 0.9807, 0.9925, 0.9750, 0.9900], "#444444", "o", "--"),
    "Router, lr 1e-3":       ([0.9950, 0.9975, 0.9925, 0.9975, 0.9850], "#1f77b4", "s", "-"),
    "Router, lr 5e-3":       ([0.9855, 0.9884, 0.9975, 0.9925, 0.9850], "#2ca02c", "^", "-"),
    "Router, lr 1e-2":       ([0.9925, 0.9900, 0.9900, 0.9975, 0.9950], "#d62728", "D", "-"),
}

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)

ax.axhline(0.970, color="#999999", lw=1.0, ls=":", zorder=1)
ax.text(30350, 0.970, "official N1.7 finetune (0.970)", va="center", fontsize=8, color="#777777")
ax.axhline(0.965, color="#bbbbbb", lw=1.0, ls=":", zorder=1)
ax.text(30350, 0.965, "StarVLA-GR00T specialist (0.965)", va="center", fontsize=8, color="#999999")

for label, (ys, color, marker, ls) in arms.items():
    ax.plot(steps, ys, marker=marker, color=color, ls=ls, lw=1.8, ms=6,
            label=f"{label}  (mean {sum(ys)/len(ys):.3f})", zorder=3)

ax.set_xlabel("Training step (stage-1 from scratch)")
ax.set_ylabel("Success rate (40 tasks × 10 episodes)")
ax.set_title("LIBERO (4 suites joint): baseline vs condition-router arms")
ax.set_xticks(steps)
ax.set_xticklabels(["22K", "24K", "26K", "28K", "30K"])
ax.set_xlim(21400, 33800)
ax.set_ylim(0.958, 1.003)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
fig.tight_layout()
fig.savefig("/tmp/libero_router_curves.png", bbox_inches="tight")
print("saved /tmp/libero_router_curves.png")
