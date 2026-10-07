"""Figure 4 (fig3_residence.pdf): residence-law dispersion.

(A) the three residence densities, matched mean 1, CV 0.5 / 1.0 / 2.1;
(B) the surface-only compound-geometric time, common mean 2.5;
(C) the complete productive flux with the diffusive search restored, by exact
    Laplace inversion;
(D) the disease-score trajectory at matched full mean.

Panels C and D read outputs/residence_curves.npz, written by
table09_residence_laws.py, so that the figure and the table cannot disagree.
Panels A and B are closed-form densities.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as GAM
from _common import banner, OUT, load_npz
from organoid_null import style
from organoid_null.style import ORANGE, SUBTLE, MUTED, RAMP
from organoid_null.phenotype import A_DOSE, T_HORIZON

SIGMA = 0.60
S15 = np.sqrt(15.345)
R1, R2 = 1 / (1 + S15), 1 / (1 - S15 / 9)
LAWS = [("Gamma, CV$_\\tau$=0.5", 0.5, lambda t: (4 ** 4 / 6) * t ** 3 * np.exp(-4 * t),
         "Gamma, CV=0.5"),
        ("exponential, CV$_\\tau$=1.0", 1.0, lambda t: np.exp(-t),
         "exponential, CV=1.0"),
        ("long-tailed, CV$_\\tau$=2.1", 2.1,
         lambda t: 0.1 * R1 * np.exp(-R1 * t) + 0.9 * R2 * np.exp(-R2 * t),
         "long-tailed, CV=2.1")]
COLS = [RAMP[1], RAMP[2], RAMP[4]]


def main():
    style.use()
    pan = lambda a, L, t, sub=None: style.panel(a, L, t, sub, x=-0.02, y=1.20,
                                                dx=0.075, dy=0.125)
    EX = load_npz("residence_curves")
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.2))

    a = ax[0, 0]; style.grid(a, "y")
    tt = np.linspace(0.001, 6, 600)
    for col, (nm, cv, f, _) in zip(COLS, LAWS):
        a.plot(tt, f(tt), color=col, lw=1.9)
        j = int(len(tt) * {0.5: 0.20, 1.0: 0.34, 2.1: 0.55}[cv])
        a.annotate(nm, xy=(tt[j], f(tt[j])), xytext=(8, 6), textcoords="offset points",
                   fontsize=7.2, color=col, fontweight="bold")
    a.set_xlabel("$\\tau$"); a.set_ylabel("$\\phi(\\tau)$")
    a.set_xlim(0, 5); a.set_ylim(0, 1.35)
    pan(a, "A", "Three residence densities", "matched mean $1$; CV $0.5$, $1.0$, $2.1$")

    # Surface-only compound-geometric time: geometric number of cycles with
    # success probability p = 1 - sigma, each of duration tau.  Mean 1/p,
    # variance CV^2/p + (1-p)/p^2, rendered as the matching Gamma density.
    b = ax[0, 1]; style.grid(b, "y")
    p = 1 - SIGMA
    tg = np.linspace(0.01, 14, 500)
    vs = []
    for col, (_, cv, _, _) in zip(COLS, LAWS):
        V = cv ** 2 / p + (1 - p) / p ** 2
        M = 1 / p
        vs.append(V)
        b.plot(tg, GAM.pdf(tg, M * M / V, scale=V / M), color=col, lw=1.9)
    b.axvline(1 / p, color=MUTED, lw=0.9, ls=(0, (2, 3)))
    b.text(1 / p + 0.18, 0.34, "common surface-only\nmean $2.5$", fontsize=7.2,
           color=SUBTLE, linespacing=1.4)
    b.set_xlabel("surface-only time"); b.set_ylabel("density")
    b.set_xlim(0, 10); b.set_ylim(0, 0.42)
    pan(b, "B", "Surface-only compound-geometric time",
        "variances " + ", ".join(f"${v:.3f}$" for v in vs))

    c = ax[1, 0]; style.grid(c, "y")
    te = EX["t"]
    for col, (nm, _, _, key) in zip(COLS, LAWS):
        c.plot(te, A_DOSE * EX[key], color=col, lw=1.9)
    c.axhline(0.8, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
    c.text(0.15, 0.83, "$\\Theta$", color=ORANGE, fontsize=7.6, fontweight="bold")
    c.set_xlabel("time"); c.set_ylabel("$\\mathcal{J}_1(t)$")
    c.set_ylim(0, 1.05); c.set_xlim(0, T_HORIZON)
    pan(c, "C", "Complete flux, search restored",
        "exact inversion; full mean $3.7685$ for all three")

    d = ax[1, 1]; style.grid(d, "y")
    for i, (col, (nm, _, _, _)) in enumerate(zip(COLS, LAWS)):
        y = EX[f"traj_{i}"]
        d.plot(EX["traj_t"], y, color=col, lw=1.9)
        dy = (9, 0, -9)[i]
        d.annotate(nm.split(",")[0], xy=(EX["traj_t"][-1], y[-1]), xytext=(5, dy),
                   textcoords="offset points", fontsize=7, color=col,
                   va="center", fontweight="bold")
    d.set_xlabel("time"); d.set_ylabel("$\\upsilon_{\\mathrm{ds}}(t)$")
    d.set_xlim(0, T_HORIZON * 1.18)
    pan(d, "D", "Phenotype at matched full mean",
        "exact inversion, not the two-moment surrogate")

    fig.tight_layout(h_pad=3.4, w_pad=2.8)
    fig.savefig(OUT / "fig3_residence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  surface-only variances: " + " ".join(f"{v:.3f}" for v in vs))
    print("  terminal disease scores: " +
          " ".join(f"{EX[f'traj_{i}'][-1]:.4f}" for i in range(3)))
    print("  -> outputs/fig3_residence.pdf")


if __name__ == "__main__":
    banner("Figure 4 -- residence laws")
    main()
