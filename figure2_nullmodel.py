"""Figure 2 (fig_nullmodel.pdf): the transport-only null.

(A) one representative ten-organoid layout with a single port, marker area
    proportional to the plateau flux;
(B) relative plateau flux for that layout under point and uniform dosing, both
    normalized to equal array-mean exposure;
(C) the ensemble distribution of the transport-only maturation CV over all
    2000 layouts, point against uniform.

Panel C reads outputs/null_ensemble_raw.npz, so table01_null_ensemble.py must
have been run first; panels A and B are recomputed here (one layout, seconds).
"""
import numpy as np
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
from _common import save, banner, OUT, load_npz
from organoid_null import style
from organoid_null.style import BLUE, ORANGE, SUBTLE, MUTED, INK, SURFACE, GRID
from organoid_null.reduced import Kinetics, fluxes_multi
from organoid_null.ensembles import bootstrap_percentile

DEMO = [(0.20, 0.25), (0.42, 0.18), (0.68, 0.28), (0.85, 0.52), (0.25, 0.55),
        (0.50, 0.50), (0.72, 0.68), (0.30, 0.82), (0.55, 0.80), (0.82, 0.85)]
SRC = (0.10, 0.12)


def main():
    style.use()
    R = load_npz("null_ensemble_raw")
    PT, UN = R["cv_mt"], R["cv_mt_uniform"]
    nlay = len(PT)
    rng = np.random.default_rng(11)

    kin = Kinetics(kp=2.0, sigma=1 / 3., eps=0.02)
    pp, uu = fluxes_multi(DEMO, [SRC, "uniform"], 0.0, 0.4, kin)
    N = len(DEMO)

    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.6))

    a = ax[0]; a.set_aspect("equal")
    a.add_patch(Rectangle((0, 0), 1, 1, fc="#f6f9fd", ec=SUBTLE, lw=1.3, zorder=1))
    a.scatter([c[0] for c in DEMO], [c[1] for c in DEMO],
              s=60 + 900 * (pp / pp.max()), c=BLUE, ec=SURFACE, lw=1.2, zorder=3)
    a.plot(*SRC, marker="*", ms=15, color=ORANGE, zorder=4)
    a.text(SRC[0] + 0.04, SRC[1] - 0.02, "dosing port", fontsize=7.4,
           color=ORANGE, fontweight="bold")
    a.set_xlim(-0.03, 1.03); a.set_ylim(-0.03, 1.03)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    style.panel(a, "A", "Identical organoids, one dosing port",
                f"marker area $\\propto\\pi_j$; max/min $=${pp.max()/pp.min():.0f}")

    b = ax[1]; style.grid(b, "y")
    idx = np.argsort(pp)[::-1]
    b.bar(np.arange(N) - 0.2, (pp / pp.mean())[idx], 0.38, color=BLUE, zorder=3)
    b.bar(np.arange(N) + 0.2, (uu / uu.mean())[idx], 0.38, color=ORANGE, zorder=3)
    b.axhline(1, color=MUTED, lw=0.9, ls=(0, (2, 3)), zorder=2)
    b.set_xticks(np.arange(N)); b.set_xticklabels(np.arange(1, N + 1), fontsize=7)
    b.set_xlabel("organoid, sorted"); b.set_ylabel("relative plateau flux")
    b.text(0.97, 0.94, "point source", transform=b.transAxes, ha="right",
           color=BLUE, fontsize=7.8, fontweight="bold")
    b.text(0.97, 0.86, "uniform source", transform=b.transAxes, ha="right",
           color=ORANGE, fontsize=7.8, fontweight="bold")
    style.panel(b, "B", "Dosing geometry sets the spread",
                "both sources normalized to equal array-mean exposure")

    c = ax[2]; style.grid(c, "x")
    for i, (v, col, lb) in enumerate([(UN, ORANGE, "uniform"), (PT, BLUE, "point")]):
        sub = v[rng.choice(len(v), 400, replace=False)]
        c.scatter(sub, i + 0.13 * rng.standard_normal(len(sub)), s=7, color=col,
                  alpha=0.30, ec="none", zorder=3)
        med, lo, hi = bootstrap_percentile(v, 50)
        c.plot([med] * 2, [i - 0.30, i + 0.30], color=col, lw=2.4, zorder=5)
        for q in (10, 90):
            m2, l2, h2 = bootstrap_percentile(v, q)
            c.plot([m2] * 2, [i - 0.22, i + 0.22], color=col, lw=1.1,
                   ls=(0, (3, 2)), zorder=5)
            c.add_patch(Rectangle((l2, i - 0.22), h2 - l2, 0.44, fc=col,
                                  alpha=0.28, ec="none", zorder=4))
        c.add_patch(Rectangle((lo, i - 0.30), hi - lo, 0.60, fc=col, alpha=0.35,
                              ec="none", zorder=4))
        c.text(med, i + 0.44, f"median {100*med:.0f}%", ha="center", fontsize=7.4,
               color=col, fontweight="bold")
    c.set_yticks([0, 1]); c.set_yticklabels(["uniform", "point"], fontsize=8)
    c.set_xlabel("transport-only maturation CV")
    c.set_ylim(-0.6, 1.75); c.set_xlim(0, 0.95)
    style.panel(c, "C", f"The null model, over {nlay} random arrays",
                "identical biology; heavy bar = median, light bars = 10th/90th pct\n"
                "95% bootstrap CIs are narrower than the linewidth (Tables 1, 3)")

    fig.tight_layout(w_pad=3.0, rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT / "fig_nullmodel.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  demo max/min {pp.max()/pp.min():.1f}   point median {np.median(PT):.4f}"
          f"   uniform median {np.median(UN):.4f}   ratio {np.median(PT)/np.median(UN):.2f}")
    print("  -> outputs/fig_nullmodel.pdf")


if __name__ == "__main__":
    banner("Figure 2 -- the null model")
    main()
