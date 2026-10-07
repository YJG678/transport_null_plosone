"""Figure S1 (fig5_robustness.pdf): acceptance probability.

(A) varying the batch variance at fixed line variance: flat in N_b, because
    neither shared component averages out;
(B) varying the process variance: the N_b-dependence changes sign near
    s_proc = 0.125, but between INTERIOR values -- the dashed marks are the
    N_b -> infinity limits, which are neither 0 nor 1;
(C) the acceptance surface at N_b = 10.

The acceptance machinery is imported from tableS2_acceptance.py so that the
figure and Table S2 use one implementation.  Runtime: about 8 minutes (panel C
is a 15 x 15 grid of Monte-Carlo cells).
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import save, banner, OUT
from organoid_null import style
from organoid_null.style import ORANGE, SUBTLE, MUTED, INK, RAMP
import tableS2_acceptance as ACC

NB = [5, 10, 20, 50]
S_GRID = np.linspace(0, 0.30, 11)
SURF_GRID = np.linspace(0, 0.28, 15)
N_SURF = 6000          # smaller batch count for the 225-cell surface


def main():
    style.use()
    pan = lambda a, L, t, sub=None: style.panel(a, L, t, sub, x=-0.02, y=1.20,
                                                dx=0.075, dy=0.125)
    rng = np.random.default_rng(ACC.SEED)
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.5))
    rec = {}

    a = ax[0]; style.grid(a, "y")
    curves_b = {}
    for col, nb in zip(RAMP[1:], NB):
        y = [ACC.monte_carlo(s, 0.0, nb, rng) for s in S_GRID]
        curves_b[nb] = y
        a.plot(S_GRID, y, color=col, lw=1.9)
    a.set_xlabel("$s_{\\mathrm{batch}}$"); a.set_ylabel("$P_{\\mathrm{accept}}$")
    a.set_ylim(0, 1.05)
    a.text(0.05, 0.12, "curves for $N_b=5,10,20,50$\nlie on top of one another",
           transform=a.transAxes, fontsize=7.2, color=SUBTLE, linespacing=1.5)
    pan(a, "A", "Varying batch variance",
        f"at fixed $s_{{\\mathrm{{line}}}}={ACC.S_LINE}$; flat in $N_b$")
    rec["batch"] = {str(k): v for k, v in curves_b.items()}

    b = ax[1]; style.grid(b, "y")
    curves_p = {}
    for col, nb in zip(RAMP[1:], NB):
        y = [ACC.monte_carlo(0.0, s, nb, rng) for s in S_GRID]
        curves_p[nb] = y
        b.plot(S_GRID, y, color=col, lw=1.9)
        b.annotate(f"$N_b={nb}$", xy=(S_GRID[-1], y[-1]), xytext=(4, 0),
                   textcoords="offset points", fontsize=6.8, color=col,
                   va="center", fontweight="bold")
    b.axvline(0.125, color=MUTED, lw=0.9, ls=(0, (2, 3)))
    b.text(0.132, 0.92, "sign of the\n$N_b$-dependence\nchanges", fontsize=6.9,
           color=SUBTLE, linespacing=1.4)
    lims = {sv: ACC.limit(0.0, sv) for sv in (0.10, 0.15, 0.20, 0.25)}
    for sv, lv in lims.items():
        b.plot([sv - 0.012, sv + 0.012], [lv, lv], color=INK, lw=1.6, zorder=6)
    b.text(0.255, 0.10, "$N_b\\to\\infty$ limits", fontsize=6.9, color=INK, ha="right")
    b.set_xlabel("$s_{\\mathrm{proc}}$"); b.set_ylabel("$P_{\\mathrm{accept}}$")
    b.set_ylim(0, 1.05); b.set_xlim(0, 0.345)
    pan(b, "B", "Varying process variance",
        f"at fixed $s_{{\\mathrm{{line}}}}={ACC.S_LINE}$; limits are interior")
    rec["proc"] = {str(k): v for k, v in curves_p.items()}
    rec["limits"] = {str(k): v for k, v in lims.items()}

    c = ax[2]
    nb_save = ACC.N_BATCH
    ACC.N_BATCH = N_SURF
    Z = np.array([[ACC.monte_carlo(sb, sp, 10, rng) for sb in SURF_GRID]
                  for sp in SURF_GRID])
    ACC.N_BATCH = nb_save
    im = c.pcolormesh(SURF_GRID, SURF_GRID, Z, cmap="Blues", vmin=0, vmax=1,
                      shading="gouraud", rasterized=True)
    cs = c.contour(SURF_GRID, SURF_GRID, Z, levels=[0.5, 0.8],
                   colors=[ORANGE, "#0d366b"], linewidths=1.3)
    c.clabel(cs, fmt="%.1f", fontsize=6.5)
    c.set_xlabel("$s_{\\mathrm{batch}}$"); c.set_ylabel("$s_{\\mathrm{proc}}$")
    c.set_aspect("equal")
    cb = fig.colorbar(im, ax=c, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
    pan(c, "C", "Acceptance surface, $N_b=10$",
        f"baseline accepted at ${Z[0,0]:.2f}$; surface uses {N_SURF} batches per cell")
    rec["surface_axis"] = SURF_GRID.tolist()
    rec["surface"] = Z.tolist()
    rec["baseline"] = float(Z[0, 0])

    fig.tight_layout(w_pad=3.2)
    fig.savefig(OUT / "fig5_robustness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Nb->inf limits: " + "  ".join(f"s_proc={k}: {v:.3f}" for k, v in lims.items()))
    print(f"  baseline acceptance {Z[0,0]:.3f}")
    print("  -> outputs/fig5_robustness.pdf")
    save("figureS1_robustness", rec)


if __name__ == "__main__":
    banner("Figure S1 -- acceptance robustness")
    main()
