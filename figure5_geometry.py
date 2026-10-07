"""Figure 5 (fig4_geometry.pdf): competition and geometry.

(A,B) lossy uptake masses for two competing targets as the source is moved over
      the chamber; because the regime is lossy these are unnormalized masses,
      not capture basins;
(C)   the competition-normalized ratio, with the raw total shown separately;
(D)   a fully specified N = 10 array under an off-centre source.

Runtime: about two minutes (panels A and B evaluate the Green's function on a
70 x 70 grid of source positions).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from _common import save, banner, OUT
from organoid_null import style
from organoid_null.style import BLUE, SUBTLE, MUTED, SURFACE, INK
from organoid_null.green import G_lossy, green_matrix_lossy, gamma_vector_lossy

D, GAM, EPS, ELL, KP = 1.0, 0.4, 0.02, 1.0, 1.5
LAM = np.sqrt(GAM / D)
NU = -1.0 / np.log(EPS)
PSI = D / (KP * ELL)                      # g(0) = 1 here: this figure is the
                                          # irreversible-boundary comparison
PAIR = [(0.35, 0.5), (0.65, 0.5)]
ARRAY = [(0.15, 0.20), (0.35, 0.20), (0.55, 0.20), (0.75, 0.20), (0.25, 0.50),
         (0.45, 0.50), (0.65, 0.50), (0.85, 0.50), (0.35, 0.80), (0.65, 0.80)]
ARRAY_SRC = (0.10, 0.85)
NGRID = 70


def amplitudes(xs, Gam):
    """nu * M^{-1} Gamma for the irreversible boundary at s = 0."""
    GM = green_matrix_lossy(xs, LAM, D, ELL)
    M = np.eye(len(xs)) + 2 * np.pi * NU * D * GM + NU * PSI * np.eye(len(xs))
    return NU * np.linalg.solve(M, Gam)


def main():
    style.use()
    pan = lambda a, L, t, sub=None: style.panel(a, L, t, sub, x=-0.02, y=1.20,
                                                dx=0.075, dy=0.125)
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.2))
    rec = {}

    g = np.linspace(0.03, 0.97, NGRID)
    X, Y = np.meshgrid(g, g)
    Z = np.column_stack([X.ravel(), Y.ravel()])
    G1 = np.array([G_lossy(PAIR[0], z, LAM, D) for z in Z])
    G2 = np.array([G_lossy(PAIR[1], z, LAM, D) for z in Z])
    A = amplitudes(PAIR, np.vstack([G1, G2]))
    P1, P2 = 2 * np.pi * D * A[0], 2 * np.pi * D * A[1]
    vmax = max(P1.max(), P2.max())
    for k, (a, Pk, L, ttl) in enumerate([(ax[0, 0], P1, "A", "target 1"),
                                         (ax[0, 1], P2, "B", "target 2")]):
        im = a.pcolormesh(X, Y, Pk.reshape(NGRID, NGRID), cmap="Blues", vmin=0,
                          vmax=vmax, shading="gouraud", rasterized=True)
        for c in PAIR:
            a.add_patch(Circle(c, 0.022, fc="none", ec=INK, lw=1.4, zorder=4))
        a.add_patch(Circle(PAIR[k], 0.022, fc="#eb6834", ec=INK, lw=1.4, zorder=5))
        a.set_aspect("equal"); a.set_xlabel("$x_{0,1}$"); a.set_ylabel("$x_{0,2}$")
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
        pan(a, L, f"Uptake mass into {ttl}",
            "lossy: $\\sum_k\\pi_k<1$, not a normalized basin")
    rec["pair_max_mass"] = float(vmax)

    c = ax[1, 0]; style.grid(c, "y")
    ds = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    U, C = [], []
    for d in ds:
        pts = [(0.5 - d / 2, 0.5), (0.5 + d / 2, 0.5)]
        GM = green_matrix_lossy(pts, LAM, D, ELL)
        Gam = G_lossy(pts[0], (0.5, 0.5), LAM, D)
        bb = 1 + 2 * np.pi * NU * D * GM[0, 0] + NU * PSI
        cc = 2 * np.pi * NU * D * GM[0, 1]
        U.append(2 * 2 * np.pi * D * NU * Gam / (bb + cc))
        C.append(bb / (bb + cc))
    U, C = np.array(U), np.array(C)
    c.plot(ds, C, color=BLUE, lw=2, marker="o", ms=4, mec=SURFACE, mew=1)
    c.plot(ds, U / U[0] * C[0], color=MUTED, lw=1.6, ls=(0, (5, 2)))
    c.annotate("competition-normalized $\\varrho_{\\mathrm{comp}}(d)$",
               xy=(0.5, C[6]), xytext=(0, -14), textcoords="offset points",
               ha="center", va="top", color=BLUE, fontsize=7.6, fontweight="bold")
    c.annotate("raw total (rescaled)", xy=(0.62, U[7] / U[0] * C[0]), xytext=(0, 10),
               textcoords="offset points", ha="center", color=SUBTLE, fontsize=7.4)
    c.set_xlabel("centre-to-centre spacing $d$"); c.set_ylabel("ratio")
    pan(c, "C", "Competition, with distance divided out",
        "$\\varrho_{\\mathrm{comp}}(d)$ is independent of source position")
    rec["spacing"] = ds.tolist(); rec["rho_comp"] = C.tolist(); rec["U_raw"] = U.tolist()

    d = ax[1, 1]; style.grid(d, "y")
    Gam = gamma_vector_lossy(ARRAY, [ARRAY_SRC], LAM, D)
    pj = np.sort(2 * np.pi * D * amplitudes(ARRAY, Gam))[::-1]
    N = len(pj)
    d.bar(np.arange(N), pj, color=BLUE, width=0.62)
    for i, v in enumerate(pj):
        d.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=6.2, color=SUBTLE)
    d.set_xticks(np.arange(N))
    d.set_xticklabels([f"{i+1}" for i in range(N)], fontsize=7)
    d.set_xlabel("organoid, sorted by $\\pi_j$"); d.set_ylabel("$\\pi_j$")
    d.text(0.97, 0.92,
           f"CV $={pj.std()/pj.mean():.2f}$,  max/min $={pj.max()/pj.min():.0f}$",
           transform=d.transAxes, ha="right", fontsize=7.4, color=INK, fontweight="bold")
    pan(d, "D", "Identical organoids, unequal access",
        "$N=10$ array, off-centre source $(0.10,0.85)$")
    rec["array_pi_sorted"] = pj.tolist()
    rec["array_cv"] = float(pj.std() / pj.mean())
    rec["array_ratio"] = float(pj.max() / pj.min())

    fig.tight_layout(h_pad=3.6, w_pad=3.0)
    fig.savefig(OUT / "fig4_geometry.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  rho_comp: {C[0]:.4f} at d=0.10 -> {C[-1]:.4f} at d=0.80")
    print(f"  N=10 array: CV {rec['array_cv']:.3f}, max/min {rec['array_ratio']:.1f}")
    print("  -> outputs/fig4_geometry.pdf")
    save("figure5_geometry", rec)


if __name__ == "__main__":
    banner("Figure 5 -- competition and geometry")
    main()
