"""Figure 3 (fig2_sigma.pdf): the retention parameter sigma.

(A) eventual uptake against sigma as bulk loss is removed;
(B) the static-Robin overestimation factor, which vanishes as gamma -> 0;
(C) conservative internalization time;
(D) conservative eventual uptake, exactly one;
(E) finite-horizon phenotype;
(F) productive flux under sustained dosing.

Panels A, B are lossy and use the Bessel image representation; C-F are
conservative and use the eigenfunction continuation, which is what is valid at
the negative stencil points of the moment differences.

Runtime: about one minute.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import save, banner, OUT
from organoid_null import style
from organoid_null.style import BLUE, ORANGE, SUBTLE, MUTED, SURFACE, RAMP
from organoid_null.green import Chamber
from organoid_null.reduced import Kinetics, fluxes, one_target_moments
from organoid_null.phenotype import phenotype, gamma_cdf, A_DOSE, T_HORIZON

X1, X0 = [(0.5, 0.5)], (0.25, 0.25)
SG = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
SG2 = np.array([0, 0.1, 0.2, 0.33, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
GAMS = [0.4, 0.1, 0.05, 0.02, 0.004]


def uptake(gam, sigma, kp=1.5):
    return float(fluxes(X1, X0, 0.0, gam, Kinetics(kp=kp, sigma=sigma, eps=0.02))[0])


def main():
    style.use()
    pan = lambda a, L, t, sub=None: style.panel(a, L, t, sub, x=-0.02, y=1.20,
                                                dx=0.075, dy=0.125)
    fig, ax = plt.subplots(2, 3, figsize=(10.6, 6.2))
    rec = {}

    a = ax[0, 0]; style.grid(a, "y")
    curves = {}
    for col, g in zip(RAMP[::-1], GAMS):
        y = [uptake(g, s) for s in SG]
        curves[g] = y
        a.plot(SG, y, color=col, lw=1.8)
        dy = {0.4: 0, 0.1: 0, 0.05: -6, 0.02: -6, 0.004: 8}[g]
        a.annotate(f"$\\gamma={g:g}$", xy=(SG[-1], y[-1]), xytext=(5, dy),
                   textcoords="offset points", fontsize=6.8, color=col,
                   va="center", fontweight="bold")
    a.axhline(1, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    a.set_xlim(-0.02, 1.32); a.set_ylim(0.44, 1.05)
    a.set_xlabel("$\\sigma$"); a.set_ylabel("eventual uptake $\\pi_1$")
    sp = {g: max(curves[g]) - min(curves[g]) for g in GAMS}
    pan(a, "A", "Loss channel drives the $\\sigma$ effect",
        f"spread ${sp[0.4]:.3f}\\to{sp[0.004]:.3f}$ as $\\gamma\\to0$")
    rec["uptake"] = {str(g): curves[g] for g in GAMS}
    rec["spread"] = {str(g): sp[g] for g in GAMS}

    b = ax[0, 1]; style.grid(b, "y")
    fac = {}
    for col, g in zip(RAMP[::-1], GAMS):
        p0 = uptake(g, 0.0)
        y = [p0 / uptake(g, s) for s in SG]
        fac[g] = float(max(y))
        b.plot(SG, y, color=col, lw=1.8)
    b.axhline(1, color=MUTED, lw=0.8, ls=(0, (2, 3)))
    b.set_xlabel("$\\sigma$"); b.set_ylabel("$\\pi^{\\mathrm{Robin}}/\\pi^{\\mathrm{dyn}}$")
    b.text(0.04, b.get_ylim()[1] * 0.97,
           f"over $\\sigma\\in[0,0.95]$:\n$\\gamma=0.4$: max ${fac[0.4]:.3f}$\n"
           f"$\\gamma=0.004$: max ${fac[0.004]:.3f}$\n$\\gamma=0$: exactly $1$",
           fontsize=7, color=SUBTLE, va="top", linespacing=1.5)
    pan(b, "B", "Static-boundary overestimation",
        "vanishes in the conservative limit")
    rec["robin_factor"] = {str(g): fac[g] for g in GAMS}

    ch = Chamber(0.0, [X1[0], X0])
    mom = [one_target_moments(ch, X1[0], X0, Kinetics(kp=2.0, sigma=s, eps=0.02))
           for s in SG2]
    T1 = np.array([float(m[1]) for m in mom])
    sdv = np.array([float(m[2]) for m in mom])
    pi_cons = np.array([float(m[0]) for m in mom])

    c = ax[0, 2]; style.grid(c, "y")
    c.plot(SG2, T1, color=BLUE, lw=2, marker="o", ms=3.6, mec=SURFACE, mew=1)
    c.set_xlabel("$\\sigma$"); c.set_ylabel("$\\mathcal{T}^{\\mathrm{tot}}_1$")
    pan(c, "C", "Conservative internalization time",
        f"${T1[0]:.3f}\\to{T1[-1]:.3f}$; retention delays, never blocks")
    rec["T1_conservative"] = T1.tolist(); rec["sd_conservative"] = sdv.tolist()

    d = ax[1, 0]; style.grid(d, "y")
    d.plot(SG2, pi_cons, color=BLUE, lw=2.2)
    d.set_ylim(0.94, 1.06); d.set_yticks([0.95, 1.0, 1.05])
    d.set_xlabel("$\\sigma$"); d.set_ylabel("$\\pi_1$")
    d.text(0.5, 1.035, "$\\pi_1=1$ to twelve digits", ha="center", fontsize=7.8,
           color=BLUE, fontweight="bold")
    pan(d, "D", "Conservative eventual uptake", "exact: a molecule cannot escape")
    rec["pi_conservative_max_dev"] = float(np.abs(pi_cons - 1).max())

    Y = np.stack([phenotype(np.array([1.0]), T1[i], sdv[i])[:, 0]
                  for i in range(len(SG2))], 1)
    e = ax[1, 1]; style.grid(e, "y")
    e.plot(SG2, Y[2], color=BLUE, lw=2, marker="o", ms=3.6, mec=SURFACE, mew=1)
    e.plot(SG2, Y[4], color=ORANGE, lw=2, marker="s", ms=3.6, mec=SURFACE, mew=1)
    e.annotate("$\\upsilon_{\\mathrm{mt}}(T)$", xy=(0.55, Y[2][5]), xytext=(0, -13),
               textcoords="offset points", ha="center", va="top", color=BLUE,
               fontsize=7.6, fontweight="bold")
    e.annotate("$\\upsilon_{\\mathrm{ds}}(T)$", xy=(0.6, Y[4][5]), xytext=(-6, 12),
               textcoords="offset points", ha="right", color=ORANGE,
               fontsize=7.6, fontweight="bold")
    e.set_xlabel("$\\sigma$"); e.set_ylabel("terminal state"); e.set_ylim(0, 0.79)
    ib = int(np.argmin(np.abs(SG2 - 0.33)))          # baseline sigma = 1/3
    pan(e, "E", "Finite-horizon phenotype",
        f"monotone; baseline ${Y[2][ib]:.3f}/{Y[4][ib]:.3f}$")
    rec["phenotype_mt"] = Y[2].tolist(); rec["phenotype_ds"] = Y[4].tolist()

    t = np.linspace(0, T_HORIZON, 400)
    f = ax[1, 2]; style.grid(f, "y")
    for col, s in zip([RAMP[1], RAMP[2], RAMP[4]], [0.0, 0.6, 0.95]):
        i = int(np.argmin(np.abs(SG2 - s)))
        H = gamma_cdf(T1[i], sdv[i])(t)
        f.plot(t, A_DOSE * H, color=col, lw=1.8)
        f.annotate(f"$\\sigma={s:g}$", xy=(t[300], A_DOSE * H[300]), xytext=(4, -6),
                   textcoords="offset points", fontsize=7, color=col, fontweight="bold")
    f.axhline(0.8, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
    f.text(0.1, 0.83, "$\\Theta$", color=ORANGE, fontsize=7.6, fontweight="bold")
    f.set_xlabel("time"); f.set_ylabel("$\\mathcal{J}_1(t)$"); f.set_ylim(0, 1.1)
    pan(f, "F", "Productive flux under sustained dosing",
        "plateau $=A$ for every $\\sigma$")

    fig.tight_layout(h_pad=3.4, w_pad=2.6)
    fig.savefig(OUT / "fig2_sigma.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Robin factors over [0,0.95]: " +
          "  ".join(f"g={g}: {fac[g]:.3f}" for g in GAMS))
    print(f"  T1 {T1[0]:.4f} -> {T1[-1]:.4f};  max |pi-1| = {rec['pi_conservative_max_dev']:.2e}")
    print("  -> outputs/fig2_sigma.pdf")
    save("figure3_sigma", dict(sigma_lossy=SG.tolist(), sigma_cons=SG2.tolist(), **rec))


if __name__ == "__main__":
    banner("Figure 3 -- retention")
    main()
