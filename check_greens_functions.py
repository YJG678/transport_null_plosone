"""Verification of the Green's-function layer.  Runtime: about two minutes.

Four independent checks:

1. Ewald against the reported values of the Neumann pseudo-Green's function of
   the unit square.  This pins the conservative branch to fixed digits.
2. Image-sum cutoff convergence in the lossy branch, at gamma small enough that
   a fixed cutoff fails.  The image sum converges like exp(-2 sqrt(gamma/D) M),
   so the required half-width scales as gamma^{-1/2}; a fixed M = 6 under-
   resolves G by 64% at gamma = 0.004, which is the failure this check exists
   to catch.
3. Eigenfunction-cutoff convergence of the complex-s continuation, and the
   residual truncation error at the largest node the Laplace inversion actually
   evaluates (|s| = 4112 at t_min = 0.02).
4. Agreement between the continuation at s = 0 and the direct evaluations, in
   both regimes.
5. The reason the Ewald splitting is needed at all: the raw eigenfunction
   series converges off the diagonal but diverges logarithmically on it, at the
   rate log 2 / 2pi per doubling of the cutoff.  Measured, not asserted.
"""
import numpy as np
from _common import save, banner
from organoid_null.green import (G0_ewald, R0_ewald, G_lossy, R_lossy, Chamber,
                                 _Mreq)
from organoid_null.inversion import A_EULER, N_TRUNC, M_EULER

XA, XB = (0.5, 0.5), (0.25, 0.25)
SERIES_CUTOFFS = (50, 100, 200, 400, 800)
G0_REF, R0_REF = -0.0137897250, -0.2085777932


def main():
    out = {}

    g0, r0 = G0_ewald(XA, XB), R0_ewald(XA)
    out["ewald"] = dict(G0=g0, R0=r0, G0_ref=G0_REF, R0_ref=R0_REF,
                        dG=abs(g0 - G0_REF), dR=abs(r0 - R0_REF))
    print(f"  1. Ewald   G0 = {g0:.10f}  (ref {G0_REF:.10f}, diff {abs(g0-G0_REF):.1e})")
    print(f"             R0 = {r0:.10f}  (ref {R0_REF:.10f}, diff {abs(r0-R0_REF):.1e})")

    print("  2. lossy image cutoff (converged value from M = 4 * M_required)")
    rows = []
    for gam in (0.4, 0.02, 0.004):
        lam = float(np.sqrt(gam))
        Mreq = _Mreq(lam)
        ref = _G_at(XA, XB, lam, 4 * Mreq)
        r = dict(gamma=gam, lam=lam, M_required=Mreq,
                 rel_at_M6=abs(_G_at(XA, XB, lam, 6) - ref) / abs(ref),
                 rel_at_Mreq=abs(_G_at(XA, XB, lam, Mreq) - ref) / abs(ref))
        rows.append(r)
        print(f"     gamma={gam:<7} lam={lam:.4f}  M_req={Mreq:<4} "
              f"rel.err at M=6: {r['rel_at_M6']:.2e}   at M=M_req: {r['rel_at_Mreq']:.2e}")
    out["image_cutoff"] = rows

    print("  3. eigenfunction cutoff of the complex-s continuation")
    s_max = abs(A_EULER / (2 * 0.02) + 1j * (N_TRUNC + M_EULER) * np.pi / 0.02)
    ref = Chamber(0.0, [XA, XB], kmax=800)
    rows = []
    for K in (100, 200, 400):
        ch = Chamber(0.0, [XA, XB], kmax=K)
        r = dict(kmax=K,
                 err_s5=float(abs(ch.G(XA, XA, 5.0) - ref.G(XA, XA, 5.0))),
                 err_s500=float(abs(ch.G(XA, XA, 500.0) - ref.G(XA, XA, 500.0))),
                 err_smax=float(abs(ch.G(XA, XA, s_max) - ref.G(XA, XA, s_max))))
        rows.append(r)
        print(f"     K_max={K:<5} |dR| at |s|=5: {r['err_s5']:.1e}"
              f"   at 500: {r['err_s500']:.1e}   at {s_max:.0f}: {r['err_smax']:.1e}")
    out["kmax"] = dict(s_max=float(s_max), rows=rows)

    print("  4. continuation at s = 0 against the direct evaluations")
    ch0 = Chamber(0.0, [XA, XB])
    d_cons = abs(ch0.G(XA, XB, 0.0) - G0_ewald(XA, XB))
    lam = float(np.sqrt(0.4))
    chL = Chamber(0.4, [XA, XB])
    d_loss = abs(chL.G(XA, XB, 0.0) - G_lossy(XA, XB, lam))
    d_lossR = abs(chL.G(XA, XA, 0.0) - R_lossy(XA, lam))
    print(f"     conservative off-diagonal {d_cons:.2e}")
    print(f"     lossy off-diagonal {d_loss:.2e},  diagonal {d_lossR:.2e}")
    out["consistency"] = dict(conservative=d_cons, lossy_off=d_loss, lossy_diag=d_lossR)

    print("  5. why the raw eigenfunction series needs an Ewald splitting")
    off = {K: raw_series(XA, XB, K) for K in (200, 300)}
    d_off = abs(off[300] - off[200])
    print(f"     off-diagonal G({XA},{XB}): cutoff 200 -> 300 changes by {d_off:.1e}")
    diag = [raw_series(XA, XA, K, gam=0.4, drop_zero=False)
            for K in SERIES_CUTOFFS]
    steps = [diag[i + 1] - diag[i] for i in range(len(diag) - 1)]
    print("     diagonal, lossy gamma=0.4, zero mode retained: "
          + ", ".join(f"{v:.4f}" for v in diag))
    print(f"     growth per doubling {np.mean(steps):.4f}"
          f"  (log 2 / 2pi = {np.log(2) / (2 * np.pi):.4f}) -- logarithmically divergent")
    out["series"] = dict(
        off_diagonal={str(k): v for k, v in off.items()}, off_diagonal_change=d_off,
        diagonal_cutoffs=list(SERIES_CUTOFFS), diagonal=diag,
        growth_per_doubling=float(np.mean(steps)),
        log2_over_2pi=float(np.log(2) / (2 * np.pi)))

    ok = (out["ewald"]["dG"] < 1e-9 and out["ewald"]["dR"] < 1e-9
          and max(r["rel_at_Mreq"] for r in out["image_cutoff"]) < 1e-8
          and max(d_cons, d_loss, d_lossR) < 1e-8
          # the divergence must actually be logarithmic at the predicted rate,
          # otherwise the justification for the Ewald splitting is not what we say
          and abs(np.mean(steps) - np.log(2) / (2 * np.pi)) < 2e-3)
    out["pass"] = bool(ok)
    print(f"  => {'PASS' if ok else 'FAIL'}")
    save("check_greens_functions", out)


def raw_series(x, z, K, gam=0.0, drop_zero=True):
    """The RAW Neumann eigenfunction series, truncated at |n_i| <= K.

    This is deliberately the thing the package does NOT use: off the diagonal it
    converges, but at x = z it diverges logarithmically, which is why the
    regular part is obtained by Ewald splitting instead.  It exists only so that
    the divergence can be measured rather than asserted.

    ``drop_zero`` removes the (0,0) mode, giving the conservative pseudo-Green's
    function; with gam > 0 the mode is finite and is retained.
    """
    n = np.arange(K + 1)
    N1, N2 = np.meshgrid(n, n, indexing="ij")
    c2 = (np.where(N1 == 0, 1.0, np.sqrt(2.0))
          * np.where(N2 == 0, 1.0, np.sqrt(2.0))) ** 2
    lam = np.pi ** 2 * (N1 ** 2 + N2 ** 2) + gam
    ph = (c2 * np.cos(N1 * np.pi * x[0]) * np.cos(N2 * np.pi * x[1])
          * np.cos(N1 * np.pi * z[0]) * np.cos(N2 * np.pi * z[1]))
    if drop_zero:
        m = ~((N1 == 0) & (N2 == 0))
        return float((ph[m] / lam[m]).sum())
    return float((ph / lam).sum())


def _G_at(x, z, lam, M):
    """G_lossy with the image half-width forced, for the cutoff study."""
    import organoid_null.green as g
    old = g._Mreq
    g._Mreq = lambda _l: M
    try:
        return g.G_lossy(x, z, lam)
    finally:
        g._Mreq = old


if __name__ == "__main__":
    banner("Check -- Green's functions")
    main()
