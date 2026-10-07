"""Section 5.5: where remixing matters.

Corollary "remix" is exercised on two disks at (0.35, 0.5) and (0.65, 0.5) with
the source at (0.20, 0.50) and kappa' = 2.0.  Three things are produced:

  * the target-1 uptake across eta in {0, 1/4, 1/2, 3/4, 1}, both with the exact
    re-adsorption matrix P(0) and with its leading-order value I, and the
    endpoint gap of each;
  * the fraction of the idealized (P = I) gap that the exact P recovers, as eps
    decreases, showing that the correction is O(nu) and therefore decays only
    logarithmically -- it is not negligible at any experimentally relevant size;
  * the homogeneous-retention case sigma = (0.5, 0.5), where the corollary
    predicts uptake independent of eta up to O(nu).

Runtime: seconds.
"""
import numpy as np
from _common import save, banner
from organoid_null.remixing import build, uptake

XS = [(0.35, 0.5), (0.65, 0.5)]
X0 = (0.20, 0.50)
SIGMA = np.array([0.1, 0.9])
ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EPS_SWEEP = [0.10, 0.05, 0.02, 0.01, 1e-3, 1e-4, 1e-6, 1e-10]

if __name__ == "__main__":
    banner("Section 5.5 -- remixing")
    pibar, P, nu = build(XS, X0)
    colsum = P.sum(0)
    print(f"  eps=0.02 (nu={nu:.4f});  pibar = ({pibar[0]:.4f}, {pibar[1]:.4f})")
    print(f"  P(0) = [[{P[0,0]:.4f}, {P[0,1]:.4f}], [{P[1,0]:.4f}, {P[1,1]:.4f}]]")
    print(f"  column sums (must be 1): {colsum[0]:.12f}, {colsum[1]:.12f}")

    exact = [float(uptake(e, SIGMA, pibar, P)[0]) for e in ETAS]
    ideal = [float(uptake(e, SIGMA, pibar, np.eye(2))[0]) for e in ETAS]
    print("  target-1 uptake, heterogeneous retention sigma=(0.1,0.9):")
    print("     eta      exact P      P = I")
    for e, a, b in zip(ETAS, exact, ideal):
        print(f"    {e:5.2f} {a:11.4f} {b:11.4f}")
    gap_exact = exact[0] - exact[-1]
    gap_ideal = ideal[0] - ideal[-1]
    print(f"  endpoint gap: exact {gap_exact:.4f}, idealized {gap_ideal:.4f}"
          f"  ({gap_exact/gap_ideal:.3f} of the idealized gap)")

    print("  how the gap converges as eps -> 0:")
    sweep = []
    for eps in EPS_SWEEP:
        pb, Pe, nue = build(XS, X0, eps=eps)
        a = float(uptake(0, SIGMA, pb, Pe)[0] - uptake(1, SIGMA, pb, Pe)[0])
        b = float(uptake(0, SIGMA, pb, np.eye(2))[0] - uptake(1, SIGMA, pb, np.eye(2))[0])
        sweep.append(dict(eps=eps, nu=float(nue), P12=float(Pe[0, 1]),
                          gap_exact=a, gap_ideal=b, fraction=a / b))
        print(f"    eps={eps:9.0e}  nu={nue:.4f}  P_12={Pe[0,1]:.4f}"
              f"  gap {a:.4f} / {b:.4f} = {a/b:.3f}")

    hom = [float(uptake(e, np.array([0.5, 0.5]), pibar, P)[0]) for e in (0.0, 0.5, 1.0)]
    print(f"  homogeneous sigma=(0.5,0.5): uptake drifts {hom[0]:.4f} -> {hom[-1]:.4f}"
          f"  (pibar_1 = {pibar[0]:.4f}); the O(nu) residual of the corollary")

    save("section55_remixing", dict(
        pibar=pibar.tolist(), P=P.tolist(), nu=float(nu),
        column_sums=colsum.tolist(), etas=ETAS,
        uptake_exact=exact, uptake_ideal=ideal,
        gap_exact=gap_exact, gap_ideal=gap_ideal,
        eps_sweep=sweep, homogeneous=hom))
