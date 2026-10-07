"""Table 5: full-PDE verification on layouts drawn from the reported ensemble.

Table S1 uses one fixed layout with d_min/(eps*ell) up to 47.  That does not
establish the accuracy of the ENSEMBLE statistics of section 4, whose layouts
are random and have d_min/(eps*ell) near 6.  This script repeats the comparison
at s = 0, eps = 0.02, on the five layouts selected by select_pde_layouts.py,
in the exact kinetics of the null model (kappa' = 2, sigma = 1/3).

The N = 20 row uses mesh levels r/12, r/18, r/24 rather than r/12, r/24, r/48,
because the finest level exhausts memory at that target count; Richardson
extrapolation on the finest pair uses the same p = 2 with the non-integer
refinement ratio.

Requires gmsh and scikit-fem.  Runtime: about 12 minutes.
"""
import time
import numpy as np
from _common import save, load, banner
from organoid_null.fem import build_mesh, fem_solve, reduced
from organoid_null.phenotype import phenotype

EPS, ELL, KP, GB, D = 0.02, 1.0, 2.0, 1.0, 1.0
SIGMA = 1 / 3.
GD = GB * SIGMA / (1 - SIGMA)
REG = (("lossy", 0.4), ("conservative", 0.0))
LEVELS_DEFAULT = (12, 24, 48)
LEVELS_N20 = (12, 18, 24)

_GRID = np.linspace(0.02, 3.0, 400)
_TABLE = phenotype(_GRID, 1.55, 1.15)
mt = lambda a: np.interp(a, _GRID, _TABLE[2])

ORDER = [("median", "median of the ensemble"),
         ("p90", "90th percentile"),
         ("max", "ensemble maximum"),
         ("N20", "N=20, median layout"),
         ("smalldsrc", "d_src=0.05 layout")]


def one_layout(key, spec, t0):
    xs = [tuple(p) for p in spec["xs"]]
    x0 = tuple(spec["x0"])
    r = EPS * ELL
    levels = LEVELS_N20 if key == "N20" else LEVELS_DEFAULT
    lev = {}
    for f in levels:
        mesh = build_mesh(xs, r, 0.035, r / f, f"e{key}{f}")
        for reg, gam in REG:
            lev[(f, reg)] = fem_solve(mesh, xs, r, x0, 0.0, gam, EPS, ELL,
                                      KP, GD, GB, D)[0]
        print(f"  {key} h=r/{f} tris={mesh.t.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    # Richardson at p = 2 on the finest pair; ratio need not be 2
    fA, fB = levels[-2], levels[-1]
    rho = fB / fA
    rows = []
    for reg, gam in REG:
        J2, J3 = lev[(fA, reg)], lev[(fB, reg)]
        Jx = J3 + (J3 - J2) / (rho ** 2 - 1.0)
        Jr = reduced(xs, x0, 0.0, gam, EPS, ELL, KP, GD, GB, D)
        a, b = mt(Jx / Jx.mean()), mt(Jr / Jr.mean())
        cf, cr = float(a.std() / a.mean()), float(b.std() / b.mean())
        af, ar = Jx / Jx.sum(), Jr / Jr.sum()
        rows.append(dict(
            layout=key, regime=reg, cv_target=spec["cv"], dmin=spec["dmin"],
            J_full=Jx.tolist(), J_red=Jr.tolist(),
            total_full=float(Jx.sum()), total_red=float(Jr.sum()),
            d_over_r=spec["dmin"] / r, dsrc_over_r=spec["portmin"] / r,
            levels=list(levels), richardson_ratio=float(rho),
            Einf=float(np.abs(Jr - Jx).max() / np.abs(Jx).max()),
            E1=float(np.abs(Jr - Jx).sum() / np.abs(Jx).sum()),
            mesherr=float(np.abs(Jx - J3).max() / np.abs(Jx).max()),
            alloc=float(np.abs(af - ar).max()),
            cv_pi_full=float(Jx.std() / Jx.mean()), cv_pi_red=float(Jr.std() / Jr.mean()),
            cv_mt_full=cf, cv_mt_red=cr, cv_mt_relerr=abs(cf - cr) / cf))
        print(f"    {key:>10} {reg:<13} Einf={rows[-1]['Einf']:.2e} "
              f"mesh={rows[-1]['mesherr']:.2e} CVmt full {cf:.4f} red {cr:.4f} "
              f"rel {rows[-1]['cv_mt_relerr']:.2e}", flush=True)
    return rows


if __name__ == "__main__":
    banner("Table 5 -- full PDE on ensemble layouts")
    L = load("pde_layouts")
    out, t0 = [], time.time()
    for key, _ in ORDER:
        out += one_layout(key, L[key], t0)
        save("table05_ensemble_pde", out)

    print("\n  reported table (errors are the worse of the two regimes,")
    print("  CV columns are the lossy regime, which is what the null model uses):")
    for key, label in ORDER:
        sel = [r for r in out if r["layout"] == key]
        lossy = [r for r in sel if r["regime"] == "lossy"][0]
        print(f"  {label:<26} d/r={lossy['d_over_r']:4.1f}  dsrc/r={lossy['dsrc_over_r']:4.1f}"
              f"  Einf={max(r['Einf'] for r in sel):.1e}"
              f"  CV full {lossy['cv_mt_full']:.4f}  red {lossy['cv_mt_red']:.4f}"
              f"  rel {lossy['cv_mt_relerr']:.1e}")
    save("table05_ensemble_pde", out)
