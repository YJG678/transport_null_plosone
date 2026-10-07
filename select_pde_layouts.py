"""Select the five layouts used in Table 5 and write them to outputs/.

Three come from the reported ensemble of Table 1 (d_src = 0.10, 2000 layouts,
seed 2024), picked at the median, 90th percentile and maximum of the
transport-only maturation CV.  Two come from the design cells that set the
endpoints of the sweep range in Table 2: the N = 20 cell (d_min = 0.085) and the
d_src = 0.05 cell, each picked at the median CV of its own draw.

Only the layout coordinates are produced here; the full-PDE comparison is
``table05_ensemble_pde.py``.  Runtime: about 8 minutes.
"""
import json
import numpy as np
from _common import save, banner, OUT
from organoid_null.reduced import Kinetics, fluxes
from organoid_null.phenotype import phenotype
from organoid_null.ensembles import LayoutLaw

GAMMA, EPS, SEED = 0.4, 0.02, 2024
_GRID = np.linspace(0.02, 3.0, 400)
_TABLE = phenotype(_GRID, 1.55, 1.15)
mt = lambda a: np.interp(a, _GRID, _TABLE[2])


def sweep(n, N=10, d_min=0.12, d_src=0.10, seed=SEED, need_cv=True):
    """Draw n layouts and return (cv, xs, x0) for each; cv is None if not needed."""
    kin = Kinetics(kp=2.0, sigma=1 / 3., eps=EPS)
    law = LayoutLaw(N=N, d_wall=0.10, d_min=d_min, d_src=d_src)
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        xs, x0 = law.draw(rng)
        if not need_cv:
            recs.append((None, xs, x0))
            continue
        J = fluxes(xs, x0, 0.0, GAMMA, kin)
        if J.min() <= 0:
            continue
        v = mt(J / J.mean())
        recs.append((float(v.std() / v.mean()), xs, x0))
    return recs


def cv_of(xs, x0):
    kin = Kinetics(kp=2.0, sigma=1 / 3., eps=EPS)
    J = fluxes(xs, x0, 0.0, GAMMA, kin)
    v = mt(J / J.mean())
    return float(v.std() / v.mean())


def describe(cv, xs, x0):
    P = np.asarray(xs)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    return dict(cv=cv, xs=[list(map(float, p)) for p in xs], x0=list(map(float, x0)),
                dmin=float(d[np.triu_indices(len(xs), 1)].min()),
                portmin=float(min(np.hypot(p[0] - x0[0], p[1] - x0[1]) for p in xs)))


if __name__ == "__main__":
    banner("Layout selection for Table 5")
    out = {}

    recs = sweep(2000)
    order = np.argsort([r[0] for r in recs])
    for key, idx in (("median", order[len(order) // 2]),
                     ("p90", order[int(0.9 * len(order))]),
                     ("max", order[-1])):
        out[key] = describe(*recs[idx])

    # the tightest source separation in the d_src = 0.05 sweep cell (n = 120,
    # the sweep's own layout count), not its median
    recs = sweep(120, d_src=0.05, need_cv=False)
    sep = [min(np.hypot(p[0] - x0[0], p[1] - x0[1]) for p in xs) for _, xs, x0 in recs]
    _, xs, x0 = recs[int(np.argmin(sep))]
    out["smalldsrc"] = describe(cv_of(xs, x0), xs, x0)

    recs = sweep(60, N=20, d_min=0.085)
    order = np.argsort([r[0] for r in recs])
    out["N20"] = describe(*recs[order[len(order) // 2]])

    for k, v in out.items():
        print(f"  {k:<10} CV={v['cv']:.4f}  d_min={v['dmin']:.4f} ({v['dmin']/(EPS):.1f} r)"
              f"  d_src={v['portmin']:.4f} ({v['portmin']/(EPS):.1f} r)")
    save("pde_layouts", out)
