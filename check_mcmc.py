"""MCMC adequacy for the N = 20 hard-core layout law.  Runtime: about 12 minutes.

At N = 20 with d_min = 0.085 rejection sampling from the uniform law effectively
never terminates, so the ensemble falls back to Metropolis-within-Gibbs from a
grid seed.  40 sweeps is the production setting, and the question is whether
that is enough.

Adequacy is measured ON THE TARGET FUNCTIONAL -- the transport-only maturation
CV -- rather than on a mixing diagnostic, and from two deliberately contrasting
starts: a spread grid, and a tight cluster occupying only the lower-left corner.
A two-sample Kolmogorov-Smirnov test against the long run (400 sweeps from the
grid start) reports whether the 40-sweep distribution is distinguishable from
the converged one.

Note what this can and cannot show: agreement of both starts with the long run
is evidence that 40 sweeps suffices for THIS functional at this n; it is not a
proof of convergence, and the KS test at n = 70 has limited power against small
differences.  That limitation is stated in the manuscript rather than papered
over.
"""
import time
import numpy as np
from scipy.stats import ks_2samp
from _common import save, banner
from organoid_null.reduced import Kinetics, fluxes
from organoid_null.phenotype import phenotype

N, LO, HI, D_MIN, D_SRC = 20, 0.10, 0.90, 0.085, 0.10
EPS, GAM, PORT_CLEAR = 0.02, 0.4, 0.05
N_LAYOUT, SEED = 70, 99
SWEEPS = (2, 5, 40, 400)

_GRID = np.linspace(0.02, 3.0, 400)
_TABLE = phenotype(_GRID, 1.55, 1.15)
mt = lambda a: np.interp(a, _GRID, _TABLE[2])
LO_V, HI_V = np.array([LO, LO]), np.array([HI, HI])


def initial(kind):
    if kind == "grid":
        gx = np.linspace(LO, HI, 5); gy = np.linspace(LO, HI, 4)
    else:                                    # tight lower-left cluster
        gx = LO + 0.09 * np.arange(5); gy = LO + 0.09 * np.arange(4)
    return np.array([(a, b) for b in gy for a in gx], float)


def sweep(xs, rng, n):
    for _ in range(n):
        for i in range(N):
            p = LO_V + (HI_V - LO_V) * rng.random(2)
            other = np.delete(xs, i, axis=0)
            if np.linalg.norm(other - p, axis=1).min() > D_MIN:
                xs[i] = p
    return xs


def port(xs, rng):
    while True:
        x0 = PORT_CLEAR + (1 - 2 * PORT_CLEAR) * rng.random(2)
        if np.linalg.norm(xs - x0, axis=1).min() >= D_SRC:
            return tuple(x0)


def cvs(kind, n_sweeps, seed=SEED):
    kin = Kinetics(kp=2.0, sigma=1 / 3., eps=EPS)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_LAYOUT):
        xs = sweep(initial(kind).copy(), rng, n_sweeps)
        x0 = port(xs, rng)
        J = fluxes([tuple(v) for v in xs], x0, 0.0, GAM, kin)
        if J.min() <= 0:
            continue
        v = mt(J / J.mean())
        out.append(float(v.std() / v.mean()))
    return np.array(out)


if __name__ == "__main__":
    banner("Check -- Metropolis adequacy at N = 20")
    t0 = time.time()
    store = {(k, s): cvs(k, s) for k in ("grid", "corner") for s in SWEEPS}
    ref = store[("grid", 400)]
    print(f"  {'init':>8}{'sweeps':>8}{'n':>5}{'median':>10}{'10th':>9}{'90th':>9}"
          f"{'KS p vs grid/400':>19}")
    rows = []
    for k in ("grid", "corner"):
        for s in SWEEPS:
            v = store[(k, s)]
            p = float(ks_2samp(v, ref).pvalue)
            rows.append(dict(init=k, sweeps=s, n=len(v), median=float(np.median(v)),
                             p10=float(np.percentile(v, 10)),
                             p90=float(np.percentile(v, 90)), ks_p=p))
            print(f"  {k:>8}{s:>8}{len(v):>5}{np.median(v):>10.4f}"
                  f"{np.percentile(v,10):>9.4f}{np.percentile(v,90):>9.4f}{p:>19.3f}")
    ok = all(r["ks_p"] > 0.05 for r in rows if r["sweeps"] >= 40)
    print(f"  40-sweep runs indistinguishable from the long run: {'yes' if ok else 'NO'}")
    print(f"  ({time.time()-t0:.0f}s)")
    save("check_mcmc", dict(n_layouts=N_LAYOUT, seed=SEED, rows=rows,
                            **{"pass": bool(ok)}))
