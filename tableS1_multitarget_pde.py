"""Table S1: multi-target full-PDE verification of the reduced system.

For each (N, eps) the transformed full problem is solved by P2 finite elements
on three nested gmsh meshes (h = r/12, r/24, r/48 with r = eps*ell), Richardson
extrapolated at the observed order p = 2, and compared with the reduced
amplitude system at s in {0, 1, 5} and in both regimes (gamma = 0, 0.4).

The mesh-error estimate |J_extrap - J_finest| / max|J_extrap| is reported so
that closure error can be separated from discretisation error; the fitted
convergence slope p uses only entries exceeding ten times that estimate.

Requires gmsh and scikit-fem.  Runtime: several hours; the N = 10, eps = 0.0125
meshes are the expensive ones.  Pass --fast to run only the N = 2 block.
"""
import sys, time
import numpy as np
from _common import save, banner
from organoid_null.fem import build_mesh, fem_solve, reduced

L10 = [(0.17, 0.16), (0.50, 0.16), (0.83, 0.16), (0.16, 0.50), (0.39, 0.50),
       (0.61, 0.50), (0.84, 0.50), (0.17, 0.84), (0.50, 0.84), (0.83, 0.84)]
SUB = {2: [0, 9], 5: [0, 2, 4, 7, 9], 10: list(range(10))}
EPS = {2: [0.10, 0.05, 0.02], 5: [0.10, 0.05, 0.02], 10: [0.05, 0.025, 0.0125]}
X0 = (0.62, 0.30)
KP, GD, GB, ELL, D = 1.5, 0.5, 1.0, 1.0, 1.0
LEVELS = [12, 24, 48]
SVALS = [0.0, 1.0, 5.0]
REG = {"conservative": 0.0, "lossy": 0.4}


def run(Ns):
    out, t00 = [], time.time()
    for N in Ns:
        xs = [L10[i] for i in SUB[N]]
        dmin = min(float(np.hypot(a[0] - b[0], a[1] - b[1]))
                   for i, a in enumerate(xs) for b in xs[i + 1:])
        for eps in EPS[N]:
            r = eps * ELL
            h_far = 0.05 if eps >= 0.05 else 0.035
            levels = {}
            for f in LEVELS:
                t = time.time()
                mesh = build_mesh(xs, r, h_far, r / f, f"m{N}_{eps}_{f}")
                per = {}
                for reg, gam in REG.items():
                    for s in SVALS:
                        per[(reg, s)] = fem_solve(mesh, xs, r, X0, s, gam, eps,
                                                  ELL, KP, GD, GB, D)[0]
                levels[f] = dict(J=per, tris=int(mesh.t.shape[1]))
                print(f"  N={N} eps={eps} h=r/{f} tris={mesh.t.shape[1]} "
                      f"({time.time()-t:.0f}s, total {time.time()-t00:.0f}s)", flush=True)
            for reg, gam in REG.items():
                for s in SVALS:
                    J1, J2, J3 = (levels[f]["J"][(reg, s)] for f in LEVELS)
                    Jx = J3 + (J3 - J2) / 3.0
                    mesherr = float(np.abs(Jx - J3).max() / np.abs(Jx).max())
                    Jr = reduced(xs, X0, s, gam, eps, ELL, KP, GD, GB, D)
                    af, ar = Jx / Jx.sum(), Jr / Jr.sum()
                    out.append(dict(
                        N=N, eps=eps, regime=reg, s=s, dmin=dmin, d_over_r=dmin / r,
                        J_full=Jx.tolist(), J_red=Jr.tolist(),
                        J_levels={str(f): levels[f]["J"][(reg, s)].tolist() for f in LEVELS},
                        tris={str(f): levels[f]["tris"] for f in LEVELS},
                        Einf=float(np.abs(Jr - Jx).max() / np.abs(Jx).max()),
                        E1=float(np.abs(Jr - Jx).sum() / np.abs(Jx).sum()),
                        mesherr=mesherr,
                        alloc_err=float(np.abs(af - ar).max()),
                        cv_full=float(Jx.std(ddof=0) / Jx.mean()),
                        cv_red=float(Jr.std(ddof=0) / Jr.mean())))
            save("tableS1_multitarget_pde", out)
    return out


def summarise(out):
    """Collapse to the reported table: worst case over s and regime per (N, eps)."""
    from organoid_null.phenotype import phenotype
    rows = []
    keys = sorted({(r["N"], r["eps"]) for r in out}, key=lambda k: (k[0], -k[1]))
    for N, eps in keys:
        sel = [r for r in out if r["N"] == N and r["eps"] == eps]
        cvmt = 0.0
        for r in sel:                                  # phenotype CV uses s=0 only
            if r["s"] != 0.0:
                continue
            a = phenotype(np.array(r["J_full"]) / np.mean(r["J_full"]))[2]
            b = phenotype(np.array(r["J_red"]) / np.mean(r["J_red"]))[2]
            cvmt = max(cvmt, abs(b.std() / b.mean() - a.std() / a.mean())
                       / (a.std() / a.mean()))
        rows.append(dict(N=N, eps=eps, d_over_r=sel[0]["d_over_r"],
                         Einf=max(r["Einf"] for r in sel),
                         E1=max(r["E1"] for r in sel),
                         alloc=max(r["alloc_err"] for r in sel),
                         cv_pi_err=max(abs(r["cv_red"] - r["cv_full"]) / r["cv_full"]
                                       for r in sel),
                         cv_mt_err=cvmt,
                         mesherr=max(r["mesherr"] for r in sel)))
    for N in sorted({r["N"] for r in rows}):
        blk = [r for r in rows if r["N"] == N]
        e = np.array([r["Einf"] for r in blk]); x = np.array([r["eps"] for r in blk])
        m = e > 10 * np.array([r["mesherr"] for r in blk])
        p = float(np.polyfit(np.log(x[m]), np.log(e[m]), 1)[0]) if m.sum() > 1 else float("nan")
        # the same fit within each regime separately, which is what the quoted
        # range of slopes reports
        per = {}
        for reg in ("conservative", "lossy"):
            pts = [(r["eps"],
                    max(q["Einf"] for q in out if q["N"] == N and q["eps"] == r["eps"]
                        and q["regime"] == reg),
                    max(q["mesherr"] for q in out if q["N"] == N and q["eps"] == r["eps"]
                        and q["regime"] == reg)) for r in blk]
            pts = [q for q in pts if q[1] > 10 * q[2]]
            per[reg] = (float(np.polyfit(np.log([q[0] for q in pts]),
                                         np.log([q[1] for q in pts]), 1)[0])
                        if len(pts) > 1 else float("nan"))
        for r in blk:
            r["slope"] = p
            r["slope_by_regime"] = per
    return rows


if __name__ == "__main__":
    banner("Table S1 -- multi-target full-PDE verification")
    out = run([2] if "--fast" in sys.argv else [2, 5, 10])
    rows = summarise(out)
    allp = [v for r in rows for v in r["slope_by_regime"].values()]
    for r in rows:
        print(f"  N={r['N']:<3} eps={r['eps']:<7} d/r={r['d_over_r']:5.1f}  "
              f"Einf={r['Einf']:.1e}  E1={r['E1']:.1e}  alloc={r['alloc']:.1e}  "
              f"CVpi={r['cv_pi_err']:.1e}  CVmt={r['cv_mt_err']:.1e}  p={r['slope']:.2f}")
    print(f"  slope range over N and regime: {min(allp):.2f}-{max(allp):.2f}")
    save("tableS1_multitarget_pde", dict(raw=out, rows=rows,
                                         slope_range=[min(allp), max(allp)]))
