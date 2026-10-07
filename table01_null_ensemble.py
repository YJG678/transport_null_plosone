"""Table 1 (and the point-source row of Table 3): transport-only spread over the
random-layout ensemble, and Figure 2 panel C data.

Runtime: about 4 minutes.
"""
import numpy as np
from _common import save, banner
from organoid_null.reduced import Kinetics, fluxes_multi
from organoid_null.phenotype import phenotype
from organoid_null.ensembles import LayoutLaw, bootstrap_percentile

N_LAYOUTS, SEED, GAMMA, EPS = 2000, 2024, 0.4, 0.02
N_BOOT = 4000          # Table 2 uses 2000; Table 1 uses 4000, as stated in section 4
RAMP_T1, RAMP_SD = 1.55, 1.15


# The common-ramp phenotype map depends on the plateau alone, so it is
# tabulated once and interpolated.  Section 4 item 5 quantifies what the common
# ramp costs relative to per-organoid ramps.
_GRID = np.linspace(0.02, 3.0, 400)
_TABLE = phenotype(_GRID, RAMP_T1, RAMP_SD)


def _map(a, kind):
    return np.interp(a, _GRID, _TABLE[2 if kind == "mt" else 4])


def run(d_src=0.10, n=N_LAYOUTS, seed=SEED):
    kin = Kinetics(kp=2.0, sigma=1 / 3., eps=EPS)
    law = LayoutLaw(N=10, d_wall=0.10, d_min=0.12, d_src=d_src)
    rng = np.random.default_rng(seed)
    K = {k: [] for k in ("cv_pi", "ratio", "cv_mt", "cv_ds", "cv_mt_uniform", "dose_ratio")}
    dropped = 0
    for _ in range(n):
        xs, x0 = law.draw(rng)
        Jp, Ju = fluxes_multi(xs, [x0, "uniform"], 0.0, GAMMA, kin)
        if Jp.min() <= 0 or Ju.min() <= 0:
            dropped += 1
            continue
        K["dose_ratio"].append(Ju.sum() / Jp.sum())
        p, u = Jp / Jp.mean(), Ju / Ju.mean()
        K["cv_pi"].append(p.std() / p.mean())
        K["ratio"].append(p.max() / p.min())
        for kind, key in (("mt", "cv_mt"), ("ds", "cv_ds")):
            v = _map(p, kind)
            K[key].append(v.std() / v.mean())
        w = _map(u, "mt")
        K["cv_mt_uniform"].append(w.std() / w.mean())
    return {k: np.asarray(v) for k, v in K.items()}, dropped


if __name__ == "__main__":
    banner("Table 1 / Table 3 -- null ensemble")
    K, dropped = run()
    n = len(K["cv_mt"])
    out = dict(n_retained=n, n_dropped=dropped, d_src=0.10, seed=SEED, n_boot=N_BOOT,
               dose_ratio_mean=float(K["dose_ratio"].mean()),
               dose_ratio_range=[float(K["dose_ratio"].min()), float(K["dose_ratio"].max())],
               ratio_of_medians=float(np.median(K["cv_mt"]) / np.median(K["cv_mt_uniform"])))
    for key in ("cv_pi", "ratio", "cv_mt", "cv_ds", "cv_mt_uniform"):
        out[key] = {f"p{q}": list(bootstrap_percentile(K[key], q, B=N_BOOT))
                    for q in (10, 50, 90)}
    print(f"  {n} layouts retained, {dropped} dropped")
    for key in ("cv_pi", "ratio", "cv_mt", "cv_ds", "cv_mt_uniform"):
        row = " ".join("%8.3f [%.3f,%.3f]" % tuple(out[key][f"p{q}"]) for q in (10, 50, 90))
        print(f"  {key:<14} {row}")
    print("  point/uniform ratio of medians %.2f" % out["ratio_of_medians"])
    save("table01_null_ensemble", out)
    np.savez(str(__import__("_common").OUT / "null_ensemble_raw.npz"), **K)
