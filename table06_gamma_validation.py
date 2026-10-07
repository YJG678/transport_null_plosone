"""Table 6: two-moment Gamma reconstruction against exact Laplace inversion.

The reduced transform is continued to complex s by the eigenfunction expansion
of ``organoid_null.green.Chamber`` (no complex Bessel functions are needed), and
inverted by the Abate-Whitt Euler algorithm.  Every evaluation point lies on the
vertical line Re s = A/(2t) > 0, strictly to the right of every singularity of
the transform.

Three families are covered:
  * Markovian boundary, N = 1, both regimes, sigma in {0.33, 0.60, 0.90};
  * Markovian boundary, N = 10, both regimes, sigma in {0.33, 0.90};
  * resetting with a non-Markovian residence law (Gamma CV 0.5, exponential,
    hyperexponential CV 2.1), N = 1.

Reported per organoid and then worst-cased over the family:
  E_CDF = sup_t |H^ref - H^Gamma|                       (absolute, in CDF units)
  E_J   = int|J^ref - J^Gamma| / int J^ref              (relative L1)
plus the relative errors in the terminal maturation, terminal disease score and,
for N = 10, the between-organoid CV.

Runtime: about 25 minutes (the N=10 reset-free cases dominate).
"""
import time
import numpy as np
from _common import save, banner
from organoid_null.green import Chamber
from organoid_null.reduced import Kinetics, fluxes_transform
from organoid_null.inversion import (invert, phi_exponential, phi_gamma,
                                     phi_hyperexponential)
from organoid_null.phenotype import gamma_cdf, integrate, T_HORIZON, A_DOSE

X1 = [(0.5, 0.5)]
X0 = (0.25, 0.25)
L10 = [(0.20, 0.25), (0.42, 0.18), (0.68, 0.28), (0.85, 0.52), (0.25, 0.55),
       (0.50, 0.50), (0.72, 0.68), (0.30, 0.82), (0.55, 0.80), (0.82, 0.85)]
SRC10 = (0.10, 0.12)
# dense early grid: the CDF rises fastest there, and sup_t is usually attained
# before t = 1.
TGRID = np.concatenate([np.linspace(0.02, 1.0, 45)[:-1], np.linspace(1.0, T_HORIZON, 60)])


def quantiles(tg, H, qs=(0.1, 0.5, 0.9)):
    H = np.clip(H, 0, None)
    return [float(np.interp(q, H, tg)) if H[-1] > q else float("nan") for q in qs]


def moments_of(F, h=1e-3):
    f0 = np.real(F(0.0 + 0j))
    d1 = (-np.real(F(2 * h + 0j)) + 8 * np.real(F(h + 0j))
          - 8 * np.real(F(-h + 0j)) + np.real(F(-2 * h + 0j))) / (12 * h)
    d2 = (-np.real(F(2 * h + 0j)) + 16 * np.real(F(h + 0j)) - 30 * f0
          + 16 * np.real(F(-h + 0j)) - np.real(F(-2 * h + 0j))) / (12 * h * h)
    T1 = -d1 / f0
    return f0, T1, np.sqrt(np.maximum(d2 / f0 - T1 ** 2, 1e-14))


def one_case(label, gam, xs, x0, sigma, phi=None):
    """phi=None -> Markovian boundary (continued search); otherwise reset."""
    ch = Chamber(gam, list(xs) + [x0])
    N = len(xs)
    if phi is None:
        kin = Kinetics(kp=2.0, sigma=sigma, eps=0.02)
        F = lambda s: fluxes_transform(ch, xs, x0, s, kin)
    else:
        # Reset protocol with a general residence law phi:
        #   J(s) = (1-sigma) phi(s) Jads(s) / (1 - sum_k sigma phi(s) Jads_k(s))
        # Jads is the ADSORPTION flux, recovered by undoing the gb/(s+gb) factor
        # and setting the boundary filter to unity.
        kin1 = Kinetics(kp=2.0, sigma=0.0, eps=0.02)
        kin1.g = lambda s: 1.0 + 0 * s
        kin1.Psi = lambda s: kin1.D / (kin1.kp * 1.0 * kin1.ell)

        def F(s):
            Jads = fluxes_transform(ch, xs, x0, s, kin1) * (s + kin1.gb) / kin1.gb
            p = phi(s)
            return (1 - sigma) * p * Jads / (1.0 - np.sum(sigma * p * Jads))

    pi = np.real(F(0.0 + 0j))
    Href = np.array([invert(lambda s: F(s) / (s * pi), t) for t in TGRID])
    T1 = np.zeros(N); sd = np.zeros(N)
    for j in range(N):
        _, T1[j], sd[j] = moments_of(lambda s: np.atleast_1d(F(s))[j])
    Hg = np.stack([gamma_cdf(T1[j], sd[j])(TGRID) for j in range(N)], 1)

    E_cdf = np.abs(Href - Hg).max(axis=0)
    E_J = (np.trapezoid(np.abs(Href - Hg), TGRID, axis=0)
           / np.trapezoid(np.clip(Href, 0, None), TGRID, axis=0))

    interp = lambda H: (lambda t: np.array([np.interp(t, TGRID, H[:, j]) for j in range(N)]))
    Yr = integrate(lambda t: A_DOSE * pi * interp(Href)(t), T_HORIZON, nstep=900)
    Yg = integrate(lambda t: A_DOSE * pi * interp(Hg)(t), T_HORIZON, nstep=900)

    rec = dict(label=label, gamma=gam, N=N, sigma=sigma,
               pi=np.atleast_1d(pi).tolist(), T1=T1.tolist(), sd=sd.tolist(),
               E_CDF=E_cdf.tolist(), E_J=E_J.tolist(),
               q_ref=[quantiles(TGRID, Href[:, j]) for j in range(N)],
               q_gam=[quantiles(TGRID, Hg[:, j]) for j in range(N)],
               mt_ref=np.atleast_1d(Yr[2]).tolist(), mt_gam=np.atleast_1d(Yg[2]).tolist(),
               ds_ref=np.atleast_1d(Yr[4]).tolist(), ds_gam=np.atleast_1d(Yg[4]).tolist())
    for k, nm in ((2, "mt"), (4, "ds")):
        a, b = np.atleast_1d(Yr[k]), np.atleast_1d(Yg[k])
        rec[f"rel_{nm}"] = float(np.abs(b - a).max() / np.abs(a).max())
        if N > 1:
            ca, cb = a.std() / a.mean(), b.std() / b.mean()
            rec[f"cv_{nm}_ref"], rec[f"cv_{nm}_gam"] = float(ca), float(cb)
            rec[f"rel_cv_{nm}"] = float(abs(cb - ca) / ca)
    return rec


def worst(recs, key):
    v = [r[key] for r in recs if key in r]
    if not v:
        return None
    return float(max(np.max(x) for x in v))


if __name__ == "__main__":
    banner("Table 6 -- Gamma reconstruction against exact inversion")
    t0 = time.time()
    markov = []
    for gam, rl in ((0.0, "conservative"), (0.4, "lossy")):
        for sigma in (0.33, 0.60, 0.90):
            markov.append(one_case(f"{rl}-N1-sig{sigma}", gam, X1, X0, sigma))
            print(f"  {markov[-1]['label']:<26} E_CDF={max(markov[-1]['E_CDF']):.3e}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    for gam, rl in ((0.0, "conservative"), (0.4, "lossy")):
        for sigma in (0.33, 0.90):
            markov.append(one_case(f"{rl}-N10-sig{sigma}", gam, L10, SRC10, sigma))
            r = markov[-1]
            print(f"  {r['label']:<26} E_CDF={max(r['E_CDF']):.3e}"
                  f"  CV ref {r['cv_mt_ref']:.4f} vs gam {r['cv_mt_gam']:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    reset = []
    for nm, ph in (("gamma0.5", phi_gamma(0.5)), ("exponential", phi_exponential()),
                   ("hyperexp", phi_hyperexponential())):
        reset.append(one_case(f"reset-N1-{nm}", 0.0, X1, X0, 0.60, phi=ph))
        print(f"  {reset[-1]['label']:<26} E_CDF={max(reset[-1]['E_CDF']):.3e}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    summary = {}
    for nm, recs in (("markovian", markov), ("reset", reset)):
        summary[nm] = dict(E_CDF=worst(recs, "E_CDF"), E_J=worst(recs, "E_J"),
                           rel_mt=worst(recs, "rel_mt"), rel_ds=worst(recs, "rel_ds"),
                           rel_cv=worst(recs, "rel_cv_mt"))
        s = summary[nm]
        print(f"  worst case, {nm}: E_CDF={s['E_CDF']:.3f}  E_J={s['E_J']:.3f}"
              f"  |d mt|={100*s['rel_mt']:.2f}%  |d ds|={100*s['rel_ds']:.2f}%"
              + (f"  |d CV|={100*s['rel_cv']:.2f}%" if s["rel_cv"] else ""))
    save("table06_gamma_validation", dict(cases=markov + reset, summary=summary))
    print(f"  total {time.time()-t0:.0f}s")
