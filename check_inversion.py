"""Verification of the Abate-Whitt Euler Laplace inversion.  Runtime: seconds.

Three analytic transforms with known inverses are inverted on the same time grid
the Gamma validation uses, so the reported accuracy is measured where the method
is actually applied and not on a more forgiving grid:

    1/[s(s+1)]              -> 1 - e^{-t}
    (1 + theta s)^{-k}/s    -> Gamma(k, theta) CDF, with (k, theta) taken from
                               the ACTUAL baseline moments of the model, so the
                               test transform has the shape the surrogate has
    (1 + s/2)^{-2}/s        -> Gamma(2, 1/2) CDF   (a second, sharper case)

The smallest grid time is where the nodes reach their largest modulus, so that
entry is reported separately: it is the only place the eigenfunction truncation
of the complex-s continuation could plausibly matter.
"""
import numpy as np
from scipy.special import gammainc
from _common import save, banner
from organoid_null.inversion import invert, A_EULER, N_TRUNC, M_EULER
from organoid_null.phenotype import T_HORIZON

TGRID = np.concatenate([np.linspace(0.02, 1.0, 45)[:-1], np.linspace(1.0, T_HORIZON, 60)])

def _baseline_gamma_shape():
    """(k, theta) of the two-moment surrogate at the conservative baseline."""
    from organoid_null.green import Chamber
    from organoid_null.reduced import Kinetics, one_target_moments
    ch = Chamber(0.0, [(0.5, 0.5), (0.25, 0.25)])
    _, T1, sd = one_target_moments(ch, (0.5, 0.5), (0.25, 0.25),
                                   Kinetics(kp=2.0, sigma=0.33, eps=0.02))
    T1, sd = float(T1), float(sd)
    return (T1 / sd) ** 2, sd ** 2 / T1


K_G, TH_G = _baseline_gamma_shape()

CASES = [
    ("1/[s(s+1)]",
     lambda s: 1.0 / (s * (s + 1.0)),
     lambda t: 1.0 - np.exp(-t)),
    (f"(1+theta s)^-k / s, k={K_G:.4f}, theta={TH_G:.4f}",
     lambda s: (1.0 + TH_G * s) ** (-K_G) / s,
     lambda t: gammainc(K_G, t / TH_G)),
    ("(1+s/2)^-2 / s",
     lambda s: (1.0 + 0.5 * s) ** (-2.0) / s,
     lambda t: gammainc(2.0, t / 0.5)),
]


def main():
    s_max = abs(A_EULER / (2 * TGRID[0]) + 1j * (N_TRUNC + M_EULER) * np.pi / TGRID[0])
    print(f"  nodes: A={A_EULER}, n_tr={N_TRUNC}, M={M_EULER}; "
          f"largest |s| = {s_max:.0f} at t = {TGRID[0]}")
    rows = []
    for name, F, exact in CASES:
        num = np.array([float(np.atleast_1d(invert(F, t))[0]) for t in TGRID])
        err = np.abs(num - exact(TGRID))
        j = int(np.argmax(err))
        rows.append(dict(case=name, max_err=float(err.max()),
                         t_at_max=float(TGRID[j]), err_at_tmin=float(err[0])))
        print(f"  {name:<36} max err {err.max():.2e} at t={TGRID[j]:.2f};"
              f"  err at t={TGRID[0]} is {err[0]:.2e}")
    worst = max(r["max_err"] for r in rows)
    ok = worst < 1e-7
    print(f"  worst over all three: {worst:.2e}  => {'PASS' if ok else 'FAIL'}")
    save("check_inversion", dict(A=A_EULER, n_trunc=N_TRUNC, M=M_EULER,
                                 s_max=float(s_max), t_min=float(TGRID[0]),
                                 cases=rows, worst=worst, **{"pass": bool(ok)}))


if __name__ == "__main__":
    banner("Check -- Laplace inversion")
    main()
