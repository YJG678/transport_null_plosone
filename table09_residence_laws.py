"""Table 9 and Figure 4: three residence laws at matched mean and matched sigma,
with the terminal states from EXACT Laplace inversion (not the two-moment
surrogate).  Runtime: about a minute."""
import numpy as np
from _common import save, banner, OUT
from organoid_null.green import Chamber
from organoid_null.reduced import Kinetics, fluxes_transform, conditional_moments
from organoid_null.inversion import (invert, phi_gamma, phi_exponential,
                                     phi_hyperexponential)
from organoid_null.phenotype import integrate, T_HORIZON

X1, X0 = [(0.5, 0.5)], (0.25, 0.25)
SIGMA = 0.60
LAWS = [("Gamma, CV=0.5", phi_gamma(0.5)),
        ("exponential, CV=1.0", phi_exponential()),
        ("long-tailed, CV=2.1", phi_hyperexponential())]
TGRID = np.concatenate([np.linspace(0.01, 1.0, 50)[:-1], np.linspace(1.0, T_HORIZON, 110)])


def reset_transform(ch, kin, phi):
    """Reset protocol with a non-Markovian residence law, eq. (7)."""
    unit = lambda s: 1.0 + 0 * s

    def F(s):
        Jads = fluxes_transform(ch, X1, X0, s, kin) * (s + kin.gb) / kin.gb
        p = phi(s)
        return (1 - SIGMA) * p * Jads / (1.0 - np.sum(SIGMA * p * Jads))
    return F


if __name__ == "__main__":
    banner("Table 9 / Figure 4 -- residence laws, exact inversion")
    ch = Chamber(0.0, X1 + [X0])
    kin_unit = Kinetics(kp=2.0, sigma=0.0, eps=0.02)     # g == 1: irreversible adsorption
    rows, curves = [], {"t": TGRID.tolist()}
    for name, phi in LAWS:
        F = reset_transform(ch, kin_unit, phi)
        pi = float(np.real(np.atleast_1d(F(0.0 + 0j)))[0])
        H = np.clip([float(invert(lambda s: F(s) / (s * pi), t)[0]) for t in TGRID], 0, None)
        f0 = np.real(np.atleast_1d(F(0.0 + 0j)))[0]
        h = 1e-3
        g = lambda s: np.real(np.atleast_1d(F(s))[0])
        d1 = (-g(2 * h) + 8 * g(h) - 8 * g(-h) + g(-2 * h)) / (12 * h)
        d2 = (-g(2 * h) + 16 * g(h) - 30 * f0 + 16 * g(-h) - g(-2 * h)) / (12 * h * h)
        T1 = -d1 / f0
        sd = float(np.sqrt(d2 / f0 - T1 ** 2))
        Y = integrate(lambda t: np.array([pi * np.interp(t, TGRID, H)]), T_HORIZON, nstep=900)
        traj_t = np.linspace(0, T_HORIZON, 41)
        traj = [0.60 if x == 0 else float(np.atleast_1d(
            integrate(lambda t: np.array([pi * np.interp(t, TGRID, H)]), max(x, 1e-6),
                      nstep=400)[4])[0]) for x in traj_t]
        rows.append(dict(law=name, T_tot=float(T1), sd=sd,
                         mt=float(Y[2][0]), ds=float(Y[4][0])))
        curves[name] = dict(H=H.tolist(), pi=pi, traj_t=traj_t.tolist(), traj=traj)
        print(f"  {name:<22} T={T1:.4f} sd={sd:.4f} mt={Y[2][0]:.4f} ds={Y[4][0]:.4f}")
    # Surface-only compound-geometric time: a Geometric(1 - sigma) number of
    # residence periods, before the diffusive search is restored.  Mean
    # <tau>/(1-sigma) = 2.5 for all three laws; they differ in dispersion, and
    # the text quotes both the variance and Pr(T_surf < 1).
    #
    # Variance of a compound geometric with unit-mean summands:
    #   Var = CV_tau^2 / p + (1 - p) / p^2,   p = 1 - sigma
    p_succ = 1 - SIGMA
    cvs = [0.5, 1.0, 2.1]
    var = [cv ** 2 / p_succ + (1 - p_succ) / p_succ ** 2 for cv in cvs]
    # Pr(T_surf < 1) by inverting the compound-geometric transform
    #   phi_surf(s) = p phi(s) / (1 - (1-p) phi(s))
    # at t = 1.  For the exponential law this is exactly 1 - exp(-p), which the
    # run prints alongside as an independent check on the inversion.
    pr1 = []
    for (name, phi) in LAWS:
        surf = lambda s, phi=phi: p_succ * phi(s) / (1.0 - (1.0 - p_succ) * phi(s))
        pr1.append(float(np.atleast_1d(invert(lambda s: surf(s) / s, 1.0))[0]))
    print("  surface-only mean %.3f; variances %s" % (
        1.0 / p_succ, " ".join(f"{v:.3f}" for v in var)))
    print("  Pr(T_surf < 1): " + " ".join(f"{v:.3f}" for v in pr1)
          + f"   (exponential closed form 1-exp(-{p_succ:g}) = "
            f"{1 - np.exp(-p_succ):.4f})")
    save("table09_residence_laws", dict(
        rows=rows, surface_variances=var, surface_mean=1.0 / p_succ,
        surface_cv_tau=cvs, surface_pr_below_1=pr1,
        surface_pr_exponential_exact=float(1 - np.exp(-p_succ))))
    np.savez(str(OUT / "residence_curves.npz"),
             **{f"{k}": np.asarray(v if not isinstance(v, dict) else v["H"])
                for k, v in curves.items() if k != "t"},
             t=TGRID, **{f"traj_{i}": np.asarray(curves[n]["traj"])
                         for i, (n, _) in enumerate(LAWS)},
             traj_t=np.linspace(0, T_HORIZON, 41),
             pis=np.array([curves[n]["pi"] for n, _ in LAWS]))
