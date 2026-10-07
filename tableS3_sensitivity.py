"""Table S3: global uncertainty and sensitivity.

Two audits share one machinery and differ only in the output functional.

  left  -- 7 inputs (D, kappa', gamma, gamma^d, gammabar, ell, r0), output the
           eventual uptake of a single target in the concentric annulus, where
           the reduced amplitude is available in closed form;
  right -- 8 inputs (the same seven plus the remixing weight eta), output the
           target-1 uptake of the two-target remixing system of
           Corollary "remix", so that eta enters at all.

Both report a Latin-hypercube uncertainty band, partial rank correlation
coefficients, and first/total Sobol indices by the Saltelli-Jansen pick-freeze
estimator.  Seeds are fixed (LHS 12345, Sobol 54321); the Sobol sample is
2**15 = 32768 base points, so each index costs (p+2) * 2**15 evaluations.

Runtime: about one minute.
"""
import numpy as np
from scipy.special import iv, kv
from scipy.stats import qmc, rankdata
from _common import save, banner
from organoid_null.green import GAMMA_EULER

EPS, R_OUT = 0.02, 1.0
NU = -1.0 / np.log(EPS)
PBAR = np.array([0.7, 0.3])           # leading-order splitting of the pair
N_LHS, N_SOBOL = 4096, 2 ** 15

NAMES7 = ["D", "kappa'", "gamma", "gamma^d", "gammabar", "ell", "r0"]
LO7 = np.array([0.5, 0.5, 0.1, 0.1, 0.5, 0.6, 0.3])
HI7 = np.array([1.5, 2.5, 0.8, 2.0, 2.0, 1.4, 0.8])


def _annulus_amplitude(th, use_g):
    """Reduced eventual uptake 2 pi D A of one target in the annulus."""
    D, kp, gam, gd, gb, ell, r0 = th.T
    g = gb / (gd + gb) if use_g else 1.0
    lam = np.sqrt(gam / D)
    c = kv(1, lam * R_OUT) / iv(1, lam * R_OUT)
    Gam = (kv(0, lam * r0) + c * iv(0, lam * r0)) / (2 * np.pi * D)
    Rr = (-np.log(lam / 2) - GAMMA_EULER + c) / (2 * np.pi * D)
    Psi = D / (kp * g * ell)
    A = NU * Gam / (1 + 2 * np.pi * NU * D * (Rr - np.log(ell) / (2 * np.pi * D)) + NU * Psi)
    return 2 * np.pi * D * A


def uptake7(th):
    """Left audit: productive uptake through the full frequency filter at s=0."""
    return _annulus_amplitude(th, use_g=True)


def uptake8(th8):
    """Right audit: target-1 uptake of the two-target remixing closure.

    q is the per-encounter adsorption probability (g(0)=1, irreversible capture),
    Sigma = diag(sigma_1, sigma_2) the retention matrix, and
    P(eta) = q [(1-eta) 1 pibar^T + eta I] the convex remixing kernel; the
    productive uptake is (I - Sigma)(I - P Sigma)^{-1} q pibar.
    """
    th, eta = th8[:, :7], th8[:, 7]
    q = _annulus_amplitude(th, use_g=False)
    gd, gb = th[:, 3], th[:, 4]
    s1 = gd / (gd + gb)
    s2 = gd / (gd + gb / 4)           # heterogeneous retention: target 2 is stickier
    out = np.empty(len(q))
    eye = np.eye(2)
    for i in range(len(q)):
        S = np.diag([s1[i], s2[i]])
        P = q[i] * ((1 - eta[i]) * np.outer(PBAR, np.ones(2)) + eta[i] * eye)
        out[i] = ((eye - S) @ np.linalg.solve(eye - P @ S, q[i] * PBAR))[0]
    return out


def prcc(X, Y):
    Rx = np.column_stack([rankdata(X[:, i]) for i in range(X.shape[1])])
    Ry = rankdata(Y)
    out = []
    for i in range(X.shape[1]):
        Z = np.column_stack([np.ones(len(Ry)), np.delete(Rx, i, axis=1)])
        ei = Rx[:, i] - Z @ np.linalg.lstsq(Z, Rx[:, i], rcond=None)[0]
        ey = Ry - Z @ np.linalg.lstsq(Z, Ry, rcond=None)[0]
        out.append(float(np.corrcoef(ei, ey)[0, 1]))
    return np.array(out)


def sobol(f, lo, hi, n=N_SOBOL, seed=54321):
    """Saltelli-Jansen pick-freeze first-order and total indices."""
    p = len(lo)
    U = qmc.Sobol(d=2 * p, scramble=True, seed=seed).random(n)
    A = lo + U[:, :p] * (hi - lo)
    B = lo + U[:, p:] * (hi - lo)
    fA, fB = f(A), f(B)
    V = np.var(np.concatenate([fA, fB]), ddof=0)
    Si, STi = np.zeros(p), np.zeros(p)
    for i in range(p):
        AB = A.copy(); AB[:, i] = B[:, i]
        fAB = f(AB)
        Si[i] = np.mean(fB * (fAB - fA)) / V
        STi[i] = np.mean((fA - fAB) ** 2) / (2 * V)
    return Si, STi


def audit(label, f, lo, hi, names):
    lh = qmc.LatinHypercube(d=len(lo), seed=12345).random(N_LHS)
    X = lo + lh * (hi - lo)
    Y = f(X)
    q = np.quantile(Y, [0.025, 0.5, 0.975])
    P = prcc(X, Y)
    Si, STi = sobol(f, lo, hi)
    print(f"  {label}: LHS band 2.5/50/97.5 = {q[0]:.3f} {q[1]:.3f} {q[2]:.3f}")
    for i, nm in enumerate(names):
        print(f"    {nm:<10} PRCC {P[i]:+.3f}   S_i {Si[i]:.3f}   S_Ti {STi[i]:.3f}")
    print(f"    sum S_i = {Si.sum():.3f}   max(S_Ti - S_i) = {(STi - Si).max():.4f}")
    return dict(inputs=names, lo=lo.tolist(), hi=hi.tolist(),
                band=q.tolist(), prcc=P.tolist(), S=Si.tolist(), ST=STi.tolist(),
                sum_S=float(Si.sum()), max_interaction=float((STi - Si).max()),
                n_lhs=N_LHS, n_sobol=N_SOBOL)


if __name__ == "__main__":
    banner("Table S3 -- global uncertainty and sensitivity")
    left = audit("loss-channel audit (7 inputs)", uptake7, LO7, HI7, NAMES7)
    lo8 = np.append(LO7, 0.0); hi8 = np.append(HI7, 1.0)
    right = audit("closure audit (8 inputs)", uptake8, lo8, hi8, NAMES7 + ["eta"])
    save("tableS3_sensitivity", dict(loss_channel=left, closure=right))
