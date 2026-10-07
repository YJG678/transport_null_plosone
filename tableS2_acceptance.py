"""Table S2 and Figure S1: acceptance probability against batch size, and its
non-degenerate large-batch limit.  Runtime: about 6 minutes.

The limit is Pr_H{ p(H) >= r_min AND mu(H) >= Ybar_min } over the SHARED
line and batch effects H, which do not average out with batch size.  The
marginal pass probability E[p(H)] is not the relevant threshold.
"""
import numpy as np
from scipy.stats import norm
from _common import save, banner
from organoid_null.green import Chamber
from organoid_null.reduced import Kinetics, one_target_moments
from organoid_null.phenotype import flux_of_t, integrate, T_HORIZON, PARAMS

X1, X0 = [(0.5, 0.5)], (0.25, 0.25)
MT_MIN, DS_MAX, MBAR_MIN, R_MIN = 0.58, 0.18, 0.62, 0.80
S_LINE, SEED, N_BATCH = 0.05, 24680, 40000
NB_GRID = [5, 10, 20, 50, 100]

_z = np.linspace(-4, 4, 2001)
_w = norm.pdf(_z); _w /= np.trapezoid(_w, _z)
V4 = float(np.trapezoid(_z ** 2 * _w, _z) - np.trapezoid(_z * _w, _z) ** 2)
_N = lambda c: float(np.trapezoid(np.exp(c * _z) * _w, _z))

#: signed loadings of eq. (33): +1 raises reactivity/maturation, -1 lowers
#: desorption and disease-promoting rates.
P_PLUS = ["kappa'", "gbar", "ell", "r_grow", "k_PD", "b_PD", "k_DM", "b_DM",
          "k_apclr", "alpha_R"]
P_MINUS = ["gamma_d", "d_0", "k_stress", "alpha_Q", "Theta"]

_ch = Chamber(0.0, X1 + [X0])


def _outputs_of_quality(q):
    """Terminal (mt, ds) as a function of the common quality multiplier q."""
    up = np.asarray(q, float); dn = 1.0 / up
    kin = Kinetics(kp=2.0 * up, ell=1.0 * up, gb=1.0 * up, sigma=0.33, eps=0.02)
    kin.gd = (0.33 / 0.67) * 1.0 * dn
    _, T1, sd = one_target_moments(_ch, X1[0], X0, kin)
    Y = integrate(flux_of_t(np.ones_like(up), T1, sd), T_HORIZON, nstep=900,
                  pars=dict(b_PD=0.60 * up, b_DM=0.45 * up, k_PD=0.08 * up,
                            k_DM=0.05 * up, alpha_Q=0.55 * dn, d_0=0.10 * dn))
    return Y[2], Y[4]


_qg = np.exp(np.linspace(-2.2, 2.2, 481))
_MT, _DS = _outputs_of_quality(_qg)
mt_of = lambda q: np.interp(q, _qg, _MT)
ds_of = lambda q: np.interp(q, _qg, _DS)


def limit(s_batch, s_proc, nz=241):
    zl = zb = np.linspace(-4, 4, nz)
    wl = norm.pdf(zl); wl /= np.trapezoid(wl, zl)
    wb = norm.pdf(zb); wb /= np.trapezoid(wb, zb)
    off = -np.log(_N(S_LINE)) - (np.log(_N(s_batch)) if s_batch > 0 else 0) \
          - (np.log(_N(s_proc)) if s_proc > 0 else 0)
    Zp, Wp = _z[::4], _w[::4]
    Wp = Wp / np.trapezoid(Wp, Zp)
    L = (S_LINE * zl[:, None, None] + s_batch * zb[None, :, None]
         + s_proc * Zp[None, None, :] + off)
    q = np.exp(L); M = mt_of(q); Dd = ds_of(q)
    pH = np.trapezoid(((M >= MT_MIN) & (Dd <= DS_MAX)) * Wp, Zp, axis=2)
    muH = np.trapezoid(M * Wp, Zp, axis=2)
    ok = ((pH >= R_MIN) & (muH >= MBAR_MIN)).astype(float)
    return float(np.trapezoid(np.trapezoid(ok * wb[None, :], zb, axis=1) * wl, zl))


def monte_carlo(s_batch, s_proc, Nb, rng):
    zl = rng.standard_normal(N_BATCH)
    zb = rng.standard_normal(N_BATCH)
    zp = rng.standard_normal((N_BATCH, Nb))
    for a in (zl, zb, zp):
        np.clip(a, -4, 4, out=a)
    off = -np.log(_N(S_LINE)) - (np.log(_N(s_batch)) if s_batch > 0 else 0) \
          - (np.log(_N(s_proc)) if s_proc > 0 else 0)
    q = np.exp(S_LINE * zl[:, None] + s_batch * zb[:, None] + s_proc * zp + off)
    M, Dd = mt_of(q), ds_of(q)
    passes = ((M >= MT_MIN) & (Dd <= DS_MAX)).mean(axis=1)
    return float(((passes >= R_MIN) & (M.mean(axis=1) >= MBAR_MIN)).mean())


def marginal_pass(s_batch, s_proc, nz=241):
    """E[p(H)]: the MARGINAL pass probability, which is NOT the threshold.

    Reported to make the error concrete: at s_proc = 0.15 the marginal is above
    r_min although the limiting acceptance probability is 0.662, and at
    s_proc = 0.25 it is below although the limit is 0.046.
    """
    zl = zb = np.linspace(-4, 4, nz)
    wl = norm.pdf(zl); wl /= np.trapezoid(wl, zl)
    wb = norm.pdf(zb); wb /= np.trapezoid(wb, zb)
    off = -np.log(_N(S_LINE)) - (np.log(_N(s_batch)) if s_batch > 0 else 0) \
          - (np.log(_N(s_proc)) if s_proc > 0 else 0)
    Zp, Wp = _z[::4], _w[::4]
    Wp = Wp / np.trapezoid(Wp, Zp)
    L = (S_LINE * zl[:, None, None] + s_batch * zb[None, :, None]
         + s_proc * Zp[None, None, :] + off)
    q = np.exp(L); M = mt_of(q); Dd = ds_of(q)
    pH = np.trapezoid(((M >= MT_MIN) & (Dd <= DS_MAX)) * Wp, Zp, axis=2)
    return float(np.trapezoid(np.trapezoid(pH * wb[None, :], zb, axis=1) * wl, zl))


CELLS = ([(sb, 0.0) for sb in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)]
         + [(0.0, sp) for sp in (0.05, 0.10, 0.125, 0.15, 0.20, 0.25)]
         + [(0.10, 0.10), (0.15, 0.15)])

if __name__ == "__main__":
    banner("Table S2 / Figure S1 -- acceptance grid")
    print(f"  v_4 = Var(Z | |Z|<=4) = {V4:.6f}")
    rng = np.random.default_rng(SEED)
    rows = []
    for sb, sp in CELLS:
        mc = [monte_carlo(sb, sp, nb, rng) for nb in NB_GRID]
        lim = limit(sb, sp)
        rows.append(dict(s_batch=sb, s_proc=sp, Nb=NB_GRID, P=mc, limit=lim,
                         marginal=marginal_pass(sb, sp)))
        print(f"  s_batch={sb:.2f} s_proc={sp:.3f}  " +
              " ".join(f"{v:.3f}" for v in mc)
              + f"  limit={lim:.3f}  E[p(H)]={rows[-1]['marginal']:.3f}")
    save("tableS2_acceptance", dict(v4=V4, s_line=S_LINE, seed=SEED,
                                    n_batches=N_BATCH, rows=rows,
                                    P_plus=P_PLUS, P_minus=P_MINUS))
