"""Table 8: conservative sigma sweep (continued search) with the finite-horizon
phenotype, plus the first-adsorption moments that set the readout horizon.

Because pi_1 = 1 for every sigma in the conservative one-target problem,
retention changes only the approach to the plateau, so the horizon T is chosen
as a small multiple of the mean first-adsorption time rather than as a long
horizon (at which the terminal state is sigma-independent).  That mean is
computed here from the same transform the sweep uses.

Runtime: about a minute."""
import numpy as np
from _common import save, banner
from organoid_null.green import Chamber
from organoid_null.reduced import (Kinetics, conditional_moments,
                                   one_target_transform)
from organoid_null.phenotype import phenotype, T_HORIZON

X1, X0 = [(0.5, 0.5)], (0.25, 0.25)
SIGMAS = [0.00, 0.10, 0.20, 0.33, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def adsorption_moments(ch, kin, h=1e-3):
    """(mean, sd) of the FIRST-ADSORPTION time alone.

    ``one_target_transform`` returns the productive flux, which carries the
    internalization factor gb/(s+gb); dividing it out leaves the adsorption
    transform, whose log-derivatives at s = 0 are the first-passage moments.
    Taking sigma = 0 makes the boundary filter g identically one, so no
    desorption cycles are folded in.
    """
    F = lambda s: (np.real(one_target_transform(ch, X1[0], X0, s, kin))
                   * (s + kin.gb) / kin.gb)
    f0 = F(0.0)
    d1 = (-F(2 * h) + 8 * F(h) - 8 * F(-h) + F(-2 * h)) / (12 * h)
    d2 = (-F(2 * h) + 16 * F(h) - 30 * f0 + 16 * F(-h) - F(-2 * h)) / (12 * h * h)
    T = -d1 / f0
    return float(T), float(np.sqrt(d2 / f0 - T ** 2))


if __name__ == "__main__":
    banner("Table 8 -- conservative sigma sweep")
    ch = Chamber(0.0, X1 + [X0])

    T_ads, sd_ads = adsorption_moments(ch, Kinetics(kp=2.0, sigma=0.0, eps=0.02))
    print(f"  mean first-adsorption time T_ads = {T_ads:.4f} "
          f"(sd {sd_ads:.4f}); readout horizon T = {T_HORIZON:g} "
          f"= {T_HORIZON / T_ads:.1f} x T_ads")

    rows = []
    for sig in SIGMAS:
        kin = Kinetics(kp=2.0, sigma=sig, eps=0.02)
        pi, T1, sd = conditional_moments(X1, X0, 0.0, kin, ch=ch)
        Y = phenotype(pi, T1, sd)
        rows.append(dict(sigma=sig, pi=float(pi[0]), T_tot=float(T1[0]), sd=float(sd[0]),
                         mt=float(Y[2][0]), ds=float(Y[4][0]), ap=float(Y[3][0])))
        print(f"  sigma={sig:<5} pi={pi[0]:.4f} T={T1[0]:.4f} sd={sd[0]:.4f} "
              f"mt={Y[2][0]:.4f} ds={Y[4][0]:.4f} ap={Y[3][0]:.4f}")
    save("table08_sigma_sweep", dict(
        rows=rows,
        adsorption=dict(T_ads=T_ads, sd_ads=sd_ads, horizon=T_HORIZON,
                        horizon_in_T_ads=T_HORIZON / T_ads)))
