"""Table 7: eventual uptake against sigma as bulk loss is removed, plus the
static-Robin overestimation factor. Runtime: about a minute."""
import numpy as np
from _common import save, banner
from organoid_null.reduced import Kinetics, fluxes

X1, X0 = [(0.5, 0.5)], (0.25, 0.25)
SIGMAS = [0.0, 0.30, 0.60, 0.90, 0.95]
GAMMAS = [0.400, 0.100, 0.020, 0.004]


def pi_one(sigma, gam, static=False):
    kin = Kinetics(kp=1.5, sigma=(0.0 if static else sigma), eps=0.02)
    return float(fluxes(X1, X0, 0.0, gam, kin)[0])


if __name__ == "__main__":
    banner("Table 7 -- loss-channel sweep")
    rows = []
    for gam in GAMMAS:
        vals = [pi_one(s, gam) for s in SIGMAS]
        stat = pi_one(0.0, gam, static=True)
        r = [stat / v for v in vals]
        rows.append(dict(gamma=gam, pi=vals, spread=vals[0] - vals[-1],
                         static=stat, ratio_to_090=max(r[:4]), ratio_to_095=max(r)))
        print(f"  gamma={gam:<6} " + " ".join(f"{v:.4f}" for v in vals) +
              f"  spread={vals[0]-vals[-1]:.3f}  static/rev max [0,.90]={max(r[:4]):.3f}"
              f" [0,.95]={max(r):.3f}")
    save("table07_loss_channel", rows)
