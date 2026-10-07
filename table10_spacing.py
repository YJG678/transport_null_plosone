"""Table 10: raw uptake and the competition-normalized ratio against spacing.
Runtime: about a minute."""
import numpy as np
from _common import save, banner
from organoid_null.reduced import Kinetics, fluxes

SRC = (0.5, 0.5)
DVALS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80]

if __name__ == "__main__":
    banner("Table 10 -- competition against spacing")
    kin = Kinetics(kp=1.5, sigma=0.0, eps=0.02)
    rows = []
    for d in DVALS:
        xs = [(0.5 - d / 2, 0.5), (0.5 + d / 2, 0.5)]
        tot = float(fluxes(xs, SRC, 0.0, 0.4, kin).sum())
        iso = float(sum(fluxes([x], SRC, 0.0, 0.4, kin)[0] for x in xs))
        rows.append(dict(d=d, U12=tot, rho=tot / iso))
        print(f"  d={d:<5} U12={tot:.3f}  rho={tot/iso:.3f}")
    # source independence of rho
    checks = []
    for src in [(0.5, 0.5), (0.5, 0.25), (0.5, 0.85), (0.25, 0.5), (0.2, 0.3)]:
        xs = [(0.35, 0.5), (0.65, 0.5)]
        tot = float(fluxes(xs, src, 0.0, 0.4, kin).sum())
        iso = float(sum(fluxes([x], src, 0.0, 0.4, kin)[0] for x in xs))
        checks.append(dict(source=src, rho=tot / iso))
    print("  rho at five source positions:", " ".join(f"{c['rho']:.4f}" for c in checks))
    save("table10_spacing", dict(rows=rows, source_independence=checks))
