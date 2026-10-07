"""Table 4: exact annular benchmark against the reduced model. Runtime: seconds."""
import numpy as np
from _common import save, banner
from organoid_null.reduced import annulus_full, annulus_reduced

EPS = [0.10, 0.05, 0.02]
SVALS = [0.0, 1.0, 5.0]

if __name__ == "__main__":
    banner("Table 4 -- annular benchmark")
    rows = []
    for eps in EPS:
        row = {"eps": eps}
        for s in SVALS:
            f, r = annulus_full(eps, s), annulus_reduced(eps, s)
            row[f"s{int(s)}"] = dict(full=f, reduced=r, err_pct=100 * (r - f) / f)
            print(f"  eps={eps:<5} s={s:<3.0f} full={f:.6f} reduced={r:.6f} err={100*(r-f)/f:+.4f}%")
        rows.append(row)
    # convergence order, halving from eps=0.2
    sweep = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]
    errs = [abs(annulus_reduced(e, 0.0) - annulus_full(e, 0.0)) / annulus_full(e, 0.0) for e in sweep]
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    order = float(np.mean([np.log2(r) for r in ratios]))
    print("  successive error ratios:", " ".join(f"{r:.2f}" for r in ratios))
    print(f"  measured convergence order {order:.3f}")
    save("table04_annulus", dict(rows=rows, eps_sweep=sweep,
                                 error_ratios=ratios, order=order))
