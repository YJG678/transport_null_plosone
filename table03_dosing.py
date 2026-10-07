"""Table 3: transport-only maturation CV by dosing geometry.

Table 3 is a re-presentation of two rows already computed by
``table01_null_ensemble.py`` (the point and uniform maturation CVs over the same
2000 layouts), so it is NOT recomputed here: recomputing it from an independent
run would silently allow the two tables to disagree.  Run table01 first.
"""
from _common import load, save, banner

if __name__ == "__main__":
    banner("Table 3 -- dosing geometry")
    r = load("table01_null_ensemble")
    rows = [("point source (localized port)", r["cv_mt"]),
            ("uniform source (medium exchange)", r["cv_mt_uniform"])]
    for name, d in rows:
        print(f"  {name:<34} " + "  ".join(
            f"{d[k][0]:.3f} [{d[k][1]:.3f},{d[k][2]:.3f}]" for k in ("p10", "p50", "p90")))
    ratio = r["cv_mt"]["p50"][0] / r["cv_mt_uniform"]["p50"][0]
    print(f"  reduction in the median: {ratio:.1f}-fold")
    save("table03_dosing", dict(rows={n: d for n, d in rows}, median_ratio=ratio))
