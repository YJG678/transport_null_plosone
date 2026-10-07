"""Cross-check the numbers this package produces against the manuscript text.

Every reported value is re-derived from the JSON in outputs/, formatted to the
precision the manuscript uses, and searched for verbatim in ms.tex plus
ms-supplement.tex.  A value that does not appear verbatim is reported as
MISSING: either the manuscript is stale or the code has changed, and both are
worth knowing.

    python scripts/audit_numbers.py [path/to/manuscript/dir]

Default manuscript directory is ../ms relative to the package root.  Scripts
whose outputs are absent are skipped with a note rather than failing, so the
audit is useful after a partial run.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from _common import ROOT, OUT

MS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ms"

_ok, _bad, _skipped = [], [], []


def _text():
    parts = []
    for name in ("ms.tex", "ms-supplement.tex"):
        p = MS_DIR / name
        if p.exists():
            parts.append(p.read_text())
    return "\n".join(parts)


TEXT = _text()


def chk(label, value, fmt="%.3f"):
    s = fmt % value
    (_ok if s in TEXT else _bad).append((label, s))


def chk_near(label, value, anchor, fmt="%.3f", window=400):
    """Check a value appears NEAR an anchor phrase, not merely somewhere.

    A bare substring search is nearly vacuous for a value like 0.330, which
    occurs in several unrelated places in a 37-page document; a prose claim can
    then be silently wrong while the audit passes.  ``anchor`` is a regex
    identifying the sentence, and the value must occur within ``window``
    characters of it.
    """
    s = fmt % value
    spots = list(re.finditer(anchor, TEXT))
    if not spots:
        _bad.append((label + " [anchor not found]", s))
        return
    # an anchor phrase may occur more than once (a definition and then its
    # use); the value need only sit near one of them
    near = any(s in TEXT[max(0, m.start() - window):m.end() + window]
               for m in spots)
    (_ok if near else _bad).append((label, s))


def load(name):
    p = OUT / f"{name}.json"
    if not p.exists():
        _skipped.append(name)
        return None
    return json.loads(p.read_text())["result"]


# ---------------------------------------------------------------- Tables 1, 3
r = load("table01_null_ensemble")
if r:
    for key, fmt in (("cv_pi", "%.3f"), ("ratio", "%.1f"), ("cv_mt", "%.3f"),
                     ("cv_ds", "%.3f"), ("cv_mt_uniform", "%.3f")):
        for q in (10, 50, 90):
            v = r[key][f"p{q}"]
            chk(f"Table1 {key} p{q}", v[0], fmt)
            chk(f"Table1 {key} p{q} lo", v[1], fmt)
            chk(f"Table1 {key} p{q} hi", v[2], fmt)
    chk("Table1 dose ratio mean", r["dose_ratio_mean"], "%.4f")
    chk("Table3 ratio of medians", r["ratio_of_medians"], "%.1f")

# ------------------------------------------------------------------- Table 2
# All seven columns, not just the phenotype CVs: the max/min column is printed
# to a coarser precision than it is computed, so a rounding slip there is
# invisible unless it is checked explicitly.
r = load("table02_design_sweep")
if r:
    def _ratio(x):
        return ("%.1f" if x < 10 else "%.0f") % x

    for row in r:
        lab = row["label"]
        chk(f"Table2 {lab} CVpi med", row["cv_pi"]["median"][0], "%.2f")
        chk(f"Table2 {lab} CVpi p90", row["cv_pi"]["p90"][0], "%.2f")
        for q in ("median", "p90"):
            v = row["cv_mt"][q]
            chk(f"Table2 {lab} CVmt {q}", v[0])
            # the manuscript writes the interval bounds without the leading zero
            for j, side in ((1, "lo"), (2, "hi")):
                s = "%.3f" % v[j]
                (_ok if (s in TEXT or s.lstrip("0") in TEXT) else _bad).append(
                    (f"Table2 {lab} CVmt {q} {side}", s))
        for q in ("median", "p90"):
            s = _ratio(row["ratio"][q][0])
            (_ok if s in TEXT else _bad).append((f"Table2 {lab} max/min {q}", s))
        chk(f"Table2 {lab} improvement", row["improvement"], "%.1f")

# ------------------------------------------------------------------- Table 4
r = load("table04_annulus")
if r:
    for row in r["rows"]:
        for s in (0, 1, 5):
            e = row[f"s{s}"]
            chk(f"Table4 eps={row['eps']} s={s} full", e["full"], "%.6f")
            chk(f"Table4 eps={row['eps']} s={s} reduced", e["reduced"], "%.6f")
    chk("Table4 convergence order", r["order"], "%.2f")

# ------------------------------------------------------------------- Table 5
r = load("table05_ensemble_pde")
if r:
    for row in r:
        if row["regime"] != "lossy":
            continue
        chk(f"Table5 {row['layout']} CV full", row["cv_mt_full"], "%.4f")
        chk(f"Table5 {row['layout']} CV reduced", row["cv_mt_red"], "%.4f")
        chk(f"Table5 {row['layout']} d/r", row["d_over_r"], "%.1f")
        chk(f"Table5 {row['layout']} dsrc/r", row["dsrc_over_r"], "%.1f")

# ------------------------------------------------------------------- Table 6
r = load("table06_gamma_validation")
if r:
    for nm, s in r["summary"].items():
        chk(f"Table6 {nm} E_CDF", s["E_CDF"])
        chk(f"Table6 {nm} E_J", s["E_J"])
        chk(f"Table6 {nm} rel mt %", 100 * s["rel_mt"], "%.2f")
        chk(f"Table6 {nm} rel ds %", 100 * s["rel_ds"], "%.2f")
    for c in r["cases"]:
        if c["label"].startswith("reset"):
            chk(f"Table9 {c['label']} mt exact", c["mt_ref"][0], "%.4f")
            chk(f"Table9 {c['label']} ds exact", c["ds_ref"][0], "%.4f")

# ---------------------------------------------------------------- Tables 7, 8
r = load("table07_loss_channel")
if r:
    for row in r:
        for v in row["pi"]:
            chk(f"Table7 gamma={row['gamma']}", v, "%.4f")
        chk(f"Table7 gamma={row['gamma']} spread", row["spread"])
        chk(f"Table7 gamma={row['gamma']} Robin [0,0.95]", row["ratio_to_095"])
        chk(f"Table7 gamma={row['gamma']} Robin [0,0.90]", row["ratio_to_090"])

r = load("table08_sigma_sweep")
if r:
    rows8 = r["rows"] if isinstance(r, dict) else r
    for row in rows8:
        chk(f"Table8 sig={row['sigma']} T1", row["T_tot"], "%.4f")
        chk(f"Table8 sig={row['sigma']} sd", row["sd"], "%.4f")
        chk(f"Table8 sig={row['sigma']} mt", row["mt"], "%.4f")
        chk(f"Table8 sig={row['sigma']} ds", row["ds"], "%.4f")
    if isinstance(r, dict):
        # the readout-horizon justification in section 3
        chk_near("Sec3 T_ads", r["adsorption"]["T_ads"],
                 r"mean first-adsorption time", "%.4f")
        chk_near("Sec3 T_ads sd", r["adsorption"]["sd_ads"],
                 r"mean first-adsorption time", "%.4f")

# ------------------------------------------------------------------ Table 9
r = load("table09_residence_laws")
if r:
    for row in r["rows"]:
        chk(f"Table9 {row['law']} T_tot", row["T_tot"], "%.4f")
        chk(f"Table9 {row['law']} sd", row["sd"], "%.4f")
        chk(f"Table9 {row['law']} mt", row["mt"], "%.4f")
        chk(f"Table9 {row['law']} ds", row["ds"], "%.4f")
    for v in r["surface_variances"]:
        chk("Table9 surface variance", v)
    for v in r.get("surface_pr_below_1", []):
        chk_near("Table9 Pr(T_surf<1)", v, r"\\Pr\(T_\{\\mathrm\{surf\}\}<1\)")

# ----------------------------------------------------------------- Table 10
r = load("table10_spacing")
if r:
    for row in r["rows"]:
        chk(f"Table10 d={row['d']} U", row["U12"], "%.3f")
        chk(f"Table10 d={row['d']} rho", row["rho"], "%.3f")

# --------------------------------------------------------------- Section 5.5
r = load("section55_remixing")
if r:
    for v in r["uptake_exact"]:
        chk("Sec5.5 uptake", v, "%.4f")
    chk("Sec5.5 gap exact", r["gap_exact"], "%.3f")
    chk("Sec5.5 gap idealized", r["gap_ideal"], "%.3f")
    for row in r["eps_sweep"]:
        if row["eps"] in (0.02, 1e-3, 1e-10):
            chk(f"Sec5.5 fraction at eps={row['eps']:g}", row["fraction"])
    chk("Sec5.5 pibar_1", r["pibar"][0], "%.4f")
    chk("Sec5.5 P_12", r["P"][0][1], "%.4f")
    chk("Sec5.5 homogeneous end", r["homogeneous"][-1], "%.4f")

# ----------------------------------------------------------------- Table S1
r = load("tableS1_multitarget_pde")
if r and isinstance(r, dict) and "rows" in r:
    for row in r["rows"]:
        chk(f"TableS1 N={row['N']} eps={row['eps']} slope", row["slope"], "%.2f")
        chk(f"TableS1 N={row['N']} eps={row['eps']} d/r", row["d_over_r"], "%.1f")
    lo, hi = r["slope_range"]
    chk("TableS1 slope range lo", lo, "%.2f")
    chk("TableS1 slope range hi", hi, "%.2f")

# ----------------------------------------------------------------- Table S2
r = load("tableS2_acceptance")
if r:
    chk("TableS2 v4", r["v4"], "%.6f")
    for row in r["rows"]:
        for v in row["P"]:
            chk(f"TableS2 P sb={row['s_batch']} sp={row['s_proc']}", v)
        chk(f"TableS2 limit sb={row['s_batch']} sp={row['s_proc']}", row["limit"])
        if row["s_proc"] in (0.15, 0.25) and row["s_batch"] == 0.0:
            chk(f"TableS2 marginal sp={row['s_proc']}", row.get("marginal", float("nan")))

# ----------------------------------------------------------------- Table S3
r = load("tableS3_sensitivity")
if r:
    for side in ("loss_channel", "closure"):
        d = r[side]
        for i, nm in enumerate(d["inputs"]):
            chk(f"TableS3 {side} {nm} PRCC", d["prcc"][i])
            chk(f"TableS3 {side} {nm} S", d["S"][i])
            chk(f"TableS3 {side} {nm} ST", d["ST"][i])
        chk(f"TableS3 {side} sum S", d["sum_S"])

# --------------------------------------------------------------------- checks
r = load("check_inversion")
if r:
    chk("check inversion worst", r["worst"] * 1e9, "%.1f")

r = load("check_mcmc")
if r:
    two = [x for x in r["rows"] if x["init"] == "grid" and x["sweeps"] == 2][0]
    chk("check mcmc p at 2 sweeps (grid)", two["ks_p"], "%.2f")
    rest = [x["ks_p"] for x in r["rows"] if x["sweeps"] >= 5]
    chk("check mcmc min p at >=5 sweeps", min(rest), "%.2f")

r = load("check_greens_functions")
if r:
    chk("check Ewald G0", r["ewald"]["G0"], "%.10f")
    chk("check Ewald R0", r["ewald"]["R0"], "%.10f")
    if "series" in r:
        for v in r["series"]["diagonal"]:
            chk_near("Sec3 raw series diagonal", v,
                     r"the raw sum diverges", "%.4f")
        chk_near("Sec3 raw series growth", r["series"]["growth_per_doubling"],
                 r"the raw sum diverges", "%.2f")

if __name__ == "__main__":
    print(f"manuscript: {MS_DIR}")
    if not TEXT:
        print("  no manuscript found -- nothing to audit against")
        sys.exit(1)
    print(f"{len(_ok)} reported values matched the manuscript verbatim; "
          f"{len(_bad)} did not")
    for lab, s in _bad:
        print(f"  MISSING  {lab:<44} computed {s}")
    if _skipped:
        print("\nskipped (output not present): " + ", ".join(sorted(set(_skipped))))
    sys.exit(1 if _bad else 0)
