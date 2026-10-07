"""Verify Table 2 cell by cell against the typeset manuscript.

``audit_numbers.py`` searches the manuscript for each computed value as a
substring.  That is the right check for a six-decimal quantity, but it is nearly
vacuous for the max/min column, where a value like ``42`` occurs somewhere in a
37-page document no matter what Table 2 actually says.  This script instead
parses the rows of the ``tab:nullsweep`` environment out of ms.tex and compares
all seven columns of each row against outputs/table02_design_sweep.json, at the
precision the manuscript prints.

    python scripts/verify_table02.py [path/to/manuscript/dir]

Exits non-zero and lists every disagreeing cell.  A disagreement means either
the table is stale or the code has changed; both need a human.
"""
import json
import re
import sys
from pathlib import Path

from _common import ROOT, OUT, banner

MS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ms"

#: script label -> the row label as it is typeset, since the manuscript uses
#: LaTeX maths where the script uses plain text
ROW_LABEL = {
    "baseline": "baseline",
    "N=5": r"$N=5$",
    "N=20 (d_min 0.085)": r"$N=20$ ($d_{\min}\,0.085$)",
    "eps=0.01": r"$\varepsilon=0.01$",
    "eps=0.04": r"$\varepsilon=0.04$",
    "gamma=0": r"$\gamma=0$ (conservative)",
    "gamma=0.1": r"$\gamma=0.1$",
    "port corner": "port at corner",
    "port centre": "port at centre",
    "two ports": "two ports",
    "four ports": "four ports",
    "chamber 2x0.5": r"chamber $2\times0.5$",
    "d_min=0.06": r"$d_{\min}=0.06$",
    "d_min=0.20": r"$d_{\min}=0.20$",
    "wall clearance 0.05": r"wall clearance $d_\partial=0.05$",
    "wall clearance 0.20": r"wall clearance $d_\partial=0.20$",
    "d_src=0.05": r"$d_{\mathrm{src}}=0.05$",
    "d_src=0.20": r"$d_{\mathrm{src}}=0.20$",
}


def ratio_fmt(x):
    return ("%.1f" if x < 10 else "%.0f") % x


def strip0(s):
    return s[1:] if s.startswith("0.") else s


def parse_rows(tex):
    """Pull the tab:nullsweep body out and return {row label: [7 cell strings]}."""
    m = re.search(r"\\label\{tab:nullsweep\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        raise SystemExit("could not find the tab:nullsweep environment in ms.tex")
    rows = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or line.startswith(("\\", "&")):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        if len(cells) != 8:
            continue
        rows[cells[0]] = cells[1:]
    return rows


def expected(row):
    """The seven cell strings Table 2 should carry for one computed cell."""
    m, p = row["cv_mt"]["median"], row["cv_mt"]["p90"]
    ci = lambda v: (r"\tiny$[" + strip0("%.3f" % v[1]) + ","
                    + strip0("%.3f" % v[2]) + r"]$")
    return ["%.2f" % row["cv_pi"]["median"][0],
            "%.2f" % row["cv_pi"]["p90"][0],
            "%.3f %s" % (m[0], ci(m)),
            "%.3f %s" % (p[0], ci(p)),
            ratio_fmt(row["ratio"]["median"][0]),
            ratio_fmt(row["ratio"]["p90"][0]),
            "$%.1f$" % row["improvement"]]


def normalise(s):
    """Ignore \\textbf{...} emphasis and whitespace when comparing cells."""
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    return re.sub(r"\s+", "", s)


COLUMNS = ["CV(pi) med", "CV(pi) 90th", "CV(v_mt) med", "CV(v_mt) 90th",
           "max/min med", "max/min 90th", "ratio"]

if __name__ == "__main__":
    banner("Verify Table 2 against the typeset manuscript")
    tex_path = MS_DIR / "ms.tex"
    if not tex_path.exists():
        raise SystemExit(f"no manuscript at {tex_path}")
    typeset = parse_rows(tex_path.read_text())
    computed = json.loads((OUT / "table02_design_sweep.json").read_text())["result"]

    n_ok, bad, missing = 0, [], []
    for row in computed:
        lab = ROW_LABEL.get(row["label"], row["label"])
        if lab not in typeset:
            missing.append(row["label"])
            continue
        for name, got, want in zip(COLUMNS, typeset[lab], expected(row)):
            if normalise(got) == normalise(want):
                n_ok += 1
            else:
                bad.append((row["label"], name, got, want))

    print(f"  {len(computed)} cells x {len(COLUMNS)} columns")
    print(f"  {n_ok} cells agree; {len(bad)} disagree")
    for lab, col, got, want in bad:
        print(f"    {lab:<22} {col:<14} manuscript {got!r}  computed {want!r}")
    if missing:
        print("  rows not found in the manuscript: " + ", ".join(missing))
    extra = set(typeset) - {ROW_LABEL.get(r["label"], r["label"]) for r in computed}
    if extra:
        print("  manuscript rows with no computed counterpart: " + ", ".join(extra))
    sys.exit(1 if (bad or missing) else 0)
