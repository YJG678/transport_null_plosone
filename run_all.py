"""Run the whole reproduction pipeline in dependency order.

    python scripts/run_all.py                # everything except the FEM tables
    python scripts/run_all.py --with-fem     # everything, several hours
    python scripts/run_all.py --checks       # verification scripts only
    python scripts/run_all.py --list         # print the plan and exit
    python scripts/run_all.py table01 fig2   # run matching scripts only

Order matters in three places, and those are the only couplings:
  * table03 and figure2 read the outputs of table01;
  * figure4 reads the curves written by table09;
  * table05 reads the layouts chosen by select_pde_layouts.

Everything else is independent, so a failure in one stage does not invalidate
the others.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: (script, group, approximate runtime in minutes)
PLAN = [
    ("check_greens_functions.py", "check", 2),
    ("check_inversion.py",        "check", 0.2),
    ("check_mcmc.py",             "check", 12),

    ("table01_null_ensemble.py",  "core", 17),
    ("table02_design_sweep.py",   "core", 40),
    ("table03_dosing.py",         "core", 0.1),
    ("table04_annulus_verification.py", "core", 0.2),
    ("table06_gamma_validation.py", "core", 25),
    ("table07_loss_channel.py",   "core", 0.5),
    ("table08_sigma_sweep.py",    "core", 0.5),
    ("table09_residence_laws.py", "core", 1),
    ("table10_spacing.py",        "core", 0.5),
    ("section55_remixing.py",     "core", 0.2),
    ("tableS2_acceptance.py",     "core", 6),
    ("tableS3_sensitivity.py",    "core", 1),

    ("figure1_schematic.py",      "figure", 0.2),
    ("figure2_nullmodel.py",      "figure", 0.3),
    ("figure3_sigma.py",          "figure", 1),
    ("figure4_residence.py",      "figure", 0.2),
    ("figure5_geometry.py",       "figure", 2),
    ("figureS1_robustness.py",    "figure", 8),

    ("select_pde_layouts.py",     "fem", 50),
    ("tableS1_multitarget_pde.py", "fem", 300),
    ("table05_ensemble_pde.py",   "fem", 12),

    ("verify_table02.py",         "audit", 0.1),
    ("audit_numbers.py",          "audit", 0.2),
]


def selected(argv):
    if "--list" in argv:
        return PLAN
    pats = [a for a in argv[1:] if not a.startswith("-")]
    if pats:
        return [p for p in PLAN if any(q in p[0] for q in pats)]
    if "--checks" in argv:
        return [p for p in PLAN if p[1] == "check"]
    if "--with-fem" in argv:
        return PLAN
    return [p for p in PLAN if p[1] != "fem"]


def main(argv):
    plan = selected(argv)
    total = sum(p[2] for p in plan)
    print(f"{len(plan)} scripts, roughly {total:.0f} minutes\n")
    for name, group, mins in plan:
        print(f"  {group:<7} {name:<32} ~{mins:g} min")
    if "--list" in argv:
        return 0
    print()
    failures = []
    for name, group, _ in plan:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE)
        status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
        print(f"\n[{name}] {status} in {time.time()-t0:.0f}s\n", flush=True)
        if r.returncode != 0:
            failures.append(name)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("all scripts completed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
