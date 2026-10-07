"""Shared plumbing: paths, output helpers, provenance stamping."""
import json, os, platform, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def save(name, payload):
    """Write a JSON result file with a provenance header."""
    import organoid_null
    meta = dict(script=Path(sys.argv[0]).name,
                package_version=organoid_null.__version__,
                python=platform.python_version(),
                numpy=np.__version__)
    try:
        import scipy
        meta["scipy"] = scipy.__version__
    except Exception:
        pass
    path = OUT / f"{name}.json"
    path.write_text(json.dumps({"_provenance": meta, "result": payload}, indent=1))
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def load(name):
    """Read a previously generated result, with a message that says what to run."""
    path = OUT / f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)}.\n"
            f"Run the script that produces it first "
            f"(see MANIFEST.md), or `python scripts/run_all.py`.")
    return json.loads(path.read_text())["result"]


def load_npz(name):
    """Same, for the .npz side-outputs."""
    path = OUT / f"{name}.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)}.\n"
            f"Run the script that produces it first "
            f"(see MANIFEST.md), or `python scripts/run_all.py`.")
    return np.load(path)


def fig(name):
    return str(OUT / f"{name}.pdf")


def banner(title):
    print(f"\n=== {title} ===", flush=True)
