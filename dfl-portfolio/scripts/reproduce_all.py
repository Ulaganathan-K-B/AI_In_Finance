"""Re-run every experiment in the report. Cross-platform replacement for reproduce_all.sh.

Works on Windows (PowerShell / cmd), macOS and Linux; needs nothing but Python.

    python scripts/reproduce_all.py                    # everything, ~4 h on 2 CPU cores
    python scripts/reproduce_all.py --quick            # 1 seed, 3 test years: smoke test, ~5-10 min
    python scripts/reproduce_all.py --processes 4      # more worker processes
    python scripts/reproduce_all.py --only figures     # just rebuild figures and tables from results/
    python scripts/reproduce_all.py --dry-run          # print the commands without running them

Every step is logged to results/logs/<step>.log (same names as reproduce_all.sh).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"


def build_steps(processes: int, quick: bool):
    p = ["--processes", str(processes)]
    q = ["--quick"] if quick else []
    walk = lambda cfg, *extra: ["scripts/run_walkforward.py", f"configs/{cfg}.yaml", *p, *q, *extra]
    synth = lambda cfg, *extra: ["scripts/run_synthetic.py", f"configs/{cfg}.yaml", *p, *q, *extra]
    steps = [
        ("download", ["scripts/download_data.py"]),
        ("tests", ["-m", "pytest", "-q", "tests"]),
        ("sector9", walk("sector9")),
        ("synthetic", synth("synthetic")),
        ("synthetic_beta", synth("synthetic_beta")),
        ("megacap10", walk("megacap10")),
        ("notebook_replication", walk("megacap10_notebook_replication")),
        ("sector9_notebook_replication", walk("sector9_notebook_replication")),
        ("ff12", walk("ff12")),
        ("ff12_eta", walk("ff12_eta")),
        ("robust", walk("robust_sector9")),
        ("robust_frozen", walk("robust_sector9", "--frozen")),
    ]
    if not quick:  # the two slowest experiments are skipped in quick mode (as in reproduce_all.sh)
        steps += [
            ("synthetic_lambda", ["scripts/run_synthetic.py", "configs/synthetic_lambda.yaml", *p]),
            ("ablation", ["scripts/run_ablation.py", "configs/ablation_sector9.yaml", *p]),
        ]
    steps.append(("figures", ["scripts/make_figures.py", *q]))
    return steps


def run_step(name: str, args: list[str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, *args]
    print(f"\n=== {name}: {' '.join(args)}", flush=True)
    t0 = time.time()
    with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:  # tee: console + log file
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    print(f"=== {name} finished with exit code {proc.returncode} in {time.time() - t0:.0f}s", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 test years per experiment (smoke test)")
    ap.add_argument("--processes", type=int, default=int(os.environ.get("PROCESSES", 2)),
                    help="worker processes per experiment (default 2)")
    ap.add_argument("--only", default=None, help="comma-separated step names to run, e.g. 'sector9,figures'")
    ap.add_argument("--skip-tests", action="store_true", help="skip the pytest step")
    ap.add_argument("--skip-download", action="store_true", help="skip download_data.py (the cached prices are used)")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing step")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never crash on a console that cannot show a character

    steps = build_steps(args.processes, args.quick or bool(os.environ.get("QUICK")))
    names = [n for n, _ in steps]
    if args.only:
        wanted = [s.strip() for s in args.only.split(",")]
        unknown = [w for w in wanted if w not in names]
        if unknown:
            ap.error(f"unknown step(s) {unknown}; choose from {names}")
        steps = [(n, a) for n, a in steps if n in wanted]
    if args.skip_tests:
        steps = [(n, a) for n, a in steps if n != "tests"]
    if args.skip_download:
        steps = [(n, a) for n, a in steps if n != "download"]

    if args.dry_run:
        for n, a in steps:
            print(f"{n:30s} {Path(sys.executable).name} {' '.join(a)}")
        return 0

    failed = []
    for n, a in steps:
        if run_step(n, a) != 0:
            failed.append(n)
            if not args.keep_going:
                print(f"\nStopped: step '{n}' failed; see results/logs/{n}.log", file=sys.stderr)
                return 1
    print("\nAll steps finished." if not failed else f"\nFailed steps: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
