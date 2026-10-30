#!/usr/bin/env python3
"""Generate SISSO screened spaces and run the conditional Bayesian analysis."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT)]
from bayes_so2.io import load_sisso_directory
from nested_cv import run_sisso
import full_data

# Original SISSO 3.5 settings, embedded so only train.dat and SISSO.out
# need to be distributed as data. Input files are generated at runtime.
SISSO_CONFIG = {
    2: """ptype=1
ntask=1
desc_dim=2
nsample=88
restart=0

fstore=1
nsf=17
ops='(+)(-)(*)(/)(^-1)(^2)'
fcomplexity=5
funit=(1:3)(4:6)(7:9)(10:12)(13:15)(16:17)
fmax_min=1e-3
fmax_max=1e5
nf_sis=200,200

method_so='L0'
fit_intercept=.true.
metric='RMSE'
nmodels=100
""",
    3: """ptype=1
ntask=1
desc_dim=3
nsample=88
restart=0

fstore=1
nsf=17
ops='(+)(-)(*)(/)(^-1)(^2)'
fcomplexity=5
funit=(1:3)(4:6)(7:9)(10:12)(13:15)(16:17)
fmax_min=1e-3
fmax_max=1e5
nf_sis=100,100,100

method_so='L0'
fit_intercept=.true.
metric='RMSE'
nmodels=100
""",
}


def prepare_space(dimension, output, executable, force=False):
    source = ROOT / f"SISSO-{dimension}D"
    work = output / "sisso" / f"SISSO-{dimension}D"
    inputs = {"train.dat": (source / "train.dat").read_text(),
              "SISSO.in": SISSO_CONFIG[dimension]}
    complete = all((work / name).is_file() for name in
                   ("SISSO.out", "SIS_subspaces/Uspace.dat", "SIS_subspaces/Uspace.expressions"))
    same_inputs = all((work / name).is_file() and (work / name).read_text() == text
                      for name, text in inputs.items())
    if complete and same_inputs and not force:
        print(f"d={dimension}: using generated screened space in {work}", flush=True)
        return work
    if executable is None:
        raise RuntimeError("Supply --sisso-executable /absolute/path/to/SISSO (version 3.5). "
                           "The supplied SISSO.out does not contain the full screened space.")
    # Remove only this generated run to avoid mixing restart/feature artifacts.
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for name, text in inputs.items():
        (work / name).write_text(text)
    print(f"d={dimension}: generating the full-data screened space with SISSO", flush=True)
    run_sisso(executable, work)
    load_sisso_directory(work)  # Validate generated output before inference.
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sisso-executable", default=os.environ.get("SISSO_EXECUTABLE"))
    parser.add_argument("--mode", choices=("full", "nested", "synthetic", "all"), default="full")
    parser.add_argument("--dimension", type=int, choices=(2, 3), help="default: both dimensions")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--force", action="store_true", help="rerun generated full-data and fold SISSO runs")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents and output.relative_to(ROOT).parts[0] in ("SISSO-2D", "SISSO-3D", "analysis", "bayes_so2"):
        parser.error("choose a separate results directory")
    executable = None
    if args.sisso_executable:
        located = shutil.which(args.sisso_executable)
        if located is None:
            parser.error(f"SISSO executable not found or not executable: {args.sisso_executable}")
        executable = Path(located).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    dimensions = [args.dimension] if args.dimension else [2, 3]
    if args.mode != "synthetic":
        for d in dimensions:
            template = prepare_space(d, output, executable, args.force)
            full_data.PROJECT = output / "sisso"
            full_data.HERE = output
            full_data.write_full_posterior(d)
            print(f"d={d}: posterior saved to {output / 'full_data_posterior' / f'd{d}'}", flush=True)
            if d == 2:
                full_data.feature_201_203_diagnostic()
            if args.mode in ("nested", "all"):
                if executable is None:
                    parser.error("nested validation requires --sisso-executable")
                command = [sys.executable, str(ROOT / "analysis/nested_cv.py"),
                           "--dimension", str(d), "--seed", "42", "--repetition", "1",
                           "--sisso-template-dir", str(template),
                           "--sisso-executable", str(executable), "--output-root", str(output)]
                if args.force:
                    command.append("--force")
                subprocess.run(command, check=True)
        if args.mode in ("nested", "all"):
            import nested_diagnostics
            nested_diagnostics.HERE = output
            nested_diagnostics.main()
    if args.mode in ("synthetic", "all"):
        import synthetic_test
        synthetic_test.HERE = output
        synthetic_test.main()
    print(f"Finished. Generated inputs, screened spaces, and results: {output}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
