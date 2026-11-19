#!/usr/bin/env python
"""Run envs x variants x seeds as a worker pool (the next run starts as soon as one ends); resumable.

Sweep YAML:
  name: final
  seeds: 0-9
  envs: [conf/envs/icr_cartpole.yaml, ...]
  variants:                                   # name -> {files, set, python}
    dqn:  {files: []}
    best_tp209: {files: [conf/best.yaml], set: {teacher.version: "2.0.9"}, python: tabpfn209}
`python`: the conda environment whose Python runs the variant (an env next to this Python's env, e.g.
tabpfn209 for teacher.version 2.0.9); default = this Python. Output: results/<name>/<env>/<variant>/s<seed>/
(summary.json present = done). Runs go seed-major: seed 0 of every env/variant first, then seed 1, ...

  python tools/sweep.py conf/sweeps/final_a.yaml --workers 4 [--seeds 0-1] [--dry-run]
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def seeds_of(s):
    if isinstance(s, list):
        return s
    a, _, b = str(s).partition("-")
    return list(range(int(a), int(b or a) + 1))


def python_of(env_name):
    if not env_name:
        return sys.executable
    exe = Path(sys.executable).resolve().parents[2] / env_name / "bin" / "python"  # <conda>/envs/<env>/bin/python
    if not exe.exists():
        raise FileNotFoundError(f"conda env '{env_name}' not found ({exe})")
    return str(exe)


def jobs_of(sw, seeds):
    jobs = []
    for seed in seeds_of(seeds if seeds is not None else sw["seeds"]):
        for env_file in sw["envs"]:
            for vname, v in sw["variants"].items():
                run_dir = ROOT / "results" / sw["name"] / Path(env_file).stem / vname / f"s{seed}"
                args = [f"--config={f}" for f in [env_file, *v.get("files", [])]]
                args += [f"{k}={yaml.safe_dump(x, default_flow_style=True).strip().removesuffix('...').strip()}"
                         for k, x in v.get("set", {}).items()]
                jobs.append((run_dir, python_of(v.get("python")), args + [f"seed={seed}"]))
    return jobs


def run(job, threads):
    run_dir, exe, args = job
    if (run_dir / "summary.json").exists():
        return "skip", run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    with open(run_dir / "log.txt", "w") as log:
        rc = subprocess.run([exe, str(ROOT / "main.py"), "--run-dir", str(run_dir), *args],
                            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
    return ("ok" if rc == 0 else f"FAIL rc={rc}"), run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", help="override the sweep's seeds, e.g. 0-3")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sw = yaml.safe_load(open(a.sweep))
    jobs = jobs_of(sw, a.seeds)
    print(f"{len(jobs)} runs in sweep '{sw['name']}'", flush=True)
    if a.dry_run:
        for d, exe, args in jobs:
            print(d.relative_to(ROOT), exe, *args)
        return
    threads = max(1, (os.cpu_count() or 1) // a.workers)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (status, d) in enumerate(ex.map(lambda j: run(j, threads), jobs), 1):
            print(f"[{i}/{len(jobs)}] {status:10s} {d.relative_to(ROOT / 'results')}", flush=True)


if __name__ == "__main__":
    main()
