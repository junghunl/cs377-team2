"""Collect the finished final runs (results/final/<env>/<variant>/s<seed>/summary.json) into two JSON files
{rows, meta} for the dashboard.

  python -I tools/collect.py RESULTS_DIR OUT_DIR SERVER        # SERVER = a | b

runs_<server>.json : one row per run (auc = mean evaluation return, final, teacher_off_step, hours, ...)
evals_<server>.json: one row per evaluation (eval = student's return, teacher = teacher's greedy return,
                     g = teacher strength, tq = teacher's max_a Q(s0, a), tdisc = teacher's discounted return)
"""
import csv
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

RES, OUT, SERVER = sys.argv[1], sys.argv[2], sys.argv[3]
CODE = open(__file__).read()


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


runs, evals = [], []
for sp in sorted(glob.glob(os.path.join(RES, "final", "*", "*", "s*", "summary.json"))):
    d = os.path.dirname(sp)
    env, variant, s = d.split(os.sep)[-3:]
    seed, sm = int(s[1:]), json.load(open(sp))
    runs.append({"server": SERVER, "env": env, "variant": variant, "seed": seed,
                 "auc": num(sm.get("auc_eval")), "final": num(sm.get("final_eval")), "steps": sm.get("steps"),
                 "version": sm.get("teacher_version"), "teacher_off_step": sm.get("teacher_off_step"),
                 "teacher_refits": sm.get("teacher_refits"), "hours": round(sm.get("wall_time_s", 0) / 3600, 2),
                 "finished_at": datetime.fromtimestamp(os.path.getmtime(sp), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    with open(os.path.join(d, "evals.csv"), newline="") as f:
        for r in csv.DictReader(f):
            evals.append({"env": env, "variant": variant, "seed": seed, "step": int(float(r["step"])),
                          "eval": num(r["eval_mean"]), "teacher": num(r["teacher_eval"]), "g": num(r["g"]),
                          "tq": num(r["teacher_v0"]), "tdisc": num(r["teacher_disc"])})

now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
os.makedirs(OUT, exist_ok=True)
for name, rows, inp in (("runs", runs, "summary.json"), ("evals", evals, "evals.csv")):
    meta = [{"label": "collect.py", "value": CODE, "type": "code", "lang": "python"},
            {"label": "Input", "value": f"results/final/<env>/<variant>/s<seed>/{inp}"}, {"label": "Server", "value": SERVER},
            {"label": "Collected at (UTC)", "value": now}, {"label": "Rows", "value": str(len(rows))}]
    path = os.path.join(OUT, f"{name}_{SERVER}.json")
    with open(path, "w") as f:
        json.dump({"rows": rows, "meta": meta}, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(path, len(rows), os.path.getsize(path))
