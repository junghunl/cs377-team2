#!/usr/bin/env python
"""Run one experiment.

  python main.py -c conf/envs/icr_cartpole.yaml -c conf/best.yaml --run-dir results/test seed=0
  python main.py -c conf/envs/icr_cartpole.yaml --run-dir results/dqn                     # plain DQN
  python main.py -c conf/envs/lunarlander.yaml -c conf/best.yaml --run-dir results/x gate.name=adaptive
  (teacher.version=2.0.9 must run with the Python of the tabpfn==2.0.9 environment)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabrl.config import load_config  # noqa: E402
from tabrl.trainer import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", action="append", default=[], help="YAML file(s), applied in order")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value, e.g. label.double=false paths=[]")
    a = ap.parse_args()
    print(json.dumps(Trainer(load_config(a.config, a.overrides), a.run_dir).run(), indent=2))


if __name__ == "__main__":
    main()
