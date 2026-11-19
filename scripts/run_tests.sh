#!/bin/bash
# Run the test lines of TESTS_FILE (name|conda env|args) as a pool of N parallel jobs: scripts/run_tests.sh N TESTS_FILE
# Conda environments are looked up in $CONDA_ENVS (default: the envs folder of the active Python's environment).
cd "$(dirname "$0")/.."
export CONDA_ENVS=${CONDA_ENVS:-$(dirname "$(dirname "$(dirname "$(command -v python)")")")}
run() { IFS='|' read -r name env args <<< "$1"; d=results/test/$name; mkdir -p $d
  eval "$CONDA_ENVS/$env/bin/python" main.py --run-dir $d $args > $d/log.txt 2>&1 && echo "OK $name" || echo "FAIL $name"; }
export -f run
grep -v "^#" "${2:?usage: run_tests.sh N TESTS_FILE}" | xargs -d '\n' -P ${1:-5} -I{} bash -c 'run "$@"' _ {}
echo TESTS_DONE
