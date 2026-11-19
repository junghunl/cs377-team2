#!/bin/bash
# Final experiments (conf/sweeps/final_<part>.yaml): one worker pool, seed-major order; 2nd pass retries failed runs.
#   PART=a setsid nohup scripts/queue_final.sh > results/queue_final_a.log 2>&1 < /dev/null &   (this server)
#   PART=b setsid nohup scripts/queue_final.sh > results/queue_final_b.log 2>&1 < /dev/null &   (other server)
# PYTHON = the main environment's Python (tabpfn 9.x); the tabpfn209 conda env must sit next to it.
cd "$(dirname "$0")/.."
P=${PYTHON:-/root/miniforge3/envs/tabpfn-rl/bin/python}
PART=${PART:?set PART=a or PART=b}
echo "START $PART $(date -u '+%F %T') UTC"
for pass in 1 2; do
  "$P" tools/sweep.py conf/sweeps/final_$PART.yaml --workers ${WORKERS:-4}
  echo "STAGE_DONE pass$pass $(date -u '+%F %T') UTC"
done
echo "QUEUE_DONE $(date -u '+%F %T') UTC"
