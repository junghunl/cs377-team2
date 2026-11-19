#!/bin/bash
# Exits as soon as the number of finished final runs grows beyond $1 (or the queue of part ${PART:-a} ends).
cd "$(dirname "$0")/.."
N0=${1:?usage: wait_new.sh <current count>}
while true; do
  n=$(ls results/final/*/*/s*/summary.json 2>/dev/null | wc -l)
  [ "$n" -gt "$N0" ] && { echo "NEW_RUNS $n (was $N0) $(date -u '+%F %T') UTC"; exit 0; }
  grep -q "^QUEUE_DONE" results/queue_final_${PART:-a}.log 2>/dev/null && { echo "QUEUE_DONE $n"; exit 0; }
  sleep 120
done
