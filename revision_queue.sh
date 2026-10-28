#!/bin/bash
# Runs every revision experiment in revision_jobs.txt, N at a time, on CPU.
# Each run is resumable (finished folds are skipped), so the queue can be
# restarted safely after an interruption.
cd "$(dirname "$0")"
N=${1:-6}
mkdir -p results/revision/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
run_one() {
  log="results/revision/logs/$(echo "$*" | tr -d '-' | tr ' ' '_').log"
  "${PYTHON:-python}" run_revision.py "$@" --device cpu --threads 2 >> "$log" 2>&1
  echo "$(date '+%F %T') exit $?: $*" >> results/revision/logs/queue.log
}
export -f run_one
grep -v '^\s*$' revision_jobs.txt | xargs -P "$N" -L 1 bash -c 'run_one "$@"' _
echo "$(date '+%F %T') QUEUE FINISHED" >> results/revision/logs/queue.log
