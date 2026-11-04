#!/usr/bin/env bash
# Rerun one V7 arm on the complete sets (1560 instances, seeds 1-10, 50 000 schedules).
#
#   analysis/v7/run_campaign.sh <arm> [processes] [julia_project]     (run from the repository root)
#
# arm: V7sb, V7s, V7b (experiments/v7_driver.jl) or V7sbk1 (experiments/v7k_driver.jl).
# The instance list is split into <processes> shards (default 6), one single-threaded
# Julia process each. Output: results/v7/<arm>/s<k>_runs.csv and s<k>_traj.csv, the
# format of analysis/v7/data/. About 6 h per arm with 6 processes on a laptop.
set -euo pipefail
arm=$1; P=${2:-6}; ENV=${3:-analysis/v7/julia-env}
D="results/v7/$arm"; mkdir -p "$D"
for ((k = 0; k < P; k++)); do awk -v k=$k -v P=$P 'NR % P == k' analysis/v7/instances_all.txt > "$D/shard$k.txt"; done
for ((k = 0; k < P; k++)); do
  JULIA_NUM_THREADS=1 julia --project="$ENV" experiments/v7k_driver.jl "$D/shard$k.txt" 1:10 "$arm" 50000 "$D/s$k" > "$D/log$k.txt" 2>&1 &
  sleep 20   # stagger the start-up (compilation)
done
wait
echo "done: $D"
