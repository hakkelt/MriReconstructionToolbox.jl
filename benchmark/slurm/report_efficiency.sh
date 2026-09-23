#!/bin/bash
# Post-hoc efficiency report for a matrix.sh job.
#
#   benchmark/slurm/report_efficiency.sh <jobid>
#
# Tasks run as plain background `numactl`-pinned processes, not srun steps (see launch_task in
# matrix.sh for why), so there is no per-task `sacct` row for `reportseff`. Each task's
# `/usr/bin/time -v` output, captured in its log, is the per-task efficiency source instead:
# "Percent of CPU this job got" is the signal CPUEff would have given. A low value on a task is the
# direct symptom of the NUMA-fragmentation anomaly the launcher exists to prevent (threads waiting on
# remote memory instead of computing). "Maximum resident set size" close to the per-domain memory
# flags a task worth checking before a later run pushes it over and --membind kills it.
#
# If `reportseff` is installed, the whole job's aggregate efficiency is printed too, as a cross-check.
set -euo pipefail

JOBID="${1:?usage: $0 <jobid>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO_ROOT/benchmark/results/slurm/matrix_$JOBID"
[ -d "$LOG_DIR" ] || { echo "### no logs for job $JOBID in $LOG_DIR" >&2; exit 1; }

printf "%-36s %10s %8s %10s\n" "task" "elapsed" "cpu%" "maxrss"
for log in "$LOG_DIR"/*.log; do
    [ -e "$log" ] || continue
    tag="$(basename "$log" .log)"
    elapsed=$(grep -m1 "Elapsed (wall clock) time" "$log" | sed 's/.*): //' || true)
    cpu=$(grep -m1 "Percent of CPU this job got" "$log" | sed 's/.*: //' || true)
    maxrss_kb=$(grep -m1 "Maximum resident set size" "$log" | sed 's/.*: //' || true)
    if [ -z "$elapsed" ]; then
        printf "%-36s %s\n" "$tag" "(no /usr/bin/time output: the task likely failed, check $log)"
        continue
    fi
    maxrss_gb=$(awk "BEGIN { printf \"%.1fG\", ${maxrss_kb:-0} / 1048576 }")
    printf "%-36s %10s %8s %10s\n" "$tag" "$elapsed" "$cpu" "$maxrss_gb"
done

if command -v reportseff >/dev/null; then
    echo
    echo "### whole-job aggregate (reportseff)"
    reportseff --format=JobID,JobName,Elapsed,TotalCPU,CPUEff,MemEff,State "$JOBID"
fi
