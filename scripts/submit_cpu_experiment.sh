#!/usr/bin/env bash
# Submit one CPU experiment to the Lab4 scheduler. All remaining arguments are
# forwarded to run_cpu_experiment.sh.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPU_CORES="${AMSS_EXPERIMENT_CPUS:-30}"
MEMORY="${AMSS_EXPERIMENT_MEMORY:-100Gi}"
WALLTIME="${AMSS_EXPERIMENT_WALLTIME:-30m}"
NAME="${AMSS_EXPERIMENT_NAME:-amss-cpu}"

if ! command -v hpc >/dev/null; then
  echo "hpc command is required; run this script from the cluster DevPod." >&2
  exit 127
fi

exec hpc submit -p lab4 -c "$CPU_CORES" -m "$MEMORY" -t "$WALLTIME" \
  -n "$NAME" --chdir "$ROOT_DIR" -- \
  bash "$ROOT_DIR/scripts/run_cpu_experiment.sh" "$@"
