#!/usr/bin/env bash
# Run one isolated CPU AMSS-NCKU experiment. Intended for execution on a lab4
# compute node (directly or through submit_cpu_experiment.sh).
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPI_RANKS=6
OMP_THREADS=1
TFINAL=5
LABEL=""
BUILD_DIR=""
OUTPUT_ROOT=""
DO_COMPILE=0
ENABLE_OPENMP=0
ENABLE_PROFILE=0
ENABLE_OMP_KERNELS=0
ENABLE_TWOP_OMP=0
ENABLE_INTERP_OMP=0
MPIEXEC_VALUE="${AMSS_MPIEXEC:-mpiexec --allow-run-as-root}"

usage() {
  cat <<'EOF'
Usage: scripts/run_cpu_experiment.sh [options]

  --mpi N              MPI ranks (default: 6)
  --omp N              OpenMP threads per rank (default: 1)
  --tfinal T           CPU final evolution time (default: 5; formal run: 40)
  --label NAME         output/build label (default includes MPI/OMP/tfinal/time)
  --build-dir PATH     isolated CMake build directory (default: build-<label>)
  --output-root PATH   isolated output parent (default: experiments/<label>)
  --compile            configure and build before running
  --openmp             configure with AMSS_ENABLE_OPENMP=ON
  --profile            configure with AMSS_ENABLE_PROFILE=ON and enable driver logs
  --omp-kernels        enable the experimental Fortran OpenMP regions (requires --openmp)
  --twop-omp           enable experimental TwoPuncture OpenMP loops (requires --openmp)
  --interp-omp         enable experimental CPU interpolation OpenMP loops (requires --openmp)
  --mpiexec VALUE      launcher plus optional arguments, e.g. "mpiexec --report-bindings"
  -h, --help           show this help

The script never enables TwoPuncture cache. The official t=40 timing must use
--tfinal 40 and should normally use --compile with an isolated build directory.
EOF
}

while (( $# )); do
  case "$1" in
    --mpi) MPI_RANKS="$2"; shift 2 ;;
    --omp) OMP_THREADS="$2"; shift 2 ;;
    --tfinal) TFINAL="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --compile) DO_COMPILE=1; shift ;;
    --openmp) ENABLE_OPENMP=1; shift ;;
    --profile) ENABLE_PROFILE=1; shift ;;
    --omp-kernels) ENABLE_OMP_KERNELS=1; shift ;;
    --twop-omp) ENABLE_TWOP_OMP=1; shift ;;
    --interp-omp) ENABLE_INTERP_OMP=1; shift ;;
    --mpiexec) MPIEXEC_VALUE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$MPI_RANKS" =~ ^[1-9][0-9]*$ ]] || { echo "--mpi must be a positive integer" >&2; exit 2; }
[[ "$OMP_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "--omp must be a positive integer" >&2; exit 2; }
[[ "$TFINAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "--tfinal must be numeric" >&2; exit 2; }
(( ENABLE_OMP_KERNELS == 0 || ENABLE_OPENMP == 1 )) || { echo "--omp-kernels requires --openmp" >&2; exit 2; }
(( ENABLE_TWOP_OMP == 0 || ENABLE_OPENMP == 1 )) || { echo "--twop-omp requires --openmp" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$LABEL" ]]; then
  LABEL="mpi${MPI_RANKS}_omp${OMP_THREADS}_t${TFINAL}_${STAMP}"
fi
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-$LABEL}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/experiments/$LABEL}"
mkdir -p "$OUTPUT_ROOT"

meta="$OUTPUT_ROOT/environment.txt"
{
  echo "timestamp_utc=$(date -u --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "pwd=$ROOT_DIR"
  echo "mpi_ranks=$MPI_RANKS"
  echo "omp_threads=$OMP_THREADS"
  echo "cpu_tfinal=$TFINAL"
  echo "build_dir=$BUILD_DIR"
  echo "output_root=$OUTPUT_ROOT"
  echo "mpiexec=$MPIEXEC_VALUE"
  echo "amss_opt=${AMSS_OPT:--O3}"
  echo "openmp=$ENABLE_OPENMP"
  echo "profile=$ENABLE_PROFILE"
  echo "omp_kernels=$ENABLE_OMP_KERNELS"
  echo "twop_omp=$ENABLE_TWOP_OMP"
  echo "--- uname ---"; uname -a
  echo "--- cgroup cpu ---"; cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
  echo "--- cpuset ---"; cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true
  echo "--- numactl ---"; command -v numactl >/dev/null && numactl --hardware || true
  echo "--- lscpu ---"; command -v lscpu >/dev/null && lscpu || true
  echo "--- MPI ---"; ${MPIEXEC_VALUE%% *} --version 2>&1 | head -20 || true
  echo "--- shm before ---"; df -h /dev/shm || true; find /dev/shm -maxdepth 1 -name 'sm_segment.*' -printf '%f\n' 2>/dev/null | wc -l || true
} > "$meta"

if (( DO_COMPILE )); then
  (
    cd "$ROOT_DIR"
    AMSS_BUILD_DIR="$BUILD_DIR" \
    AMSS_ENABLE_OPENMP="$ENABLE_OPENMP" \
    ./compile.sh "-DAMSS_ENABLE_PROFILE=$ENABLE_PROFILE" "-DAMSS_ENABLE_OMP_KERNELS=$ENABLE_OMP_KERNELS" "-DAMSS_ENABLE_TWOP_OMP=$ENABLE_TWOP_OMP" "-DAMSS_ENABLE_INTERP_OMP=$ENABLE_INTERP_OMP"
  ) 2>&1 | tee "$OUTPUT_ROOT/build.log"
fi

[[ -x "$BUILD_DIR/ABE" ]] || { echo "Missing $BUILD_DIR/ABE; pass --compile or --build-dir for a built tree." >&2; exit 2; }
[[ -x "$BUILD_DIR/TwoPunctureABE" ]] || { echo "Missing $BUILD_DIR/TwoPunctureABE." >&2; exit 2; }

started="$(date +%s)"
set +e
(
  cd "$ROOT_DIR"
  export AMSS_BUILD_DIR="$BUILD_DIR"
  export AMSS_OUTPUT_ROOT="$OUTPUT_ROOT"
  export AMSS_MPIEXEC="$MPIEXEC_VALUE"
  export AMSS_MPI_PROCESSES="$MPI_RANKS"
  export AMSS_OMP_THREADS="$OMP_THREADS"
  export AMSS_CPU_FINAL_EVOLUTION_TIME="$TFINAL"
  export AMSS_PROFILE="$ENABLE_PROFILE"
  export OMP_NUM_THREADS="$OMP_THREADS"
  ./run.sh
) 2>&1 | tee "$OUTPUT_ROOT/run.log"
run_status=${PIPESTATUS[0]}
set -e
ended="$(date +%s)"
printf 'runner_wall_s=%s\nrun_exit_code=%s\n' "$((ended - started))" "$run_status" | tee -a "$meta"

result_dir="$OUTPUT_ROOT/GW250118/AMSS_NCKU_output"
if (( run_status == 0 )) && [[ -d "$result_dir" ]]; then
  (
    cd "$ROOT_DIR"
    if [[ "$TFINAL" == "40" || "$TFINAL" == "40.0" ]]; then
      bash ./check.sh "$result_dir" "$ROOT_DIR/golden"
    else
      python3 ./scripts/check_result.py --allow-partial "$result_dir" "$ROOT_DIR/golden"
    fi
  ) 2>&1 | tee "$OUTPUT_ROOT/check.log"
else
  echo "Checker skipped: run failed or result directory is absent: $result_dir" | tee "$OUTPUT_ROOT/check.log"
fi

{
  echo "--- shm after ---"; df -h /dev/shm || true; find /dev/shm -maxdepth 1 -name 'sm_segment.*' -printf '%f\n' 2>/dev/null | wc -l || true
  echo "--- profile summary ---"; grep 'AMSS_PROFILE' "$OUTPUT_ROOT/run.log" || true
  echo "--- driver stage summary ---"; grep 'AMSS_DRIVER_PROFILE' "$OUTPUT_ROOT/run.log" || true
  echo "--- program cost ---"; grep 'This Program Cost' "$OUTPUT_ROOT/run.log" || true
} >> "$meta"

exit "$run_status"
