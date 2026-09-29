#!/usr/bin/env bash
# run_ns.sh — Main entry point for LJ NS phase diagram runs
#
# Usage:
#   ./scripts/run_ns.sh [CONFIG] [CORES]
#
#   CONFIG  : Path to JSON config (default: config/default.json)
#   CORES   : Number of worker processes (default: 64, clamped to 64 max)
#
# This script enforces the 64-core ceiling (machine has 128, 64 reserved)
# and sets threading env-vars to prevent over-subscription.

set -euo pipefail

# ---------- Configuration ----------
CONFIG="${1:-config/default.json}"
CORES="${2:-128}"
MAX_CORES=128

# Resolve repo root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ---------- Core ceiling ----------
if (( CORES > MAX_CORES )); then
    echo "WARN: Requested $CORES cores, but machine cap is $MAX_CORES (128 logical, 64 reserved)."
    echo "      Clamping to $MAX_CORES."
    CORES=$MAX_CORES
fi

# ---------- Thread limiting ----------
# Prevent OpenMP/MKL/OpenBLAS from spawning their own threads inside workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---------- Sanity checks ----------
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

if [[ ! -f "src/ljns/cli.py" ]]; then
    echo "ERROR: Must be run from the lj_ns_phase_diagram repo root" >&2
    exit 1
fi

# ---------- Run ----------
echo "==> LJ Bulk Fluid Phase Diagram via Nested Sampling"
echo "    Config : $CONFIG"
echo "    Cores  : $CORES / $MAX_CORES max"
echo "    Host   : $(hostname) ($(nproc) logical cores)"
echo ""

python -m ljns.cli --config "$CONFIG" --cores "$CORES"

echo ""
echo "==> Done. Results in $(grep -E '^ *\"output_dir\"' "$CONFIG" | cut -d'"' -f4)"
