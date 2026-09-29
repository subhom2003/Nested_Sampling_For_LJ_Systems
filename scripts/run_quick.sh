#!/usr/bin/env bash
# run_quick.sh — Quick smoke test (~5 min)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

./scripts/run_ns.sh config/quick.json 4