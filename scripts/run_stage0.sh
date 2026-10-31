#!/usr/bin/env bash
# Stage 0 rebuild against the checked-in manifests: audit -> dedup -> check -> datasets -> report.
# (freeze-grid and split are one-time commands; they refuse to overwrite existing manifests.)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
$PY -m pytest -q tests
$PY -m rrc audit
$PY -m rrc dedup
$PY -m rrc check
$PY -m rrc datasets
$PY -m rrc report
