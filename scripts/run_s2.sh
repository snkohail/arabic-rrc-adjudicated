#!/usr/bin/env bash
# S2: lexical baselines (B0/B1/B2), then M1 and M2, then aggregation.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "START $(date)"
$PY -m rrc s2-baselines && echo "BASELINES DONE $(date)" \
 && $PY -m rrc s2-neural --model m1_arabert && echo "M1 DONE $(date)" \
 && $PY -m rrc s2-neural --model m2_camelbert && echo "M2 DONE $(date)" \
 && $PY -m rrc s2-aggregate > /dev/null && echo "AGGREGATE DONE $(date)"
echo "EXIT=$?"
