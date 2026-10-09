#!/usr/bin/env bash
# S1-M1 then S1-M2 under the frozen protocol, then aggregation. Run detached: nohup scripts/run_s1_neural.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export PYTHONWARNINGS=ignore HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "START $(date)"
$PY -m rrc s1-neural --model m1_arabert && echo "M1 DONE $(date)" \
 && $PY -m rrc s1-neural --model m2_camelbert && echo "M2 DONE $(date)" \
 && $PY -m rrc s1-aggregate > /dev/null && echo "AGGREGATE DONE $(date)"
echo "EXIT=$?"
