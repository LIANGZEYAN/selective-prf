#!/bin/bash
# Run missing Mistral-Small-24B experiments sequentially:
# Pairwise + Setwise + SC-Pairwise + SC-Setwise (workers 19, 20)

set -e
cd "/mnt/primary/QE audit/rankllm"

echo "========================================"
echo " [1/4] Pairwise-Mistral-24B (posthoc)"
echo "========================================"
python scripts/run_all_posthoc.py --worker 19

echo "========================================"
echo " [2/4] Setwise-Mistral-24B (posthoc)"
echo "========================================"
python scripts/run_all_posthoc.py --worker 20

echo "========================================"
echo " [3/4] SC-Pairwise-Mistral-24B (SC)"
echo "========================================"
python scripts/run_all_sc.py --worker 19

echo "========================================"
echo " [4/4] SC-Setwise-Mistral-24B (SC)"
echo "========================================"
python scripts/run_all_sc.py --worker 20

echo "========================================"
echo " ALL MISTRAL-24B JOBS COMPLETE"
echo "========================================"
