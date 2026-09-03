#!/usr/bin/env bash
set -euo pipefail

cd /home/h3250102096/lab5_work_p1
uv pip install -e . --no-deps
unset HPC101_PREFILL_CHUNK_SIZE
unset HPC101_INT4_BLOCK_N
unset HPC101_INT4_GEMV_BLOCK_K
unset HPC101_INT4_GEMV_NUM_WARPS
unset HPC101_DEQUANT_BLOCK_N
unset HPC101_DEQUANT_BLOCK_K
unset HPC101_DEQUANT_NUM_WARPS
unset HPC101_FUSED_GATE_UP

python -m pytest -q tests/test_gptq.py

MODEL_DIR=/home/h3250102096/gemma4-gptq-p3-group64

for run in 1 2 3; do
    echo "===default_run${run}==="
    python3 scripts/run_generation_queue.py \
        --model "${MODEL_DIR}" \
        --input datasets/performance_public.jsonl \
        --output "/tmp/lab5-g64-final-default-run${run}.jsonl" \
        --summary-output "/tmp/lab5-g64-final-default-run${run}.json" \
        --config config.yaml \
        --linear-backend int4_reference \
        --max-sequence-length 2048 \
        --max-new-tokens 32 \
        --seed 42 \
        --no-progress
done

echo "===SUMMARIES==="
for summary in /tmp/lab5-g64-final-default-run*.json; do
    printf '%s ' "$(basename "${summary}")"
    cat "${summary}"
done
