#!/usr/bin/env bash
set -euo pipefail

cd /home/h3250102096/lab5_work_p1
uv pip install -e . --no-deps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HPC101_PIPELINE_WARMUP=1
export HPC101_INT4_BLOCK_N=4
export HPC101_INT4_GEMV_BLOCK_K=1024
export HPC101_INT4_GEMV_NUM_WARPS=4
export HPC101_DEQUANT_BLOCK_N=128
export HPC101_DEQUANT_BLOCK_K=64
export HPC101_DEQUANT_NUM_WARPS=8
export HPC101_PREFILL_CHUNK_SIZE=512
export HPC101_FUSED_GATE_UP=1

MODEL_DIR=/home/h3250102096/gemma4-gptq-p3-group64

for run in 1 2 3; do
    echo "===fused_run${run}==="
    python3 scripts/run_generation_queue.py \
        --model "${MODEL_DIR}" \
        --input datasets/performance_public.jsonl \
        --output "/tmp/lab5-g64-b4-fused-run${run}.jsonl" \
        --summary-output "/tmp/lab5-g64-b4-fused-run${run}.json" \
        --config config.yaml \
        --linear-backend int4_reference \
        --batch-size 4 \
        --max-batch-size 4 \
        --max-sequence-length 2048 \
        --max-new-tokens 32 \
        --seed 42 \
        --no-progress
done

echo "===SUMMARIES==="
for summary in /tmp/lab5-g64-b4-fused-run*.json; do
    printf '%s ' "$(basename "${summary}")"
    cat "${summary}"
done
