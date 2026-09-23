#!/usr/bin/env bash
set -euo pipefail

cd /home/h3250102096/lab5_work_p1
uv pip install -e . --no-deps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HPC101_PIPELINE_WARMUP=1
export HPC101_INT4_BLOCK_N=4
export HPC101_INT4_GEMV_BLOCK_K=1024
export HPC101_INT4_GEMV_NUM_WARPS=4
export HPC101_DEQUANT_BLOCK_K=64
export HPC101_DEQUANT_NUM_WARPS=8

MODEL_DIR=/home/h3250102096/gemma4-gptq-p3-group64

run_case() {
    local label="$1"
    local chunk="$2"
    local dequant_n="$3"

    echo "===${label}==="
    HPC101_PREFILL_CHUNK_SIZE="${chunk}" \
    HPC101_DEQUANT_BLOCK_N="${dequant_n}" \
    python3 scripts/run_generation_queue.py \
        --model "${MODEL_DIR}" \
        --input datasets/performance_public.jsonl \
        --output "/tmp/lab5-g64-pair-${label}.jsonl" \
        --summary-output "/tmp/lab5-g64-pair-${label}.json" \
        --config config.yaml \
        --linear-backend int4_reference \
        --batch-size 3 \
        --max-batch-size 3 \
        --max-sequence-length 2048 \
        --max-new-tokens 32 \
        --seed 42 \
        --no-progress
}

run_case 1152_n64_run1  1152 64
run_case 1152_n128_run1 1152 128
run_case 1536_n64_run1  1536 64
run_case 1536_n128_run1 1536 128

run_case 1536_n128_run2 1536 128
run_case 1536_n64_run2  1536 64
run_case 1152_n128_run2 1152 128
run_case 1152_n64_run2  1152 64

run_case 1152_n128_run3 1152 128
run_case 1536_n64_run3  1536 64
run_case 1152_n64_run3  1152 64
run_case 1536_n128_run3 1536 128

echo "===SUMMARIES==="
for summary in /tmp/lab5-g64-pair-*.json; do
    printf '%s ' "$(basename "${summary}")"
    cat "${summary}"
done
