#!/usr/bin/env bash
set -euo pipefail

cd /home/h3250102096/lab5_work_p1
uv pip install -e . --no-deps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HPC101_PIPELINE_WARMUP=1

MODEL_DIR=/home/h3250102096/gemma4-gptq-p3-group64

run_case() {
    local label="$1"
    local chunk="$2"
    local block_n="$3"
    local block_k="$4"
    local warps="$5"
    local dequant_n="$6"
    local dequant_k="$7"
    local dequant_warps="$8"

    echo "===${label}==="
    HPC101_PREFILL_CHUNK_SIZE="${chunk}" \
    HPC101_INT4_BLOCK_N="${block_n}" \
    HPC101_INT4_GEMV_BLOCK_K="${block_k}" \
    HPC101_INT4_GEMV_NUM_WARPS="${warps}" \
    HPC101_DEQUANT_BLOCK_N="${dequant_n}" \
    HPC101_DEQUANT_BLOCK_K="${dequant_k}" \
    HPC101_DEQUANT_NUM_WARPS="${dequant_warps}" \
    python3 scripts/run_generation_queue.py \
        --model "${MODEL_DIR}" \
        --input datasets/performance_public.jsonl \
        --output "/tmp/lab5-g64-${label}.jsonl" \
        --summary-output "/tmp/lab5-g64-${label}.json" \
        --config config.yaml \
        --linear-backend int4_reference \
        --batch-size 3 \
        --max-batch-size 3 \
        --max-sequence-length 2048 \
        --max-new-tokens 32 \
        --seed 42 \
        --no-progress
}

run_case baseline        1152 4 1024 4 64  64  8
run_case gemv_bn2        1152 2 1024 4 64  64  8
run_case gemv_bn8        1152 8 1024 4 64  64  8
run_case gemv_bk512      1152 4  512 4 64  64  8
run_case gemv_bk2048     1152 4 2048 4 64  64  8
run_case gemv_warps2     1152 4 1024 2 64  64  8
run_case gemv_warps8     1152 4 1024 8 64  64  8
run_case dequant_n128    1152 4 1024 4 128 64  8
run_case dequant_k128    1152 4 1024 4 64  128 8
run_case dequant_nk128   1152 4 1024 4 128 128 8
run_case dequant_warps16 1152 4 1024 4 64  64  16
run_case chunk1536       1536 4 1024 4 64  64  8

echo "===SUMMARIES==="
for summary in /tmp/lab5-g64-*.json; do
    printf '%s ' "$(basename "${summary}")"
    cat "${summary}"
done
