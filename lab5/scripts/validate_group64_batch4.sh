#!/usr/bin/env bash
set -uo pipefail

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

MODEL_DIR=/home/h3250102096/gemma4-gptq-p3-group64

run_case() {
    local label="$1"
    local chunk="$2"

    echo "===${label}==="
    if HPC101_PREFILL_CHUNK_SIZE="${chunk}" \
    python3 scripts/run_generation_queue.py \
        --model "${MODEL_DIR}" \
        --input datasets/performance_public.jsonl \
        --output "/tmp/lab5-g64-${label}.jsonl" \
        --summary-output "/tmp/lab5-g64-${label}.json" \
        --config config.yaml \
        --linear-backend int4_reference \
        --batch-size 4 \
        --max-batch-size 4 \
        --max-sequence-length 2048 \
        --max-new-tokens 32 \
        --seed 42 \
        --no-progress; then
        echo "===${label}_PASS==="
    else
        echo "===${label}_FAILED==="
    fi
}

run_case batch4_chunk512 512
run_case batch4_chunk768 768
run_case batch4_chunk1024 1024
run_case batch4_chunk1152 1152

echo "===SUMMARIES==="
for summary in /tmp/lab5-g64-batch4_chunk*.json; do
    if [ -f "${summary}" ]; then
        printf '%s ' "$(basename "${summary}")"
        cat "${summary}"
    fi
done
