#!/usr/bin/env bash
set -euo pipefail

cd /home/h3250102096/lab5_work_p1
uv pip install -e . --no-deps

python3 scripts/evaluate_quality.py \
    --model /home/h3250102096/gemma4-gptq-p3-group64 \
    --dataset datasets/quality_public.jsonl \
    --output /tmp/lab5-g64-final-quality.json \
    --reference results/bf16-public-quality.json \
    --linear-backend int4_reference \
    --device cuda \
    --dtype bfloat16 \
    --max-sequence-length 2048 \
    --chunk-size 128 \
    --no-progress

cat /tmp/lab5-g64-final-quality.json
