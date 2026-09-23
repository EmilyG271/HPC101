#!/bin/bash
cd ~/hpc101-lab3
export PATH=/opt/lab3-venv/bin:$PATH
export TILELANG_CACHE_DIR=/opt/lab3-cache/tilelang
python inspect_policy.py 2>&1
echo BENCHMARK_DONE
