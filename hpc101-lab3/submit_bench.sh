#!/bin/bash
#HPC -p lab3
#HPC -c 8
#HPC -g 1
#HPC -t 4m
#HPC --output bench_result.out
cd ~/hpc101-lab3
export PATH=/opt/lab3-venv/bin:$PATH
export TILELANG_CACHE_DIR=/opt/lab3-cache/tilelang
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" 2>&1
git pull origin main 2>&1
python3 inspect_policy.py 2>&1
python3 run.py --output-format csv 2>&1
echo BENCHMARK_DONE
