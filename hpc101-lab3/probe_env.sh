#!/bin/bash
#HPC -p lab3
#HPC -c 8
#HPC -g 1
#HPC -t 4m
#HPC --output probe_env.out
#HPC --chdir hpc101-lab3
echo "=== PYTHON ==="
which python3 python
python3 --version
echo "=== PIP ==="
pip3 --version 2>&1
echo "=== TYPING_EXTENSIONS ==="
python3 -c "import typing_extensions; print(typing_extensions.__version__)" 2>&1
echo "=== TORCH ==="
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1
echo "=== TILELANG ==="
python3 -c "import tilelang; print(tilelang.__version__)" 2>&1
echo "=== NVCC ==="
which nvcc 2>/dev/null || echo "no nvcc"
nvcc --version 2>&1 | head -5
echo "=== SITE-PACKAGES ==="
python3 -c "import site; print(site.getsitepackages())" 2>&1
echo "=== /opt ==="
ls /opt/ 2>/dev/null
echo "=== /usr/local/cuda ==="
ls /usr/local/cuda*/bin/nvcc 2>/dev/null || echo "no cuda dir"
echo PROBE_DONE
