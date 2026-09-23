from ssh_helper import run_remote, submit_job

# Create inspect_bench.sh on the pod
bench_script = """#!/bin/bash
export PATH=/opt/lab3-venv/bin:$PATH
export TILELANG_CACHE_DIR=/opt/lab3-cache/tilelang
cd ~/hpc101-lab3
python ~/inspect_tl2.py
"""
out, err = run_remote(f"cat > ~/inspect_bench.sh << 'HEREDOC'\n{bench_script}HEREDOC", timeout=30)
print("Create:", out.strip())

result = submit_job("inspect_bench.sh")
print("Submit:", result)
