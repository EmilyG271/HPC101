import subprocess
import time

USERNAME = "h3250102096+mypod+hpc101"
KEY_PATH = r"C:\Users\20923\.ssh\id_ed25519"
HOST = "clusters.zju.edu.cn"
PORT = "443"
SSH_BASE = [
    "ssh", "-p", PORT,
    "-4",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=30",
    "-o", "ServerAliveInterval=30",
    "-i", KEY_PATH,
    f"{USERNAME}@{HOST}",
]


def run_remote(cmd, timeout=60):
    full_cmd = SSH_BASE + [cmd]
    result = subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    out = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    err = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    return out, err


def git_pull():
    out, err = run_remote("cd ~/hpc101-lab3 && git fetch origin && git reset --hard origin/main 2>&1")
    return out.strip()


def submit_job(script_name, partition="lab3", time_limit="5m"):
    cmd = f"hpc submit -p {partition} -c 8 -g 1 -m 32Gi -t {time_limit} -d -n bench -o bench.out -- ~/{script_name} 2>&1"
    out, err = run_remote(cmd, timeout=30)
    for line in out.split("\n"):
        if "submitted job" in line:
            return line.strip()
    return out.strip()


def check_job():
    out, err = run_remote("hpc ls -l 5 2>&1")
    return out.strip()


def get_output(filename="bench.out"):
    out, err = run_remote(f"cat ~/{filename} 2>&1")
    return out.strip()


def wait_and_get_output(wait_seconds=120, filename="bench.out"):
    print(f"Waiting {wait_seconds}s...")
    time.sleep(wait_seconds)
    status = check_job()
    print("=== status ===")
    print(status)
    output = get_output(filename)
    print("=== output ===")
    print(output[-8000:])
    return output


def submit_and_wait(script_name="bench.sh", wait_seconds=150):
    out = git_pull()
    print(f"Git pull: {out}")
    result = submit_job(script_name)
    print(f"Submit: {result}")
    return wait_and_get_output(wait_seconds)


if __name__ == "__main__":
    out, err = run_remote("echo CONNECTED && hostname")
    print(f"OUT: {out}")
    print(f"ERR: {err}")
