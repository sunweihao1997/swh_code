import os
import schedule
import subprocess
import time
from datetime import datetime

# 替换为你自己的脚本路径
script1 = "/home/ubuntu/swh_code/other/pybroker/cal_52week_sequence_hk_tencent2_250825.py"
script2 = "/home/ubuntu/swh_code/other/pybroker/monitor_hkstock_status_Tencent2_250825.py"
log_file = "/home/ubuntu/stock_data/log/HK_stock_log.txt"

# Conda 配置
conda_path = "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
conda_env = "sun"

os.makedirs(os.path.dirname(log_file), exist_ok=True)

def run_command_stream(script_path, f):
    """
    通过 bash -lc 在非交互 shell 中 source conda 脚本并激活环境，
    用 Python -u 取消缓冲，Popen 按行读取并实时写入日志。
    """
    bash_cmd = (
        f"source {conda_path} && "
        f"conda activate {conda_env} && "
        f"python -u {script_path}"
    )

    # 使用 bash -lc 确保 'source' 与 'conda activate' 可用
    p = subprocess.Popen(
        ["bash", "-lc", bash_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # 行缓冲
    )

    # 实时把子进程输出写入日志
    for line in p.stdout:
        # 可选：为每行加上时间戳（注释掉的两行二选一）
        # ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # f.write(f"[{ts}] {line}")
        f.write(line)
        f.flush()

    return p.wait()

def run_scripts():
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"\n=== [{start_ts}] Starting Scripts ===\n")

        # 第一个脚本（先写标记，再启动，便于排查）
        f.write(f"\n--- Running: {script1} ---\n")
        f.flush()
        rc1 = run_command_stream(script1, f)
        f.write(f"--- Exit code: {rc1} ---\n")
        f.flush()

        f.write(f"\n--- Sleeping 6 hours before next script ---\n")
        f.flush()
        time.sleep(5*60)  # 睡眠6小时

        # 第二个脚本
        f.write(f"\n--- Running: {script2} ---\n")
        f.flush()
        rc2 = run_command_stream(script2, f)
        f.write(f"--- Exit code: {rc2} ---\n")

        f.write(f"\n=== [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished ===\n")
        f.flush()

# 设置调度
schedule.every().day.at("14:00").do(run_scripts)
# schedule.every().wednesday.at("02:00").do(run_scripts)
# schedule.every().friday.at("02:00").do(run_scripts)

print("Scheduler is running... (Press Ctrl+C to stop)")

while True:
    schedule.run_pending()
    time.sleep(30)
