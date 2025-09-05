import schedule
import subprocess
import time
from datetime import datetime

# 替换为你自己的脚本路径
script1 = "/home/sun/swh_code/other/pybroker/cal_52week_sequence_250803.py"
script2 = "/home/sun/swh_code/other/pybroker/monitor_stock_status_250806.py"
log_file = "/home/sun/data/other/log/stock_log.txt"

# Conda 配置
conda_path = "/home/sun/miniconda3/etc/profile.d/conda.sh"
conda_env = "akshare"

def run_command(script_path):
    # 这个命令：激活 conda -> 切换环境 -> 运行脚本
    command = f'''
    source {conda_path} && conda activate {conda_env} && python {script_path}
    '''
    return subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True)

def run_scripts():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"\n=== [{now}] Starting Scripts ===\n")

        # 第一个脚本
        result = run_command(script1)
        f.write(f"\n--- Running: {script1} ---\n")
        f.write(result.stdout)
        f.write(result.stderr)

        f.write(f"\n--- Sleeping 6 hours before next script ---\n")
        f.flush()  # 确保日志立即写入
        time.sleep(6 * 60 * 60)  # 睡眠6小时

        # 第二个脚本
        result = run_command(script2)
        f.write(f"\n--- Running: {script2} ---\n")
        f.write(result.stdout)
        f.write(result.stderr)

        f.write(f"\n=== [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished ===\n")

# 设置调度
# 设置调度：每天 12:00 运行
schedule.every().day.at("09:15").do(run_scripts)

print("Scheduler is running... (Press Ctrl+C to stop)")

while True:
    schedule.run_pending()
    time.sleep(30)
