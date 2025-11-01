import cdsapi
from pathlib import Path
import time

# -----------------------------
# 配置
# -----------------------------
dataset = "reanalysis-era5-single-levels-monthly-means"

# 变量列表；确保 total_precipitation 放在第一位
variables_all = [
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "mean_wave_direction",
    "mean_wave_period",
    "sea_surface_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
    "surface_pressure",
]

# 年份范围
years_all = list(map(str, range(1940, 2026)))     # 1940–2025
years_priority = list(map(str, range(1980, 2026))) # 1980–2025（优先）
years_rest = [y for y in years_all if y not in years_priority]

months = [f"{m:02d}" for m in range(1, 13)]
time_utc = ["00:00"]  # ERA5 monthly means 要求的时刻

# 输出目录 & 文件名模板
out_dir = Path("downloads/era5-monthly")
out_dir.mkdir(parents=True, exist_ok=True)
fname = lambda var, year: out_dir / f"era5-monthly_{var}_{year}.nc"

# 下载重试设置（CDS 偶发 5xx 或连接问题时很有用）
max_retries = 5
backoff_sec = 15

# -----------------------------
# 下载函数：一次仅下载 1 年 × 1 变量
# -----------------------------
def download_one(var: str, year: str, client: cdsapi.Client):
    target_path = fname(var, year)

    # 如果文件已存在则跳过
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[SKIP] {target_path} 已存在")
        return

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [var],
        "year": [year],
        "month": months,
        "time": time_utc,
        # 保留你原脚本中的字段名
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    # 带重试的下载
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{var} {year}] 正在请求（第 {attempt}/{max_retries} 次）...")
            result = client.retrieve(dataset, request)
            result.download(str(target_path))  # 指定目标文件名
            # 简单校验
            if not target_path.exists() or target_path.stat().st_size == 0:
                raise RuntimeError("下载后文件为空或不存在")
            print(f"[OK] 已保存：{target_path}")
            return
        except Exception as e:
            print(f"[WARN] 下载 {var} {year} 失败：{e}")
            if attempt < max_retries:
                print(f"  将在 {backoff_sec}s 后重试...")
                time.sleep(backoff_sec)
            else:
                print(f"[FAIL] 放弃 {var} {year}")

# -----------------------------
# 主流程
# -----------------------------
if __name__ == "__main__":
    client = cdsapi.Client()

    # 变量顺序：先 total_precipitation，再其它变量
    variables_ordered = (
        ["total_precipitation"]
        + [v for v in variables_all if v != "total_precipitation"]
    )

    # 1) 先下载 1980–2025（所有变量，但先降水）
    for var in variables_ordered:
        for year in years_priority:
            download_one(var, year, client)

    # 2) 再下载其余年份 1940–1979（所有变量，同样先降水）
    for var in variables_ordered:
        for year in years_rest:
            download_one(var, year, client)
