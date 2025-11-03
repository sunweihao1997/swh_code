# -*- coding: utf-8 -*-
"""
风速风向：小时 -> 日平均（矢量平均）并绘制“时间轴风矢量图”（每个sheet一张）
用法：
    python wind_daily_stickplot.py --input 风速风向.xlsx --output 风速风向_日平均.xlsx --plots ./wind_daily_plots

说明：
- 自动识别每个sheet中的“时间/风速/风向”列（尽量智能匹配）。
- 日平均采用“矢量平均”：先把(风速, 风向FROM) -> (U, V)；对U、V做逐日平均；再还原到(风速, 风向FROM)。
- 方向默认为“来自（FROM）”的气象学方位（0/360=北，90=东）。如你的数据是“向（TO）”的方向，请把代码里
  wind_from_dir_speed_to_uv 的公式按注释改为 TO 模式。
"""

import os
import math
import argparse
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def infer_datetime_col(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() > 0.6:
            candidates.append((parsed.notna().mean(), col))
    if not candidates:
        common = ["datetime", "time", "date", "日期", "时间", "时刻", "时间戳"]
        for c in common:
            for col in df.columns:
                if c.lower() in str(col).lower():
                    return col
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def infer_speed_col(df: pd.DataFrame) -> Optional[str]:
    priority = ["风速", "速度", "windspeed", "wind_speed", "ws", "speed", "平均风速"]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for key in priority:
        for low, orig in cols_lower.items():
            if key in low:
                if pd.api.types.is_numeric_dtype(df[orig]) or pd.to_numeric(df[orig], errors="coerce").notna().mean() > 0.6:
                    return orig
    for c in df.columns:
        low = str(c).lower()
        if any(u in low for u in ["m/s", "mps", "米每秒"]) and (
            pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.6
        ):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
            return c
    return None

def infer_dir_col(df: pd.DataFrame) -> Optional[str]:
    priority = ["风向", "方向", "winddir", "wind_dir", "wd", "dir", "direction"]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for key in priority:
        for low, orig in cols_lower.items():
            if key in low:
                if pd.api.types.is_numeric_dtype(df[orig]) or pd.to_numeric(df[orig], errors="coerce").notna().mean() > 0.6:
                    return orig
    return None

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def wind_from_dir_speed_to_uv(direction_deg_from: pd.Series, speed: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    来向(气象)：u=-S*sin(theta), v=-S*cos(theta)
    如果你的风向是“去向(TO)”，请改为：
        u = S * np.sin(theta)
        v = S * np.cos(theta)
    """
    theta = np.deg2rad(direction_deg_from % 360.0)
    u = -speed * np.sin(theta)
    v = -speed * np.cos(theta)
    return u, v

def uv_to_speed_dir_from(u: pd.Series, v: pd.Series) -> Tuple[pd.Series, pd.Series]:
    speed = np.sqrt(u**2 + v**2)
    dir_rad = np.arctan2(-u, -v)
    direction_deg = (np.rad2deg(dir_rad) + 360.0) % 360.0
    return speed, direction_deg

def daily_vector_mean(df: pd.DataFrame, tcol: str, speed_col: str, dir_col: str) -> pd.DataFrame:
    work = df[[tcol, speed_col, dir_col]].copy()
    work[tcol] = pd.to_datetime(work[tcol], errors="coerce", infer_datetime_format=True)
    work = work.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)

    spd = to_numeric_safe(work[speed_col])
    direction = to_numeric_safe(work[dir_col])

    u, v = wind_from_dir_speed_to_uv(direction, spd)
    vec = pd.DataFrame({"U": u, "V": v}, index=work.index)
    daily_uv = vec.resample("D").mean()

    daily_speed, daily_dir = uv_to_speed_dir_from(daily_uv["U"], daily_uv["V"])
    out = pd.DataFrame({"U": daily_uv["U"], "V": daily_uv["V"], "Speed": daily_speed, "DirFrom": daily_dir}).dropna(how="all")
    return out

def make_stick_quiver(ax, dates_num: np.ndarray, u: np.ndarray, v: np.ndarray, title: str):
    import matplotlib.dates as mdates
    y = np.zeros_like(dates_num)
    max_speed = float(np.nanmax(np.sqrt(u**2 + v**2))) if len(u) else 1.0
    if not np.isfinite(max_speed) or max_speed == 0:
        max_speed = 1.0
    ylim = max(2.0, max_speed * 1.5)
    ax.set_ylim(-ylim, ylim)
    scale = max_speed / 1.0  # 控制箭头长度比例
    ax.quiver(dates_num, y, u, v, angles="xy", scale_units="xy", scale=scale*10.0, pivot="middle",
              width=0.0025, headwidth=4, headlength=5, headaxislength=4)
    ax.set_title(title)
    ax.set_ylabel("矢量幅度 (≈ m/s)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

def process_sheet(name: str, df: pd.DataFrame, plots_dir: str):
    tcol = infer_datetime_col(df)
    spd_col = infer_speed_col(df)
    dir_col = infer_dir_col(df)

    if tcol is None or spd_col is None or dir_col is None:
        print(f"[警告] 工作表《{name}》无法自动识别列。时间列:{tcol}, 风速列:{spd_col}, 风向列:{dir_col}。已跳过。")
        return None

    daily = daily_vector_mean(df, tcol, spd_col, dir_col)

    if not daily.empty:
        import matplotlib.dates as mdates
        dates_num = mdates.date2num(daily.index.to_pydatetime())
        u = daily["U"].to_numpy(dtype=float)
        v = daily["V"].to_numpy(dtype=float)
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
        make_stick_quiver(ax, dates_num, u, v, f"{name}：日平均风（矢量）")
        plt.tight_layout()
        os.makedirs(plots_dir, exist_ok=True)
        png_path = os.path.join(plots_dir, f"{name}_daily_wind.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        print(f"[提示] 工作表《{name}》日平均后为空。")
    return daily

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="输入的Excel文件（包含多个站点sheet）")
    parser.add_argument("--output", "-o", default="风速风向_日平均.xlsx", help="输出的日平均Excel")
    parser.add_argument("--plots", "-p", default="wind_daily_plots", help="输出图像文件夹")
    args = parser.parse_args()

    xls = pd.ExcelFile(args.input)
    sheets = xls.sheet_names

    daily_results: Dict[str, pd.DataFrame] = {}
    for sh in sheets:
        df = pd.read_excel(args.input, sheet_name=sh)
        daily_df = process_sheet(sh, df, args.plots)
        if daily_df is not None and not daily_df.empty:
            daily_results[sh] = daily_df

    if daily_results:
        with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
            for sh, ddf in daily_results.items():
                ddf.to_excel(writer, sheet_name=sh)
        print(f"已保存日平均数据到：{args.output}")
        print(f"图像输出目录：{os.path.abspath(args.plots)}")
    else:
        print("未生成任何日平均结果，请检查数据列名是否可识别（时间/风速/风向）。")

if __name__ == "__main__":
    main()
