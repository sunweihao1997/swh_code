# -*- coding: utf-8 -*-
'''20251021
v2: add time selection function to determine the start-end period
'''
import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pypinyin import lazy_pinyin

# === 配置区（可按需调整默认值）===
EXCEL_PATH = "/mnt/f/数据_东海局/salinity/盐度 含宝钢水库.xlsx"
SHEET_KEYWORD = "宝钢水库"
OUTPUT_DIR = "/mnt/f/wsl_plot/donghai/salinity/baogang_daily_series"
DATE_COL_CANDIDATES = ["time", "datetime", "date", "时间", "日期", "采样时间"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_pinyin(name):
    """中文转无声调拼音，空格变下划线"""
    return "_".join(lazy_pinyin(str(name)))

def find_datetime_col(df):
    """尽量稳健地找出日期时间列"""
    for c in df.columns:
        if str(c).strip().lower() in DATE_COL_CANDIDATES or str(c).strip() in DATE_COL_CANDIDATES:
            return c
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col])
        return first_col
    except Exception:
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                pass
    raise ValueError("未能识别日期时间列，请检查列名或在脚本中设置 DATE_COL_CANDIDATES。")

def to_daily(df):
    """将小时级别数据聚合为日均"""
    dt_col = find_datetime_col(df)
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError("未找到数值列用于计算日均。")
    daily = df[num_cols].resample("D").mean()
    return daily

def slice_by_time(df_daily, start, end):
    """按[start, end]（闭区间）裁剪；参数可为 None"""
    if start is None and end is None:
        return df_daily
    # to_datetime 能识别多种格式，如 2020-1-1 / 2020/01/01 / 2020.01.01
    if start is not None:
        start = pd.to_datetime(start)
    if end is not None:
        end = pd.to_datetime(end)
    # 用 .loc 进行区间筛选
    if start is not None and end is not None:
        return df_daily.loc[start:end]
    elif start is not None:
        return df_daily.loc[start:]
    else:
        return df_daily.loc[:end]

def main():
    parser = argparse.ArgumentParser(description="宝钢水库：小时转日均并绘图（可选时间窗口）")
    parser.add_argument("--excel", default=EXCEL_PATH, help="Excel 文件路径")
    parser.add_argument("--keyword", default=SHEET_KEYWORD, help="工作表名包含的关键字")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--start", default=None, help="起始日期（含），如 2019-01-01")
    parser.add_argument("--end", default=None, help="结束日期（含），如 2024-12-31")
    args = parser.parse_args()

    excel_path = args.excel
    keyword = args.keyword
    outdir = args.outdir
    start = args.start
    end = args.end
    os.makedirs(outdir, exist_ok=True)

    # 读取所有 sheet
    xls = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    target_items = [(name, df) for name, df in xls.items() if keyword in str(name)]
    if not target_items:
        raise ValueError(f"未在工作表名称中找到包含“{keyword}”的 sheet。")

    daily_dict = {}
    for name, df in target_items:
        daily = to_daily(df)
        # 时间裁剪
        daily = slice_by_time(daily, start, end)
        if daily.empty:
            print(f"[警告] {name} 在所选时间范围内无数据，已跳过绘图。")
            continue

        daily_dict[name] = daily

        # sheet 名转拼音
        name_en = to_pinyin(name)

        # 保存聚合结果
        daily_out = os.path.join(outdir, f"{name_en}_daily.csv")
        daily.to_csv(daily_out, encoding="utf-8-sig")

        # —— 绘图（自定义 legend 文本）——
        fig, ax = plt.subplots(figsize=(10, 4))
        # 如果你只有“盐度”这一个变量，也可以直接 label="Salinity"
        for col in daily.columns:
            # 自定义每条线的英文名：用字典映射更自然；缺省则用拼音
            col_map = {"盐度": "Salinity", "温度": "Temperature"}
            label_en = col_map.get(str(col), to_pinyin(col))
            ax.plot(daily.index, daily[col]/1.80655*1000, label=label_en)
            ax.axhline(y=250, color='r', linestyle='--', label='Threshold = 250')

        # 标题可附带所选时间段提示
        time_suffix = ""
        if start or end:
            ts = start if start else str(daily.index.min().date())
            te = end if end else str(daily.index.max().date())
            time_suffix = f" [{ts} ~ {te}]"

        ax.set_title(f"BaoGang - Daily Mean{time_suffix}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value (daily mean)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name_en}_daily.png"), dpi=300)
        plt.close(fig)

    # 合并所有 sheet（如有），再按时间窗口裁剪后总图
    if daily_dict:
        merged = []
        for name, d in daily_dict.items():
            # 给列加前缀（拼音），避免重名
            pref = to_pinyin(name)
            renamed = d.add_prefix(f"{pref}_")
            merged.append(renamed)
        combined = pd.concat(merged, axis=1).sort_index()
        combined_out = os.path.join(outdir, "ALL_sheets_daily.csv")
        combined.to_csv(combined_out, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(12, 5))
        combined.plot(ax=ax, title=f"Sheets with '{keyword}' - Daily Mean (Combined)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value (daily mean)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "ALL_sheets_daily.png"), dpi=300)
        plt.close(fig)

    print(f"完成：结果已输出到 {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()
