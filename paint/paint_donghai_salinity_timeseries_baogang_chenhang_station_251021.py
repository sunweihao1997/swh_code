# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pypinyin import lazy_pinyin

# === 配置区 ===
EXCEL_PATH = "/mnt/f/数据_东海局/salinity/盐度 含宝钢水库.xlsx"   # 或者改成你的绝对路径
SHEET_KEYWORD = "宝钢水库"
OUTPUT_DIR = "/mnt/f/wsl_plot/donghai/salinity/baogang_daily_series"
DATE_COL_CANDIDATES = ["time", "datetime", "date", "时间", "日期", "采样时间"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_pinyin(name):
    """把中文转成无声调拼音，空格改成下划线"""
    return "_".join(lazy_pinyin(str(name)))

def find_datetime_col(df):
    """尽量稳健地找出日期时间列：优先常见列名；否则尝试把首列解析为时间。"""
    for c in df.columns:
        if str(c).strip().lower() in DATE_COL_CANDIDATES or str(c).strip() in DATE_COL_CANDIDATES:
            return c
    # 尝试把第一列解析为时间
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col])
        return first_col
    except Exception:
        # 遍历所有列，找能被解析为时间的
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                pass
    raise ValueError("未能识别日期时间列，请检查列名或在脚本中设置 DATE_COL_CANDIDATES。")

def to_daily(df):
    """将小时级别数据聚合为日均。"""
    dt_col = find_datetime_col(df)
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    # 仅选择数值列进行聚合
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError("未找到数值列用于计算日均。")

    daily = df[num_cols].resample("D").mean()
    return daily

def main():
    # 读取所有sheet
    xls = pd.read_excel(EXCEL_PATH, sheet_name=None, engine="openpyxl")
    target_items = [(name, df) for name, df in xls.items() if SHEET_KEYWORD in str(name)]

    if not target_items:
        raise ValueError(f"未在工作表名称中找到包含“{SHEET_KEYWORD}”的sheet。")

    daily_dict = {}

    for name, df in target_items:
        daily = to_daily(df)
        daily_dict[name] = daily

        # 将 sheet 名转拼音
        name_en = to_pinyin(name)

        # 保存聚合结果
        daily_out = os.path.join(OUTPUT_DIR, f"{name_en}_daily.csv")
        daily.to_csv(daily_out, encoding="utf-8-sig")

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 4))
        for col in daily.columns:
            label_en = to_pinyin(col)  # 或用手动映射字典，比如 {"盐度": "salinity"}[col]
            ax.plot(daily.index, daily[col], label="Salinity")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value (daily mean)")
        ax.legend(title="BaoGang", loc="best")  # 图例用拼音列名
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{name_en}_daily.png"), dpi=300)
        plt.close(fig)

    # 合并所有sheet，做一张总图（按列名区分）
    # 给每个sheet的列加上前缀，避免列名冲突
    merged = []
    for name, d in daily_dict.items():
        renamed = d.add_prefix(f"{name}_")
        merged.append(renamed)
    combined = pd.concat(merged, axis=1)

    combined_out = os.path.join(OUTPUT_DIR, "ALL_sheets_daily.csv")
    combined.to_csv(combined_out, encoding="utf-8-sig")

    ax = combined.plot(figsize=(12, 5), title=f"Sheets with '{SHEET_KEYWORD}' - Daily Mean (Combined)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (daily mean)")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "ALL_sheets_daily.png"), dpi=150)
    plt.close(fig)

    print(f"完成：结果已输出到 {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
