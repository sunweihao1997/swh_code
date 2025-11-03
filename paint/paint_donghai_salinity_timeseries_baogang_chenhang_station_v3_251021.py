# -*- coding: utf-8 -*-
'''20251021
v4: time selection + interpolate missing + shaded background > threshold + export shaded intervals
'''
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pypinyin import lazy_pinyin
import matplotlib.dates as mdates

# === 配置区（可按需调整默认值）===
EXCEL_PATH = "/mnt/f/数据_东海局/salinity/盐度 含宝钢水库.xlsx"
SHEET_KEYWORD = "宝钢水库"
OUTPUT_DIR = "/mnt/f/wsl_plot/donghai/salinity/baogang_daily_series"
DATE_COL_CANDIDATES = ["time", "datetime", "date", "时间", "日期", "采样时间"]

# —— 数值转换与阈值 —— #
CONVERT_FACTOR = 1.80655            # 分母
CONVERT_MULTIPLIER = 1000.0         # 乘数
THRESHOLD = 250.0                   # 阈值（转换后单位下）

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
    """将小时级数据聚合为日均，并**对齐日频 + 插值补齐**"""
    dt_col = find_datetime_col(df)
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError("未找到数值列用于计算日均。")

    daily = df[num_cols].resample("D").mean()

    # 对齐完整日频并插值
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    daily = daily.interpolate(method="time")
    # 如希望首尾也填充，可启用：
    # daily = daily.ffill().bfill()

    daily.index.name = dt_col
    return daily

def slice_by_time(df_daily, start, end):
    """按 [start, end]（闭区间）裁剪；参数可为 None"""
    if start is None and end is None:
        return df_daily
    if start is not None:
        start = pd.to_datetime(start)
    if end is not None:
        end = pd.to_datetime(end)
    if start is not None and end is not None:
        return df_daily.loc[start:end]
    elif start is not None:
        return df_daily.loc[start:]
    else:
        return df_daily.loc[:end]

def convert_values(df):
    """统一做数值转换：df/CONVERT_FACTOR*CONVERT_MULTIPLIER"""
    return df / CONVERT_FACTOR * CONVERT_MULTIPLIER

def shade_exceed_regions(ax, series_bool, alpha=0.12):
    """
    对布尔序列为 True 的连续时间段做竖向背景着色。
    series_bool：以日期索引的布尔序列（True 表示超阈值）。
    """
    if series_bool.empty:
        return
    grp = (series_bool != series_bool.shift()).cumsum()
    for _, seg in series_bool.groupby(grp):
        if not seg.iloc[0]:  # 只填 True 的段
            continue
        start = seg.index[0]
        end = seg.index[-1]
        ax.axvspan(start, end, alpha=alpha)

def bool_to_intervals(series_bool):
    """
    把逐日布尔序列（True=超阈值）转换为若干个区间（start_date, end_date, duration_days）。
    返回 DataFrame：columns = ["start_date", "end_date", "duration_days"]
    """
    out = []
    if series_bool.empty:
        return pd.DataFrame(columns=["start_date", "end_date", "duration_days"])

    grp = (series_bool != series_bool.shift()).cumsum()
    for _, seg in series_bool.groupby(grp):
        if not seg.iloc[0]:
            continue
        start = seg.index[0]
        end = seg.index[-1]
        duration_days = (end - start).days + 1  # 按日频，包含端点
        out.append({"start_date": start.date(), "end_date": end.date(), "duration_days": duration_days})

    return pd.DataFrame(out)

def main():
    parser = argparse.ArgumentParser(description="宝钢水库：小时转日均并绘图（时间窗口 + 插值 + 超阈值背景 + 区间导出）")
    parser.add_argument("--excel", default=EXCEL_PATH, help="Excel 文件路径")
    parser.add_argument("--keyword", default=SHEET_KEYWORD, help="工作表名包含的关键字")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--start", default=None, help="起始日期（含），如 2019-01-01")
    parser.add_argument("--end", default=None, help="结束日期（含），如 2024-12-31")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="阈值（转换后单位）")
    args = parser.parse_args()

    excel_path = args.excel
    keyword = args.keyword
    outdir = args.outdir
    start = args.start
    end = args.end
    threshold = args.threshold
    os.makedirs(outdir, exist_ok=True)

    # 读取所有 sheet
    xls = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    target_items = [(name, df) for name, df in xls.items() if keyword in str(name)]
    if not target_items:
        raise ValueError(f"未在工作表名称中找到包含“{keyword}”的 sheet。")

    daily_dict = {}
    for name, df in target_items:
        daily = to_daily(df)                # 含插值
        daily = slice_by_time(daily, start, end)
        if daily.empty:
            print(f"[警告] {name} 在所选时间范围内无数据，已跳过绘图。")
            continue

        # 数值转换
        daily_conv = convert_values(daily)
        daily_dict[name] = daily_conv

        # sheet 名转拼音
        name_en = to_pinyin(name)

        # 保存聚合（转换后）结果
        daily_out = os.path.join(outdir, f"{name_en}_daily_converted.csv")
        daily_conv.to_csv(daily_out, encoding="utf-8-sig")

        # 计算超阈值区间并导出
        exceed_any = (daily_conv > threshold).any(axis=1)
        intervals_df = bool_to_intervals(exceed_any)
        intervals_path = os.path.join(outdir, f"{name_en}_exceed_intervals.csv")
        intervals_df.to_csv(intervals_path, index=False, encoding="utf-8-sig")

        # 控制台打印这些区间
        if intervals_df.empty:
            print(f"[{name}] 无超过阈值({threshold:g})的时间段。")
        else:
            print(f"[{name}] 超过阈值({threshold:g})的时间段：")
            for _, r in intervals_df.iterrows():
                print(f"  {r['start_date']} ~ {r['end_date']}  (共 {int(r['duration_days'])} 天)")

        # —— 绘图 —— #
        fig, ax = plt.subplots(figsize=(10, 4))
        col_map = {"盐度": "Salinity", "温度": "Temperature"}
        for col in daily_conv.columns:
            label_en = col_map.get(str(col), to_pinyin(col))
            ax.plot(daily_conv.index, daily_conv[col], label="chlorinity")

        # 阈值线（只画一次）
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1.2, label=f"Threshold = {threshold:g}")
        # 背景着色（根据任一变量超阈值）
        shade_exceed_regions(ax, exceed_any, alpha=0.12)

        # 标题加时间段提示
        time_suffix = ""
        if start or end:
            ts = start if start else str(daily_conv.index.min().date())
            te = end if end else str(daily_conv.index.max().date())
            time_suffix = f" [{ts} ~ {te}]"
        ax.set_title(f"BaoGang - Daily Mean{time_suffix}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value (daily mean)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{name_en}_daily.png"), dpi=300)
        plt.close(fig)

    # 合并所有 sheet（如有），再出一张总图 + 导出区间
    if daily_dict:
        merged = []
        for name, d in daily_dict.items():
            pref = to_pinyin(name)
            renamed = d.add_prefix(f"{pref}_")
            merged.append(renamed)
        combined = pd.concat(merged, axis=1).sort_index()
        combined_out = os.path.join(outdir, "ALL_sheets_daily_converted.csv")
        combined.to_csv(combined_out, encoding="utf-8-sig")

        # 计算“任一列超阈值”的区间并导出
        exceed_any_all = (combined > threshold).any(axis=1)
        all_intervals = bool_to_intervals(exceed_any_all)
        all_intervals_path = os.path.join(outdir, "ALL_sheets_exceed_intervals.csv")
        all_intervals.to_csv(all_intervals_path, index=False, encoding="utf-8-sig")

        if all_intervals.empty:
            print(f"[ALL] 无超过阈值({threshold:g})的时间段。")
        else:
            print(f"[ALL] 超过阈值({threshold:g})的时间段：")
            for _, r in all_intervals.iterrows():
                print(f"  {r['start_date']} ~ {r['end_date']}  (共 {int(r['duration_days'])} 天)")

        # 绘总图
        fig, ax = plt.subplots(figsize=(20, 5))
        combined.plot(ax=ax, title=f"Baogang (Chenghang) Reservoir - Daily Mean")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_ylabel("Value (daily mean)")
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1.2, label=f"Threshold = {threshold:g}")
        shade_exceed_regions(ax, exceed_any_all, alpha=0.10)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "Baogang_salinity_daily.pdf"), dpi=300)
        plt.close(fig)

    print(f"完成：结果已输出到 {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()
