# -*- coding: utf-8 -*-
"""
按“日等权”计算多年逐月径流量，但只纳入“该年该月有 >= min_days (默认20) 天”的 year-month。
输入：清洗后的 Excel（包含列：year, month, day, value）
输出：
  1) ym_counts.csv                 —— 各 year-month 的天数统计（诊断用）
  2) ym_kept_min20.csv             —— 被纳入统计的 year-month 列表（阈值可调）
  3) monthly_mean_dayweighted.csv  —— 多年逐月均值（只对纳入的观测做日等权）
"""

import os
import argparse
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # 列名兼容大小写
    cols_lower = {c.lower(): c for c in df.columns}
    need = ["year", "month", "day", "value"]
    for n in need:
        if n not in cols_lower:
            raise ValueError(f"输入文件缺少必要列：{n}")
    df = df[[cols_lower["year"], cols_lower["month"], cols_lower["day"], cols_lower["value"]]].copy()
    df.columns = ["year", "month", "day", "value"]

    # 基本类型转换
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 过滤掉明显非法记录
    df = df[
        df["year"].notna()
        & df["month"].between(1, 12)
        & df["day"].between(1, 31)
        & df["value"].notna()
    ].copy()

    return df


def main():
    ap = argparse.ArgumentParser(description="多年逐月径流量（日等权，按月最少天数筛选）")
    ap.add_argument("input", help="清洗后的 Excel 文件路径（含 year, month, day, value）")
    ap.add_argument("--outdir", default=None, help="输出目录（默认与输入同目录）")
    ap.add_argument("--min-days", type=int, default=20,
                    help="纳入统计的 year-month 最少天数（默认 20）")
    ap.add_argument("--year-min", type=int, default=2000, help="起始年份（默认 2000）")
    ap.add_argument("--year-max", type=int, default=2021, help="结束年份（默认 2021）")
    args = ap.parse_args()

    in_path = args.input
    out_dir = args.outdir or os.path.dirname(os.path.abspath(in_path))
    os.makedirs(out_dir, exist_ok=True)

    # 1) 读数据并按年限过滤
    df = load_data(in_path)
    df = df[df["year"].between(args.year_min, args.year_max)].copy()

    # 2) 诊断：每年、每年每月的记录天数
    year_counts = df.groupby("year").size().sort_index()
    print("每年记录条数：")
    print(year_counts)

    ym_counts = (
        df.groupby(["year", "month"])
          .size()
          .rename("n_days")
          .reset_index()
          .sort_values(["year", "month"])
    )
    ym_counts.to_csv(os.path.join(out_dir, "ym_counts.csv"), index=False, encoding="utf-8-sig")

    # 3) 仅保留 n_days >= min_days 的 year-month
    good_ym = ym_counts[ym_counts["n_days"] >= int(args.min_days)][["year", "month"]]
    good_ym.to_csv(os.path.join(out_dir, "ym_kept_min20.csv"), index=False, encoding="utf-8-sig")

    # 4) 只保留这些 year-month 的原始日值
    df_keep = df.merge(good_ym, on=["year", "month"], how="inner")

    # 5) 按“日等权”计算：同一个月份跨所有被保留的 year-month 的日值合在一起求均值
    clim_dayweighted = (
        df_keep.groupby("month")["value"]
               .mean()
               .round(3)
               .reset_index()
               .sort_values("month")
    )
    out_csv = os.path.join(out_dir, "monthly_mean_dayweighted.csv")
    clim_dayweighted.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 6) 控制台预览
    print("\n=== 多年逐月径流量（日等权，已筛 year-month≥{} 天） ===".format(args.min_days))
    print(clim_dayweighted)
    print(f"\n结果文件：\n  {os.path.join(out_dir, 'ym_counts.csv')}\n  {os.path.join(out_dir, 'ym_kept_min20.csv')}\n  {out_csv}")


if __name__ == "__main__":
    main()
