#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IBTrACS 西北太平洋(WP) 年度统计 + 两张图（CSV/Excel）

统计（按风暴SID计数）：
- TS_PLUS: 热带风暴及以上(>=34 kt) 年个数（排除热带低压 TD）
- TY_PLUS: 台风及以上(>=64 kt) 年个数
- SUPER:   超强台风(>=130 kt 默认) 年个数
- SUPER_RATIO_IN_TY_PLUS: 超强 / TY+ 比例（图2使用）

绘图（不加 title 和 legend）：
1) 图1：1961-2025 年 TS+ 个数折线图 + 最小二乘线性拟合直线
2) 图2：1990-2025 年 SUPER 在 TY+ 中占比折线图 + 最小二乘线性拟合直线

关键：强制只用 --wind-col 指定的风速列，避免两次运行结果不一致。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_input(path: Path, fmt: str | None, sheet=None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if fmt:
        fmt = fmt.lower().strip()

    if fmt is None:
        if suffix == ".csv":
            fmt = "csv"
        elif suffix in [".xlsx", ".xls"]:
            fmt = "excel"
        else:
            raise ValueError(f"无法从扩展名判断格式：{suffix}，请用 --format 指定 csv 或 excel")

    if fmt == "csv":
        return pd.read_csv(path, low_memory=False)
    if fmt == "excel":
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    raise ValueError(f"--format 仅支持 csv 或 excel，当前：{fmt}")


def norm_col(c: str) -> str:
    return str(c).strip().upper()


def find_col_exact(df: pd.DataFrame, wanted: str) -> str | None:
    w = norm_col(wanted)
    for c in df.columns:
        if norm_col(c) == w:
            return c
    return None


def find_col_any(df: pd.DataFrame, candidates) -> str | None:
    cand = {norm_col(x) for x in candidates}
    for c in df.columns:
        if norm_col(c) in cand:
            return c
    return None


def add_linear_fit(ax, x_years, y_values):
    """最小二乘线性拟合，健壮处理 NaN/pd.NA/object。"""
    x = pd.to_numeric(pd.Series(x_years), errors="coerce").astype(float).to_numpy()
    y = pd.to_numeric(pd.Series(y_values), errors="coerce").astype(float).to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return

    a, b = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = a * x_fit + b
    ax.plot(x_fit, y_fit, linestyle="--", linewidth=2)


def ensure_full_years(df_yearly: pd.DataFrame, start: int, end: int, cols: list[str]) -> pd.DataFrame:
    years = np.arange(start, end + 1)
    base = pd.DataFrame({"SEASON": years})
    return base.merge(df_yearly[["SEASON"] + cols], on="SEASON", how="left")


def main():
    parser = argparse.ArgumentParser(description="IBTrACS(WP) 年度统计 + 两张图（CSV/Excel）")
    parser.add_argument("input_file", help="IBTrACS CSV 或 Excel 文件路径")
    parser.add_argument("--format", default=None, help="强制指定格式：csv 或 excel（默认自动识别）")
    parser.add_argument("-s", "--sheet", default=None, help="Excel sheet名/索引（仅Excel有效）")

    parser.add_argument("--basin", default="WP", help="流域代码（默认WP）")
    parser.add_argument("--wind-col", default="USA_WIND",
                        help="风速列名（默认 USA_WIND）。脚本将强制只用该列。")

    parser.add_argument("--ts-threshold", type=float, default=34.0, help="TS+阈值（kt，默认34）")
    parser.add_argument("--ty-threshold", type=float, default=64.0, help="TY+阈值（kt，默认64）")
    parser.add_argument("--super-threshold", type=float, default=130.0, help="Super阈值（kt，默认130）")

    parser.add_argument("--plot1-start", type=int, default=1961)
    parser.add_argument("--plot1-end", type=int, default=2025)
    parser.add_argument("--plot2-start", type=int, default=1990)
    parser.add_argument("--plot2-end", type=int, default=2025)

    parser.add_argument("-o", "--output", default="wp_stats.xlsx", help="输出年度统计Excel")
    parser.add_argument("--fig1", default="fig1_tsplus_1961_2025.png", help="图1输出文件名")
    parser.add_argument("--fig2", default="fig2_super_ratio_in_typlus_1990_2025.png", help="图2输出文件名")

    args = parser.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        print(f"[ERROR] 输入文件不存在：{in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = read_input(in_path, args.format, sheet=args.sheet)
    except Exception as e:
        print(f"[ERROR] 读取文件失败：{e}", file=sys.stderr)
        sys.exit(1)

    sid_col = find_col_any(df, ["SID", "IBTRACS_ID", "STORM_ID"])
    season_col = find_col_any(df, ["SEASON", "YEAR"])
    basin_col = find_col_any(df, ["BASIN", "BASIN1", "BASIN_CODE"])
    wind_col = find_col_exact(df, args.wind_col)  # 强制只用指定列

    missing = []
    if sid_col is None: missing.append("SID")
    if season_col is None: missing.append("SEASON")
    if basin_col is None: missing.append("BASIN")
    if wind_col is None: missing.append(f"WIND({args.wind_col})")

    if missing:
        print("[ERROR] 缺少必要列：", ", ".join(missing), file=sys.stderr)
        print("       你的表头前60列：", list(df.columns)[:60], file=sys.stderr)
        print("       请用 --wind-col 指定真实风速列名（如 USA_WIND / WMO_WIND / CMA_WIND 等）。",
              file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Using wind column: {wind_col}")
    print(f"[INFO] thresholds: TS+={args.ts_threshold} kt, TY+={args.ty_threshold} kt, Super={args.super_threshold} kt")
    print(f"[INFO] basin filter: {args.basin}")

    work = df[[sid_col, season_col, basin_col, wind_col]].copy()
    work.rename(columns={sid_col: "SID", season_col: "SEASON", basin_col: "BASIN", wind_col: "WIND"}, inplace=True)

    work["SID"] = work["SID"].astype(str).str.strip()
    work["BASIN"] = work["BASIN"].astype(str).str.strip().str.upper()
    work["SEASON"] = pd.to_numeric(work["SEASON"], errors="coerce")
    work["WIND"] = pd.to_numeric(work["WIND"], errors="coerce")

    work = work.dropna(subset=["SEASON"])
    work["SEASON"] = work["SEASON"].astype(int)

    basin_code = str(args.basin).strip().upper()
    work = work[work["BASIN"] == basin_code].copy()
    if work.empty:
        print("[WARN] BASIN 过滤后为空：请检查 BASIN 是否确为 WP。", file=sys.stderr)
        sys.exit(0)

    storm = (
        work.groupby("SID", as_index=False)
            .agg(SEASON=("SEASON", "min"),
                 MAX_WIND=("WIND", "max"))
    )

    storm["IS_TS_PLUS"] = storm["MAX_WIND"] >= float(args.ts_threshold)
    storm["IS_TY_PLUS"] = storm["MAX_WIND"] >= float(args.ty_threshold)
    storm["IS_SUPER"] = storm["MAX_WIND"] >= float(args.super_threshold)

    yearly = (
        storm.groupby("SEASON", as_index=False)
             .agg(
                 TS_PLUS=("IS_TS_PLUS", "sum"),
                 TY_PLUS=("IS_TY_PLUS", "sum"),
                 SUPER=("IS_SUPER", "sum"),
             )
    )
    yearly["SUPER_RATIO_IN_TY_PLUS"] = yearly["SUPER"] / yearly["TY_PLUS"].replace({0: np.nan})
    yearly = yearly.sort_values("SEASON").reset_index(drop=True)

    # 输出表
    out_path = Path(args.output)
    csv_path = out_path.with_suffix(".csv")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        yearly.to_excel(writer, index=False, sheet_name="yearly_stats")
        storm.to_excel(writer, index=False, sheet_name="storm_level")
    yearly.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 图1：TS+ counts（无title/legend）
    y1_full = ensure_full_years(yearly, args.plot1_start, args.plot1_end, ["TS_PLUS"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y1_full["SEASON"].values, y1_full["TS_PLUS"].values, marker="o", linewidth=1.5)
    add_linear_fit(ax, y1_full["SEASON"].values, y1_full["TS_PLUS"].values)
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(args.fig1), dpi=200)
    plt.close(fig)

    # 图2：Super/TY+ ratio（无title/legend）
    y2_full = ensure_full_years(yearly, args.plot2_start, args.plot2_end, ["SUPER_RATIO_IN_TY_PLUS"])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y2_full["SEASON"].values, y2_full["SUPER_RATIO_IN_TY_PLUS"].values, marker="o", linewidth=1.5)
    add_linear_fit(ax, y2_full["SEASON"].values, y2_full["SUPER_RATIO_IN_TY_PLUS"].values)
    ax.set_xlabel("Year")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(args.fig2), dpi=200)
    plt.close(fig)

    print("完成 ✅")
    print(f"- 年度统计Excel：{out_path.resolve()}")
    print(f"- 年度统计CSV：{csv_path.resolve()}")
    print(f"- 图1：{Path(args.fig1).resolve()}")
    print(f"- 图2：{Path(args.fig2).resolve()}")


if __name__ == "__main__":
    main()