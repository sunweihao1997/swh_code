# -*- coding: utf-8 -*-
"""
功能：
1) 读取 Excel（默认取第一个工作表，包含列：站名、时间、水位、流量）
2) 将流量 < 100 视为缺测（NaN）
3) 按“站名 + 日期”求日平均（同时统计有效样本数/总样本数）
4) 输出到 Excel：大通径流量_日平均.xlsx（工作表：日平均）
"""

import pandas as pd
import numpy as np

IN_PATH = "/mnt/f/data_donghai/salinity/大通径流量.xlsx"          # 输入文件
OUT_PATH = "/mnt/f/data_donghai/salinity/大通径流量_日平均.xlsx"   # 输出文件

def main():
    # 读取
    df = pd.read_excel(IN_PATH, sheet_name=0)
    # 基本检查
    expected = {"站名", "时间", "水位", "流量"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {missing}；需要包含列 {expected}")

    # 时间 → 日期
    df["时间"] = pd.to_datetime(df["时间"], errors="coerce") #如果遇到无法解析成时间格式的数据（比如空字符串、无效日期），就把它强制转换为 NaT（Not a Time，表示缺失时间）而不是报错。
    df["日期"] = df["时间"].dt.date

    # 流量 < 100 视为缺测
    df["流量_用于统计"] = df["流量"].where(df["流量"] >= 100, np.nan)

    # 分组求日平均（按站名+日期）
    daily = (
        df.groupby(["站名", "日期"], as_index=False)
          .agg(
              水位日平均=("水位", "mean"),
              流量日平均=("流量_用于统计", "mean"),
              有效流量样本数=("流量_用于统计", lambda x: x.notna().sum()),
              原始样本数=("流量", "size"),
          )
    )

    # 美化数值
    daily["水位日平均"] = daily["水位日平均"].round(3)
    daily["流量日平均"] = daily["流量日平均"].round(3)

    # 导出
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        daily.to_excel(writer, sheet_name="日平均", index=False)

    print(f"已生成：{OUT_PATH}")

if __name__ == "__main__":
    main()
