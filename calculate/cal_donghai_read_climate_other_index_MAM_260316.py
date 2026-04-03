
from pathlib import Path
import pandas as pd


def extract_mam_and_yoy(input_csv: str, output_csv: str = None):
    """
    读取 atmosphere_index_3mon CSV：
    1. 提取每一年的 MAM 数据
    2. 计算 year-by-year variation（与上一年相比的差值）
    3. 另存为新文件

    输出结果中：
    - year: 年份
    - 原始数值列: 各指数的 MAM 值
    - *_yoy_change: 各指数相对上一年的变化量
    """

    input_csv = Path(input_csv)

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_MAM_yoy.csv")
    else:
        output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    # 解析日期
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 提取 MAM
    mam_df = df[df["season"].astype(str).str.upper() == "MAM"].copy()

    if mam_df.empty:
        raise ValueError("文件中未找到 season == 'MAM' 的记录。")

    # 提取年份并排序
    mam_df["year"] = mam_df["date"].dt.year
    mam_df = mam_df.sort_values("year").reset_index(drop=True)

    # 识别数值列：排除时间和 season 字段
    exclude_cols = {"date", "start_time", "end_time", "season", "year"}
    value_cols = [col for col in mam_df.columns if col not in exclude_cols]

    # 尝试转成数值
    for col in value_cols:
        mam_df[col] = pd.to_numeric(mam_df[col], errors="coerce")

    # 计算逐年变化（与上一年相比的差值）
    yoy_change_df = mam_df[value_cols].diff()
    yoy_change_df.columns = [f"{col}_yoy_change" for col in value_cols]

    # 如需同比百分比，可取消下面两行注释
    # yoy_pct_df = mam_df[value_cols].pct_change() * 100
    # yoy_pct_df.columns = [f"{col}_yoy_pct" for col in value_cols]

    # 合并输出
    out_df = pd.concat(
        [
            mam_df[["year", "date", "start_time", "end_time", "season"]],
            mam_df[value_cols],
            yoy_change_df,
            # yoy_pct_df,   # 如果需要同比百分比，把这一行也放开
        ],
        axis=1,
    )

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已输出文件: {output_csv}")
    print(f"MAM 年份数: {len(out_df)}")
    print("前5行预览：")
    print(out_df.head())


if __name__ == "__main__":
    input_csv = "/home/sun/wd_14/data/data/download_data/climate_index_national_climate_center/other_index_3mon.csv"
    output_csv = "/home/sun/wd_14/data/data/download_data/climate_index_national_climate_center/other_index_3mon-260316_MAM_yoy.csv"

    extract_mam_and_yoy(input_csv, output_csv)
